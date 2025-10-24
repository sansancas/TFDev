from tensorflow.keras import layers, models, activations
import tensorflow as tf
from keras import ops as K

# ---------- Bloques utilitarios ----------
def gelu(x):
    return activations.gelu(x) 

def se_block_1d(x, se_ratio=16, name="se"):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D(name=f"{name}_squeeze")(x)
    s = layers.Dense(max(1, ch // se_ratio), activation="relu", name=f"{name}_reduce")(s)
    s = layers.Dense(ch, activation="sigmoid", name=f"{name}_expand")(s)
    s = layers.Reshape((1, ch), name=f"{name}_reshape")(s)
    return layers.Multiply(name=f"{name}_scale")([x, s])

@tf.keras.utils.register_keras_serializable()
class FiLM1D(layers.Layer):
    def __init__(self, channels, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.dg = layers.Dense(channels, name=(name or "film")+"_g")
        self.db = layers.Dense(channels, name=(name or "film")+"_b")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config
    
    def call(self, x, f):
        # x: (B,T,C), f: (B,F)
        g = self.dg(f)[:, tf.newaxis, :]
        b = self.db(f)[:, tf.newaxis, :]
        return g * x + b

@tf.keras.utils.register_keras_serializable()
class AttentionPooling1D(layers.Layer):
    """ Atención sobre tiempo: w=softmax(Dense(1)), salida (B,C). """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.score = layers.Dense(1, name=(name or "attnpool") + "_score")
    
    def get_config(self):
        config = super().get_config()
        return config
    
    def call(self, x):
        w = self.score(x)                     # (B,T,1)
        w = tf.nn.softmax(w, axis=1)
        return tf.reduce_sum(w * x, axis=1)   # (B,C)

@tf.keras.utils.register_keras_serializable()
class AddScalarMSELoss(layers.Layer):
    """Layer that adds weight * mean(square(x)) to the model loss and passes x through.
    Use it to register auxiliary losses (Koopman/Recon) without calling model.add_loss.
    """
    def __init__(self, weight: float, name: str | None = None):
        super().__init__(name=name, trainable=False)
        self.weight = float(weight)


    def call(self, x):
        if self.weight > 0.0:
            loss = tf.reduce_mean(tf.square(x))
            self.add_loss(self.weight * loss)
        return x

@tf.keras.utils.register_keras_serializable()
class DiversityPenalty(layers.Layer):
    """Graph-safe diversity penalty between multiple head logits.
    - Input: list/tuple of tensors with shapes (B,C) or (B,T,C).
    - Computes cosine-similarity matrix between heads per sample and penalizes off-diagonals.
    - RETURNS a real tensor: zeros_like(first_head), so Keras op output is a tensor (no specs).
    - Also calls add_loss(weight * mean(offdiag(sim^2))).
    """
    def __init__(self, weight: float = 1e-2, name: str | None = None):
        super().__init__(name=name, trainable=False)
        self.weight = float(weight)

    def call(self, tensors, **kwargs):
        # Ensure list/tuple of tensors
        if not isinstance(tensors, (list, tuple)):
            return tf.zeros_like(tensors)
        vecs = []
        for t in tensors:
            t = tf.cast(t, tf.float32)
            t = tf.reshape(t, [tf.shape(t)[0], -1])   # (B,D)
            t = tf.math.l2_normalize(t, axis=-1)
            vecs.append(t)
        if (len(vecs) >= 2) and (self.weight > 0.0):
            V = tf.stack(vecs, axis=1)                     # (B,H,D)
            sim = tf.matmul(V, V, transpose_b=True)        # (B,H,H)
            h = tf.shape(sim)[-1]
            eye = tf.eye(h, batch_shape=[tf.shape(sim)[0]])
            off = sim * (1.0 - eye)
            loss = tf.reduce_mean(tf.square(off))
            self.add_loss(self.weight * loss)
        # Return a dummy tensor with same batch/shape as first head
        first = tensors[0]
        return tf.zeros_like(first)
    
# ---------- Modelo principal ----------
def build_hybrid(
    input_shape,                  # (T, C)
    num_classes=2,
    one_hot=True,                 # softmax (C) vs sigmoid (1)
    time_step=True,               # salida por frame vs por ventana
    conv_type="conv",             # "conv" o "separable"
    num_filters=64,
    kernel_size=7,
    se_ratio=16,
    dropout_rate=0.25,
    num_heads=4,
    rnn_units=64,
    feat_input_dim: int | None = None,
    use_se_after_cnn=True,
    use_se_after_rnn=True,
    use_between_attention=True,
    use_final_attention=True,
    # ===== NUEVO: Koopman head =====
    koopman_latent_dim: int = 0,         # 0 = desactivado
    koopman_loss_weight: float = 0.0,    # e.g., 0.1
    # ===== NUEVO: Reconstrucción (AE) =====
    use_reconstruction_head: bool = False,
    recon_weight: float = 0.0,           # e.g., 0.05
    recon_target: str = "signal",        # "signal" (por ahora)
    # ===== NUEVO: Bottleneck / Expansión =====
    bottleneck_dim: int | None = None,
    expand_dim: int | None = None,
    # === Multi-head extras ===
    diversity_weight: float = 5e-3):

    Inp = layers.Input(shape=input_shape, name="input")        # (T,C)
    x = Inp

    # --- Front-end CNN (elige tipo) ---
    Conv = layers.Conv1D if conv_type == "conv" else layers.SeparableConv1D
    x = Conv(num_filters, kernel_size, padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation(gelu, name="gelu1")(x)

    x = Conv(num_filters, kernel_size, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation(gelu, name="gelu2")(x)

    if use_se_after_cnn:
        x = se_block_1d(x, se_ratio=se_ratio, name="se_after_cnn")

    # --- FiLM temprano (opcional) ---
    try:
        if isinstance(feat_input_dim, (tuple, list)):
            feat_input_dim = feat_input_dim[0]
        if feat_input_dim is not None:
            feat_input_dim = int(feat_input_dim)
    except Exception:
        feat_input_dim = None

    feat_in = None
    if feat_input_dim is not None and feat_input_dim > 0:
        feat_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
        x = FiLM1D(channels=x.shape[-1], name="film_after_cnn")(x, feat_in)

    # --- RNN 1: Bidireccional, mantiene secuencia ---
    x = layers.Bidirectional(
        layers.LSTM(rnn_units, return_sequences=True), name="bilstm1"
    )(x)
    x = layers.LayerNormalization(name="ln_after_bilstm1")(x)
    x = layers.Dropout(dropout_rate, name="drop_after_bilstm1")(x)

    # --- Atención "entre" (opcional) ---
    if use_between_attention:
        attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=max(8, rnn_units // num_heads), name="mha_between"
        )(x, x)
        x = layers.Add(name="add_mha_between")([x, attn])
        x = layers.LayerNormalization(name="ln_mha_between")(x)

    # --- RNN 2 ---
    return_seq_2 = time_step or use_final_attention
    x = layers.LSTM(rnn_units, return_sequences=return_seq_2, name="lstm2")(x)
    if return_seq_2:
        x = layers.LayerNormalization(name="ln_after_lstm2")(x)

    if use_se_after_rnn and return_seq_2:
        x = se_block_1d(x, se_ratio=se_ratio, name="se_after_rnn")

    # === Secuencia latente para heads auxiliares ===
    seq_for_aux = x if return_seq_2 else None  # (B,T,C') o None si ya se colapsó

    # --- Atención final + pooling por ventana (si corresponde) ---
    if use_final_attention:
        if time_step:
            attn = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=max(8, rnn_units // num_heads), name="mha_final"
            )(x, x)
            x = layers.Add(name="add_mha_final")([x, attn])
            x = layers.LayerNormalization(name="ln_mha_final")(x)
            seq_for_aux = x
        else:
            attn = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=max(8, rnn_units // num_heads), name="mha_final"
            )(x, x)
            xf = layers.Add(name="add_mha_final")([x, attn])
            xf = layers.LayerNormalization(name="ln_mha_final")(xf)
            seq_for_aux = xf
            gap = layers.GlobalAveragePooling1D(name="gap")(xf)
            ap  = AttentionPooling1D(name="attnpool")(xf)
            x   = layers.Concatenate(name="pool_concat")([gap, ap])
            x   = layers.Dropout(dropout_rate, name="drop_head")(x)

    # ===== Koopman regularizer (opcional) =====
    if (seq_for_aux is not None) and (koopman_latent_dim and koopman_latent_dim > 0 and koopman_loss_weight > 0.0):
        latent_seq = layers.Conv1D(rnn_units, 1, padding="same", name="latent_proj")(seq_for_aux)
        latent_seq = layers.LayerNormalization(name="latent_ln")(latent_seq)
        z_seq = layers.Dense(koopman_latent_dim, name="koop_z")(latent_seq)  # (B,T,dk)
        z_t  = layers.Lambda(lambda t: t[:, :-1, :], name="koop_t")(z_seq)
        z_tp = layers.Lambda(lambda t: t[:, 1:,  :], name="koop_tp")(z_seq)
        A = layers.Dense(koopman_latent_dim, use_bias=False, name="koop_A")
        z_pred = A(z_t)
        diff = layers.Subtract(name="koop_diff")([z_tp, z_pred])
        _ = AddScalarMSELoss(koopman_loss_weight, name="koop_loss")(diff)

    # ===== Reconstrucción (autoencoder ligero) =====
    if (seq_for_aux is not None) and (use_reconstruction_head and recon_weight > 0.0 and recon_target == "signal"):
        latent_seq = layers.Conv1D(num_filters, 1, padding="same", name="recon_latent_proj")(seq_for_aux)
        latent_seq = layers.LayerNormalization(name="recon_latent_ln")(latent_seq)
        dec = latent_seq
        dec = layers.Conv1D(num_filters, 3, padding="same", activation=gelu, name="recon_c1")(dec)
        dec = layers.Conv1D(max(1, num_filters//2), 3, padding="same", activation=gelu, name="recon_c2")(dec)
        x_rec = layers.Conv1D(input_shape[-1], 1, padding="same", name="recon_out")(dec)  # (B,T,Cin)
        rdiff = layers.Subtract(name="recon_diff")([Inp, x_rec])
        _ = AddScalarMSELoss(recon_weight, name="recon_loss")(rdiff)

    # ===== HEAD-T (temporal) =====
    if time_step:
        hT = layers.TimeDistributed(layers.Dense(64, activation="relu"), name="td_fc")(x)
        if bottleneck_dim:
            hT = layers.Conv1D(bottleneck_dim, 1, padding="same", activation="relu", name="bneck_ts")(hT)
            if expand_dim:
                hT = layers.Conv1D(expand_dim, 1, padding="same", activation="relu", name="expand_ts")(hT)
        if feat_in is not None:
            hT = FiLM1D(channels=hT.shape[-1], name="film_head_ts")(hT, feat_in)
        logits_T = layers.Conv1D(num_classes, 1, padding="same", name="headT_logits")(hT)  # (B,T,C)
    else:
        # window level
        base = x
        if feat_in is not None:
            base = layers.Dense(128, activation="relu", name="fc_win")(base)
            base = layers.Dropout(dropout_rate, name="drop_win")(base)
            base = layers.Lambda(lambda t: K.expand_dims(t, axis=1), name="expand_win")(base)
            base = FiLM1D(channels=base.shape[-1], name="film_head_win")(base, feat_in)
            base = layers.Reshape((-1,), name="flatten_win")(base)
        else:
            base = layers.Dense(128, activation="relu", name="fc_win")(base)
            base = layers.Dropout(dropout_rate, name="drop_win")(base)
        if bottleneck_dim:
            base = layers.Dense(bottleneck_dim, activation="relu", name="bneck_win")(base)
            if expand_dim:
                base = layers.Dense(expand_dim, activation="relu", name="expand_win")(base)
        logits_T = layers.Dense(num_classes, name="headT_logits")(base)  # (B,C)

    # ===== HEAD-F (features) =====
    logits_F = None
    if feat_in is not None:
        f = layers.Dense(num_filters, activation="relu", name="headF_fc1")(feat_in)
        f = layers.Dense(num_filters, activation="relu", name="headF_fc2")(f)
        logits_F = layers.Dense(num_classes, name="headF_logits")(f)  # (B,C)

    # ===== HEAD-K (Koopman-derived) =====
    logits_K = None
    if koopman_latent_dim and koopman_latent_dim > 0 and seq_for_aux is not None:
        z_base = layers.Conv1D(num_filters, 1, padding="same", name="headK_proj")(seq_for_aux)
        z_base = layers.LayerNormalization(name="headK_ln")(z_base)
        if time_step:
            logits_K = layers.Conv1D(num_classes, 1, padding="same", name="headK_logits")(z_base)  # (B,T,C)
        else:
            z_pool = layers.GlobalAveragePooling1D(name="headK_gap")(z_base)
            logits_K = layers.Dense(num_classes, name="headK_logits")(z_pool)  # (B,C)

    # ===== FUSIÓN + Diversidad =====
    heads = [logits_T]
    if logits_F is not None:
        heads.append(logits_F)
    if logits_K is not None:
        heads.append(logits_K)

    if time_step:
        def _to_time(t):
            if len(t.shape) == 3:
                return t
            return layers.Lambda(lambda u: tf.tile(tf.reshape(u, (-1,1,tf.shape(u)[-1])), [1, tf.shape(logits_T)[1], 1]))(t)
        heads_T = [_to_time(hh) for hh in heads]
        fused_logits = layers.Average(name="fuse_logits")(heads_T)
        # heads_T = [logits_T, logits_F(→expandido a T), logits_K]  todos (B,T,C)
        _ = DiversityPenalty(weight=diversity_weight, name="div_heads")(heads_T)
        fused_logits = layers.Average(name="fuse_logits")(heads_T)

        Out = layers.Softmax(name="softmax_ts")(fused_logits) if one_hot else layers.Activation("sigmoid", name="sigmoid_ts")(fused_logits)
        inputs = [Inp, feat_in] if feat_in is not None else Inp
    else:
        def _to_win(t):
            if len(t.shape) == 2:
                return t
            return layers.GlobalAveragePooling1D()(t)
        heads_W = [_to_win(hh) for hh in heads]
        fused_logits = layers.Average(name="fuse_logits")(heads_W)
       # heads_W = [logits_T, logits_F, logits_K] ya alineados a (B,C)
        _ = DiversityPenalty(weight=diversity_weight, name="div_heads")(heads_W)
        fused_logits = layers.Average(name="fuse_logits")(heads_W)

        Out = layers.Activation("softmax", name="softmax")(fused_logits) if one_hot else layers.Activation("sigmoid", name="sigmoid")(fused_logits)
        inputs = [Inp, feat_in] if feat_in is not None else Inp

    return models.Model(inputs=inputs, outputs=Out, name="Hybrid_Multihead_CNN_BiRNN_MHA_SE_FiLM")