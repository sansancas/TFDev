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
    use_final_attention=True
):
    """
    Híbrido CNN + BiRNN + (MHA entre y final) + SE + FiLM (Keras 3 safe).
    """
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

    # --- Atención final + pooling por ventana (si corresponde) ---
    if use_final_attention:
        if time_step:
            attn = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=max(8, rnn_units // num_heads), name="mha_final"
            )(x, x)
            x = layers.Add(name="add_mha_final")([x, attn])
            x = layers.LayerNormalization(name="ln_mha_final")(x)
        else:
            attn = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=max(8, rnn_units // num_heads), name="mha_final"
            )(x, x)
            xf = layers.Add(name="add_mha_final")([x, attn])
            xf = layers.LayerNormalization(name="ln_mha_final")(xf)
            gap = layers.GlobalAveragePooling1D(name="gap")(xf)
            ap  = AttentionPooling1D(name="attnpool")(xf)
            x   = layers.Concatenate(name="pool_concat")([gap, ap])
            x   = layers.Dropout(dropout_rate, name="drop_head")(x)

    # --- Cabezas de salida ---
    if time_step:
        x = layers.TimeDistributed(layers.Dense(64, activation="relu"), name="td_fc")(x)
        if feat_input_dim is not None and feat_input_dim > 0:
            # Modulación adicional por FiLM (manteniendo el tiempo)
            x = FiLM1D(channels=x.shape[-1], name="film_head_ts")(x, feat_in)
            inputs = [Inp, feat_in]
        else:
            inputs = Inp

        if one_hot:
            Out = layers.TimeDistributed(
                layers.Dense(num_classes, activation="softmax"), name="softmax_ts"
            )(x)  # (B,T',C)
        else:
            Out = layers.TimeDistributed(
                layers.Dense(1, activation="sigmoid"), name="sigmoid_ts"
            )(x)  # (B,T',1)

    else:
        if feat_input_dim is not None and feat_input_dim > 0:
            x = layers.Dense(128, activation="relu", name="fc_win")(x)
            x = layers.Dropout(dropout_rate, name="drop_win")(x)
            # expandimos (B,C)->(B,1,C) sin usar tf.newaxis
            x = layers.Lambda(lambda t: K.expand_dims(t, axis=1), name="expand_win")(x)
            x = FiLM1D(channels=x.shape[-1], name="film_head_win")(x, feat_in)
            x = layers.Reshape((-1,), name="flatten_win")(x)
            inputs = [Inp, feat_in]
        else:
            x = layers.Dense(128, activation="relu", name="fc_win")(x)
            x = layers.Dropout(dropout_rate, name="drop_win")(x)
            inputs = Inp

        if one_hot:
            Out = layers.Dense(num_classes, activation="softmax", name="softmax")(x)
        else:
            Out = layers.Dense(1, activation="sigmoid", name="sigmoid")(x)

    return models.Model(inputs=inputs, outputs=Out, name="Hybrid_CNN_BiRNN_MHA_SE_FiLM")