import tensorflow as tf
from tensorflow.keras import layers, models, activations, initializers
from keras import ops as K

def se_block_1d(x, se_ratio=16, name="se"):
    """SE block 1D implementado directamente para evitar imports circulares."""
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D(name=f"{name}_squeeze")(x)
    s = layers.Dense(max(1, ch // se_ratio), activation="relu", name=f"{name}_reduce")(s)
    s = layers.Dense(ch, activation="sigmoid", name=f"{name}_expand")(s)
    s = layers.Reshape((1, ch), name=f"{name}_reshape")(s)
    return layers.Multiply(name=f"{name}_scale")([x, s])

# ---------- Utilidades ----------
def gelu(x):
    return activations.gelu(x)

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

def rotary_embedding(q, k):
    """
    RoPE con keras.ops (backend-agnostic).
    q,k: (B,H,T,Hd) con Hd par
    """
    hd = q.shape[-1]
    if hd is None or (hd % 2) != 0:
        raise ValueError("head_dim debe ser par y estático para RoPE.")
    T = K.shape(q)[-2]
    dim = hd

    pos = K.arange(T)                          # (T,)
    pos = K.reshape(pos, (1, 1, T, 1))         # (1,1,T,1)
    idx = K.arange(dim // 2)
    idx = K.reshape(idx, (1, 1, 1, -1))        # (1,1,1,Hd/2)

    inv_freq = 1.0 / (10000.0 ** (idx / (dim / 2.0)))  # (1,1,1,Hd/2)
    angles = pos * inv_freq                              # (1,1,T,Hd/2)
    sin = K.sin(angles)
    cos = K.cos(angles)

    def rot(x):
        x1, x2 = K.split(x, 2, axis=-1)  # (..,Hd/2) y (..,Hd/2)
        return K.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    return rot(q), rot(k)

@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttentionRoPE(layers.Layer):
    """MHA con RoPE usando keras.ops (sin tf.* crudo)."""
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        assert embed_dim % num_heads == 0, "embed_dim % num_heads == 0"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = attn_dropout
        base = (name or "mha")
        self.wq = layers.Dense(embed_dim, name=f"{base}_wq")
        self.wk = layers.Dense(embed_dim, name=f"{base}_wk")
        self.wv = layers.Dense(embed_dim, name=f"{base}_wv")
        self.wo = layers.Dense(embed_dim, name=f"{base}_wo")
        self.dropout = layers.Dropout(attn_dropout)
        self.sm = layers.Softmax(axis=-1, name=f"{base}_softmax")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "attn_dropout": self.attn_dropout,
        })
        return config

    def _split_heads(self, x):
        # (B,T,E)->(B,H,T,Hd) con K.reshape/K.transpose
        B = K.shape(x)[0]
        T = K.shape(x)[1]
        x = K.reshape(x, (B, T, self.num_heads, self.head_dim))
        return K.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x):
        # (B,H,T,Hd)->(B,T,E)
        B = K.shape(x)[0]
        T = K.shape(x)[2]
        x = K.transpose(x, (0, 2, 1, 3))
        return K.reshape(x, (B, T, self.embed_dim))

    def call(self, x, training=None, mask=None):
        q = self._split_heads(self.wq(x))
        k = self._split_heads(self.wk(x))
        v = self._split_heads(self.wv(x))

        q, k = rotary_embedding(q, k)
        scale = (self.head_dim ** -0.5)
        logits = K.matmul(q, K.transpose(k, (0,1,3,2))) * scale  # (B,H,T,T)

        if mask is not None:
            # (B,T) o (B,1,1,T) -> (B,1,1,T)
            if K.ndim(mask) == 2:
                mask = K.expand_dims(K.expand_dims(mask, axis=1), axis=1)
            # logits += (1-mask)*(-1e9) con ops
            logits = logits + (1.0 - K.cast(mask, "float32")) * (-1e9)

        attn = self.sm(logits)
        attn = self.dropout(attn, training=training)
        y = K.matmul(attn, v)              # (B,H,T,Hd)
        y = self._combine_heads(y)         # (B,T,E)
        return self.wo(y)

@tf.keras.utils.register_keras_serializable()
class AttentionPooling1D(layers.Layer):
    """Atención temporal ligera (backend-safe)."""
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        base = (name or "attnpool")
        self.score = layers.Dense(1, name=f"{base}_score")
        self.sm = layers.Softmax(axis=1, name=f"{base}_softmax")
    
    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        w = self.score(x)     # (B,T,1)
        w = self.sm(w)
        return K.sum(w * x, axis=1)  # (B,C)

@tf.keras.utils.register_keras_serializable()
class AddCLSToken(layers.Layer):
    """Inserta un token [CLS] entrenable al inicio (B,T,E)->(B,T+1,E)."""
    def __init__(self, embed_dim, name="cls", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
        })
        return config

    def build(self, input_shape):
        # Peso entrenable (1,1,E)
        self.cls = self.add_weight(
            name="token",
            shape=(1, 1, self.embed_dim),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, x):
        b = K.shape(x)[0]
        cls_batched = K.tile(self.cls, (b, 1, 1))
        return K.concatenate([cls_batched, x], axis=1)

# ---------- Encoder Transformer con RoPE ----------
def transformer_block_rope(x, embed_dim, num_heads, mlp_dim, dropout_rate, name):
    attn = MultiHeadSelfAttentionRoPE(embed_dim, num_heads, attn_dropout=dropout_rate, name=name+"_mha")(x)
    attn = layers.Dropout(dropout_rate, name=name+"_attn_dropout")(attn)
    x = layers.Add(name=name+"_attn_add")([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6, name=name+"_attn_ln")(x)

    y = layers.Dense(mlp_dim, activation=gelu, name=name+"_mlp_fc1")(x)
    y = layers.Dropout(dropout_rate, name=name+"_mlp_dropout")(y)
    y = layers.Dense(embed_dim, name=name+"_mlp_fc2")(y)
    x = layers.Add(name=name+"_mlp_add")([x, y])
    x = layers.LayerNormalization(epsilon=1e-6, name=name+"_mlp_ln")(x)
    return x

# ---------- Modelo principal ----------
def build_transformer(
    input_shape,                 # (T, C)  -> T tiempo, C canales (o features por canal ya apilados)
    num_classes=2,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    mlp_dim=256,
    dropout_rate=0.1,
    time_step_classification=True,  # True: frame-level, False: window-level
    one_hot=True,
    use_se=False,     # si quieres enchufar tu se_block_1d
    se_ratio=16,
    feat_input_dim=None  # dim de features contextuales por ventana/sesión
):
    inp = layers.Input(shape=input_shape, name="input")
    x = inp

    # Front-end (separable) con BN+GELU
    x = layers.SeparableConv1D(64, 7, strides=2, padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation(gelu, name="gelu1")(x)

    x = layers.SeparableConv1D(128, 7, strides=2, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation(gelu, name="gelu2")(x)

    if use_se:
        x = se_block_1d(x, se_ratio=se_ratio, name="se_after_cnn")

    # Proyección a embedding
    x = layers.Dense(embed_dim, name="proj")(x)
    x = layers.Dropout(dropout_rate, name="proj_dropout")(x)

    # Token [CLS] como capa
    x = AddCLSToken(embed_dim, name="cls")(x)

    # Features contextuales vía FiLM (opcional)
    feat_inp = None
    if feat_input_dim is not None and feat_input_dim > 0:
        feat_inp = layers.Input(shape=(feat_input_dim,), name="feat_input")
        x = FiLM1D(channels=embed_dim, name="film0")(x, feat_inp)

    # Pila Transformer con RoPE (+FiLM opcional)
    for i in range(num_layers):
        x = transformer_block_rope(x, embed_dim, num_heads, mlp_dim, dropout_rate, name=f"encoder{i+1}")
        if feat_inp is not None:
            x = FiLM1D(embed_dim, name=f"film{i+1}")(x, feat_inp)
        if use_se and i in (0, num_layers-1):
            x = se_block_1d(x, se_ratio=se_ratio, name=f"se_enc_{i+1}")

    # --- Cabezas ---
    if time_step_classification:
        # Slicing seguro: Lambda (evita posibles rarezas con []
        x_frames = layers.Lambda(lambda t: t[:, 1:, :], name="slice_drop_cls")(x)
        logits = layers.Dense(num_classes, name="fc_frames")(x_frames)
        if one_hot:
            out = layers.Softmax(name="softmax_ts")(logits)
        else:
            out = layers.Dense(1, activation='sigmoid', name='sigmoid_ts')(logits)
    else:
        cls_token = layers.Lambda(lambda t: t[:, 0, :],  name="pick_cls")(x)
        body      = layers.Lambda(lambda t: t[:, 1:, :], name="pick_body")(x)
        attn_pool = AttentionPooling1D(name="attnpool")(body)
        h = layers.Concatenate(name="win_head_concat")([cls_token, attn_pool])
        h = layers.Dropout(dropout_rate, name="win_head_drop")(h)
        logits = layers.Dense(num_classes, name="fc_window")(h)
        if one_hot:
            out = layers.Softmax(name="softmax_win")(logits)
        else:
            out = layers.Dense(1, activation='sigmoid', name='sigmoid_win')(logits)

    inputs = [inp] if feat_inp is None else [inp, feat_inp]
    return models.Model(inputs=inputs, outputs=out, name="eeg_transformer_rope_film")