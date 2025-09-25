from tensorflow.keras import layers, models, activations
from keras import ops as K   # <- Keras 3 ops backend-agnostic
import tensorflow as tf

def gelu(x):
    return activations.gelu(x) 

def se_block_1d(x, se_ratio=16, name="se"):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D(name=f"{name}_sq")(x)
    s = layers.Dense(max(1, ch // se_ratio), activation="relu", name=f"{name}_rd")(s)
    s = layers.Dense(ch, activation="sigmoid", name=f"{name}_ex")(s)
    s = layers.Reshape((1, ch), name=f"{name}_rs")(s)
    return layers.Multiply(name=f"{name}_sc")([x, s])

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

def causal_sepconv1d(x, filters, kernel_size, dilation_rate, name):
    """Separable causal: pad left + valid."""
    pad = (kernel_size - 1) * dilation_rate
    x = layers.ZeroPadding1D(padding=(pad, 0), name=f"{name}_pad")(x)
    x = layers.SeparableConv1D(filters=filters,
                               kernel_size=kernel_size,
                               dilation_rate=dilation_rate,
                               padding="valid",
                               depth_multiplier=1,
                               use_bias=False,
                               name=name)(x)
    return x

def gated_res_block(x, filters, kernel_size, dilation, separable, se_ratio, name):
    """
    Bloque residual 'gated' tipo WaveNet:
    - (Conv tanh) ⊙ (Conv sigmoid) -> z
    - Proyección 1x1 a filtros (residual) y a filtros (skip)
    - SE opcional sobre la rama residual
    Devuelve (x_residual, x_skip)
    """
    inp = x
    # Conv(·) para filtro y puerta
    if separable:
        a = causal_sepconv1d(x, filters, kernel_size, dilation, name=f"{name}_a")
        b = causal_sepconv1d(x, filters, kernel_size, dilation, name=f"{name}_b")
    else:
        a = layers.Conv1D(filters, kernel_size, dilation_rate=dilation,
                          padding="causal", kernel_initializer="he_normal",
                          name=f"{name}_a")(x)
        b = layers.Conv1D(filters, kernel_size, dilation_rate=dilation,
                          padding="causal", kernel_initializer="he_normal",
                          name=f"{name}_b")(x)
    a = layers.LayerNormalization(name=f"{name}_ln_a")(a)
    b = layers.LayerNormalization(name=f"{name}_ln_b")(b)
    z = activations.tanh(a) * activations.sigmoid(b)              # gating

    # SE en la rama intermedia (opcional)
    z = se_block_1d(z, se_ratio=se_ratio, name=f"{name}_se")

    # proyecciones residual y skip
    res = layers.Conv1D(filters, 1, padding="same", name=f"{name}_res")(z)
    skip = layers.Conv1D(filters, 1, padding="same", name=f"{name}_skip")(z)

    # alinear canales de la entrada si difiere
    if inp.shape[-1] != filters:
        inp = layers.Conv1D(filters, 1, padding="same", name=f"{name}_inproj")(inp)

    out = layers.Add(name=f"{name}_add")([inp, res])  # residual
    out = layers.SpatialDropout1D(0.1, name=f"{name}_drop")(out)
    return out, skip

def build_tcn(input_shape,
                 num_classes=2,
                 num_filters=64,
                 kernel_size=7,
                 dropout_rate=0.25,
                 num_blocks=8,
                 time_step_classification=True,
                 one_hot=True,
                 hpc=False,
                 separable=False,
                 se_ratio=16,
                 cycle_dilations=(1,2,4,8),
                 feat_input_dim: int | None = None,
                 use_attention_pool_win=True):
    """
    Mejora sobre tu TCN:
    - Bloques residuales 'gated' + skip global acumulado
    - Causalidad estricta también en separables
    - SE por bloque
    - FiLM para fusionar features
    - Cycling de dilataciones
    - Attention pooling opcional en ventana
    """
    dtype = 'float32' if hpc else None
    Inp = layers.Input(shape=input_shape, dtype=dtype, name="input")
    x = Inp

    # FiLM temprano si hay features globales
    feat_in = None
    if feat_input_dim is not None and feat_input_dim > 0:
        feat_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
        x = FiLM1D(channels=input_shape[-1], name="film_in")(x, feat_in)

    skips = []
    # pila de bloques
    for i in range(num_blocks):
        dilation = cycle_dilations[i % len(cycle_dilations)]
        x, s = gated_res_block(x, num_filters, kernel_size, dilation,
                               separable, se_ratio, name=f"blk{i+1}")
        skips.append(s)

    # fusión de skips (WaveNet-like)
    s_sum = layers.Add(name="skip_sum")(skips) if len(skips) > 1 else skips[0]
    s_sum = layers.Activation(gelu, name="skip_gelu")(s_sum)
    s_sum = layers.LayerNormalization(name="skip_ln")(s_sum)
    s_sum = layers.SpatialDropout1D(dropout_rate, name="skip_drop")(s_sum)

    # cabeza
    if time_step_classification:
        h = layers.Conv1D(num_filters, 1, padding="same", name="head_ts_proj")(s_sum)
        h = layers.LayerNormalization(name="head_ts_ln")(h)
        if feat_input_dim is not None and feat_input_dim > 0:
            # Modulación tardía adicional
            h = FiLM1D(channels=num_filters, name="film_ts")(h, feat_in)
        logits = layers.Conv1D(num_classes, 1, padding="same", name="fc_ts")(h)
        if one_hot:
            Out = layers.Softmax(name="softmax_ts")(logits)
        else:
            Out = layers.Activation("sigmoid", name="sigmoid_ts")(logits)
        inputs = [Inp, feat_in] if feat_in is not None else Inp

    else:
        # pooling por ventana
        # combinación GAP + attention pooling (ligero)
        xf = s_sum
        gap = layers.GlobalAveragePooling1D(name="gap")(xf)
        if use_attention_pool_win:
            w = layers.Dense(1, name="attn_score")(xf)
            w = layers.Softmax(axis=1, name="attn_sm")(w)
            ap = K.sum(w * xf, axis=1)  # (B,C)
            h = layers.Concatenate(name="pool_concat")([gap, ap])
        else:
            h = gap

        h = layers.Dropout(dropout_rate, name="head_drop")(h)
        if feat_input_dim is not None and feat_input_dim > 0:
            # FiLM también puede aplicarse aquí: proyectamos y modulamos
            h = layers.Dense(num_filters, activation=gelu, name="head_fc")(h)
            h = layers.Dropout(dropout_rate, name="head_fc_drop")(h)
            # para FiLM necesitamos (B,T,C): extendemos T=1, modulamos y volvemos a aplanar
            h = layers.Reshape((1, num_filters), name="head_rs")(h)
            h = FiLM1D(channels=num_filters, name="film_win")(h, feat_in)
            h = layers.Reshape((num_filters,), name="head_flat")(h)
            inputs = [Inp, feat_in]
        else:
            h = layers.Dense(num_filters, activation=gelu, name="head_fc")(h)
            h = layers.Dropout(dropout_rate, name="head_fc_drop")(h)
            inputs = Inp

        logits = layers.Dense(num_classes, name="fc")(h)
        if one_hot:
            Out = layers.Activation("softmax", name="softmax")(logits)
        else:
            Out = layers.Activation("sigmoid", name="sigmoid_win")(logits)

    return models.Model(inputs=inputs, outputs=Out, name="tcn_eeg_v2")