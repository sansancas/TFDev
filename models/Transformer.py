import tensorflow as tf
from tensorflow.keras import layers, models
from models.Hybrid import se_block_1d
 
def positional_encoding_1d(sequence_length, d_model):
    """
    Builds a 1D sinusoidal positional encoding layer.
    Returns a (1, sequence_length, d_model) tensor.
    """
    # positions: shape (seq_len, 1)
    positions = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
    # i: shape (1, d_model)
    i = tf.range(d_model, dtype=tf.int32)[tf.newaxis, :]
    # angle rates: shape (1, d_model)
    angle_rates = 1.0 / tf.pow(
        10000.0,
        tf.cast(2 * (i // 2), tf.float32) / tf.cast(d_model, tf.float32)
    )
    # angle radians: shape (seq_len, d_model)
    angle_rads = positions * angle_rates

    # apply sin to even indices; cos to odd indices
    sines = tf.sin(angle_rads[:, 0::2])
    coses = tf.cos(angle_rads[:, 1::2])

    # concat the sines and coses back to (seq_len, d_model)
    pos_encoding = tf.concat([sines, coses], axis=-1)
    # add batch dimension: (1, seq_len, d_model)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return pos_encoding

def transformer_encoder(x,
                        head_dim,
                        num_heads,
                        mlp_dim,
                        dropout_rate=0.1,
                        name=None):
    # multi-head self-attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_dim,
        dropout=dropout_rate,
        name=name+"_mha"
    )(x, x)
    attn = layers.Dropout(dropout_rate, name=name+"_attn_dropout")(attn)
    out1 = layers.Add(name=name+"_attn_add")([x, attn])
    out1 = layers.LayerNormalization(epsilon=1e-6, name=name+"_attn_ln")(out1)

    # feedforward network
    mlp = layers.Dense(mlp_dim, activation='relu', name=name+"_mlp_fc1")(out1)
    mlp = layers.Dropout(dropout_rate, name=name+"_mlp_dropout")(mlp)
    mlp = layers.Dense(x.shape[-1], name=name+"_mlp_fc2")(mlp)
    out2 = layers.Add(name=name+"_mlp_add")([out1, mlp])
    out2 = layers.LayerNormalization(epsilon=1e-6, name=name+"_mlp_ln")(out2)
    return out2

def build_transformer(input_shape,
                      num_classes=2,
                      embed_dim=128,
                      num_layers=4,
                      num_heads=4,
                      mlp_dim=256,
                      dropout_rate=0.1,
                      time_step_classification=True,
                      one_hot=True,
                      hpc=False,
                      use_se=False,
                      se_ratio=16,
                      feat_input_dim: int | None = None):
    # Input
    dtype = 'float32' if hpc else None
    inputs = layers.Input(shape=input_shape, dtype=dtype, name='input')  # (seq_len, features)
    x = inputs

    # optional SE on raw features
    if use_se:
        x = se_block_1d(x, se_ratio=se_ratio, name='se_input')

    # linear projection to embedding dim
    x = layers.Dense(embed_dim, name='proj')(x)
    # add positional encoding
    pos_enc = positional_encoding_1d(input_shape[0], embed_dim)
    x = layers.Add(name='pos_add')([x, pos_enc])
    x = layers.Dropout(dropout_rate, name='proj_dropout')(x)

    # Transformer encoder stack
    head_dim = embed_dim // num_heads
    for i in range(num_layers):
        if use_se and i == 0:
            x = se_block_1d(x, se_ratio=se_ratio, name=f'se_pre_enc{i+1}')
        x = transformer_encoder(x,
                                head_dim=head_dim,
                                num_heads=num_heads,
                                mlp_dim=mlp_dim,
                                dropout_rate=dropout_rate,
                                name=f'encoder{i+1}')
        if use_se and i == num_layers-1:
            x = se_block_1d(x, se_ratio=se_ratio, name=f'se_post_enc{i+1}')

    # Classification head
    if time_step_classification:
        if feat_input_dim is not None and feat_input_dim > 0:
            f_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
            f_proj = layers.Dense(64, activation='relu', name='feat_proj')(f_in)
            f_rep = layers.RepeatVector(input_shape[0], name='feat_repeat')(f_proj)
            x = layers.Concatenate(name='concat_ts')([x, f_rep])
            head_inputs = [inputs, f_in]
        else:
            head_inputs = inputs
        logits = layers.Dense(num_classes, name='fc')(x)
    else:
        x = layers.GlobalAveragePooling1D(name='gap')(x)
        if feat_input_dim is not None and feat_input_dim > 0:
            f_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
            f_proj = layers.Dense(64, activation='relu', name='feat_proj')(f_in)
            x = layers.Concatenate(name='concat_win')([x, f_proj])
            head_inputs = [inputs, f_in]
        else:
            head_inputs = inputs
        logits = layers.Dense(num_classes, name='fc')(x)
        
    if one_hot:
        outputs = layers.Softmax(name="softmax")(logits)
    else:
        outputs =  layers.Dense(1, activation='sigmoid', name='output')(logits)

    return models.Model(inputs=head_inputs, outputs=outputs, name='transformer_seizure')