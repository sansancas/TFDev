from tensorflow.keras import layers, models

def se_block_1d(x, se_ratio=16, name=None):
    """
    Squeeze-and-Excitation block for 1D inputs.
    x: Tensor, shape (batch, time, channels)
    se_ratio: reduction ratio
    """
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling1D(name=name+"_se_squeeze")(x)
    se = layers.Dense(max(channels // se_ratio, 1), activation='relu',
                      name=name+"_se_reduce")(se)
    se = layers.Dense(channels, activation='sigmoid',
                      name=name+"_se_expand")(se)
    se = layers.Reshape((1, channels), name=name+"_se_reshape")(se)
    return layers.Multiply(name=name+"_se_scale")([x, se])


def build_hybrid(
    input_shape,                  # (T, C) e.g., (256,22)
    num_classes=2,
    num_filters=64,
    kernel_size=7,
    dropout_rate=0.25,
    one_hot=True,                 # softmax (C) vs sigmoid (1)
    time_step=True,              # per-timestep output vs per-window
    se_position='after_conv',     # None, 'after_conv', 'after_fc'
    attention_position='final',   # None, 'between_lstm', 'final'
    rnn_type='lstm',              # 'lstm' or 'gru'
    se_ratio=16,
    num_heads=4,
    pool_size_if_ts=1,            # use 1 if time_step=True (keeps time length)
    pool_size_if_win=2,           # your original 2 for window-level
    feat_input_dim: int | None = None
):
    """
    DSCNN + 2xRNN hybrid with optional S.E. and MHA.
    time_step=False → window-level: (B, 1) or (B, C)
    time_step=True  → time-step:   (B, T', 1) or (B, T', C)
    NOTE: if time_step=True and you keep any temporal downsampling (pool_size>1),
          your labels must be downsampled by the same factor in the input pipeline.
    """
    def se_block_1d(x, se_ratio=16, name="se"):
        ch = x.shape[-1]
        s = layers.GlobalAveragePooling1D(name=f"{name}_squeeze")(x)
        s = layers.Dense(max(1, ch // se_ratio), activation="relu", name=f"{name}_reduce")(s)
        s = layers.Dense(ch, activation="sigmoid", name=f"{name}_expand")(s)
        s = layers.Reshape((1, ch), name=f"{name}_reshape")(s)
        return layers.Multiply(name=f"{name}_scale")([x, s])

    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs

    # --- Conv front-end ---
    x = layers.SeparableConv1D(num_filters, kernel_size, padding='same', activation='relu', name='sep_conv1')(x)
    if se_position == 'after_conv':
        x = se_block_1d(x, se_ratio=se_ratio, name='se_after_conv')

    # Choose pool size based on time_step mode
    pool_size = pool_size_if_ts if time_step else pool_size_if_win
    x = layers.MaxPooling1D(pool_size=pool_size, name='pool1')(x)

    # --- Dense along feature dim (applied per time step if 3D) ---
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    if se_position == 'after_fc':
        x = se_block_1d(x, se_ratio=se_ratio, name='se_after_fc')
    x = layers.Dropout(dropout_rate, name='drop1')(x)

    # --- RNN 1 ---
    RNN = layers.GRU if rnn_type.lower() == 'gru' else layers.LSTM
    x = RNN(64, return_sequences=True, name=('gru1' if rnn_type.lower()=='gru' else 'lstm1'))(x)

    # --- Optional attention between RNNs ---
    if attention_position == 'between_lstm':
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64 // num_heads, name='mha_between')(x, x)
        x = layers.Add(name='add_mha_between')([x, attn])
        x = layers.LayerNormalization(name='ln_mha_between')(x)

    # --- RNN 2 ---
    # If we need time-step outputs OR final attention, we must keep sequences
    return_seq_for_second = time_step or (attention_position == 'final')
    x = RNN(64, return_sequences=return_seq_for_second, name=('gru2' if rnn_type.lower()=='gru' else 'lstm2'))(x)

    # --- Optional attention at the end ---
    if attention_position == 'final':
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64 // num_heads, name='mha_final')(x, x)
        x = layers.Add(name='add_mha_final')([x, attn])
        x = layers.LayerNormalization(name='ln_mha_final')(x)
        if not time_step:
            # window-level: collapse time here
            x = layers.GlobalAveragePooling1D(name='gap_attn_final')(x)

    # --- Head ---
    if time_step:
        # keep time axis → use TimeDistributed head
        x = layers.TimeDistributed(layers.Dense(64, activation='relu'), name='td_fc2')(x)
        if feat_input_dim is not None and feat_input_dim > 0:
            f_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
            f_proj = layers.Dense(64, activation='relu', name='feat_proj')(f_in)
            f_rep = layers.RepeatVector(input_shape[0], name='feat_repeat')(f_proj)
            x = layers.Concatenate(name='concat_ts')([x, f_rep])
            head_inputs = [inputs, f_in]
        else:
            head_inputs = inputs
        if one_hot:
            outputs = layers.TimeDistributed(
                layers.Dense(num_classes, activation='softmax'),
                name='softmax_ts'
            )(x)  # (B, T', C)
        else:
            outputs = layers.TimeDistributed(
                layers.Dense(1, activation='sigmoid'),
                name='output_ts'
            )(x)  # (B, T', 1)
    else:
        # window-level head (your original style)
        x = layers.Dense(64, activation='relu', name='fc2')(x)
        if feat_input_dim is not None and feat_input_dim > 0:
            f_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
            f_proj = layers.Dense(64, activation='relu', name='feat_proj')(f_in)
            x = layers.Concatenate(name='concat_win')([x, f_proj])
            head_inputs = [inputs, f_in]
        else:
            head_inputs = inputs
        if one_hot:
            outputs = layers.Dense(num_classes, activation='softmax', name='softmax')(x)  # (B, C)
        else:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)            # (B, 1)

    return models.Model(inputs=head_inputs, outputs=outputs, name='DSCNN-2RNN-Hybrid')