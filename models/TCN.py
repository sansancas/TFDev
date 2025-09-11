from tensorflow.keras import layers, models

def build_tcn(input_shape,
              num_classes=2,
              num_filters=68,
              kernel_size=7,
              dropout_rate=0.25,
              num_blocks=7,
              time_step_classification=True,
              one_hot=True,
              hpc=False,
              separable=False,
              feat_input_dim: int | None = None):
    
    # define the input layer
    dtype = 'float32' if hpc else None
    inputs = layers.Input(shape=input_shape, dtype=dtype)
    x = inputs

    # build TCN blocks
    for i in range(num_blocks):
        dilation = 2 ** i
        residual = x

        # first causal conv + norm + dropout
        if separable:
            x = layers.SeparableConv1D(filters=num_filters,
                                       kernel_size=kernel_size,
                                       dilation_rate=dilation,
                                       padding='same',
                                       depth_multiplier=1,
                                       use_bias=False,
                                       name=f"sepconv1_block{i+1}")(x)
        else:
            x = layers.Conv1D(filters=num_filters,
                          kernel_size=kernel_size,
                          dilation_rate=dilation,
                          padding='causal',
                          kernel_initializer='he_normal',
                          name=f"conv2_block{i+1}")(x)
        x = layers.LayerNormalization(name=f"ln1_block{i+1}")(x)
        x = layers.SpatialDropout1D(rate=dropout_rate,
                                    name=f"drop1_block{i+1}")(x)

        # second causal conv + norm + relu + dropout
        if separable:
            x = layers.SeparableConv1D(filters=num_filters,
                                       kernel_size=kernel_size,
                                       dilation_rate=dilation,
                                       padding='same',
                                       depth_multiplier=1,
                                       use_bias=False,
                                       name=f"sepconv2_block{i+1}")(x)
        else:
            x = layers.Conv1D(filters=num_filters,
                          kernel_size=kernel_size,
                          dilation_rate=dilation,
                          padding='causal',
                          kernel_initializer='he_normal',
                          name=f"conv1_block{i+1}")(x)
        
        x = layers.LayerNormalization(name=f"ln2_block{i+1}")(x)
        x = layers.Activation('relu', name=f"relu_block{i+1}")(x)
        x = layers.SpatialDropout1D(rate=dropout_rate,
                                    name=f"drop2_block{i+1}")(x)

        # skip connection
        if i == 0:
            if separable:
                skip = layers.SeparableConv1D(filters=num_filters,
                                              kernel_size=1,
                                              padding='same',
                                              depth_multiplier=1,
                                              use_bias=False,
                                              name="sepconv_skip")(residual)
            else:
                skip = layers.Conv1D(filters=num_filters,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 name="conv_skip")(residual)
            x = layers.Add(name=f"add_block{i+1}")([x, skip])
        else:
            x = layers.Add(name=f"add_block{i+1}")([x, residual])

    # classification head (+ optional feature fusion)
    if time_step_classification:
        # Time-step: if feat_input exists, broadcast and fuse
        if feat_input_dim is not None and feat_input_dim > 0:
            f_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
            f_proj = layers.Dense(64, activation='relu', name='feat_proj')(f_in)
            f_rep = layers.RepeatVector(input_shape[0], name='feat_repeat')(f_proj)  # broadcast to T length
            x = layers.Concatenate(name='concat_ts')([x, f_rep])
            head_inputs = [inputs, f_in]
        else:
            head_inputs = inputs
        x = layers.Dense(units=num_classes, kernel_initializer='he_normal', name="fc")(x)
    else:
        x = layers.GlobalAveragePooling1D(name="gap")(x)
        if feat_input_dim is not None and feat_input_dim > 0:
            f_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
            f_proj = layers.Dense(64, activation='relu', name='feat_proj')(f_in)
            x = layers.Concatenate(name='concat_win')([x, f_proj])
            head_inputs = [inputs, f_in]
        else:
            head_inputs = inputs
        x = layers.Dense(units=num_classes, kernel_initializer='he_normal', name="fc")(x)
    if one_hot:
        outputs = layers.Softmax(name="softmax")(x)
    else:
        outputs =  layers.Dense(1, activation='sigmoid', name='output')(x)
    model = models.Model(inputs=head_inputs, outputs=outputs, name="tcn_eegnet")
    return model