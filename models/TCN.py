import tensorflow as tf
from tensorflow.keras import layers, models

class CausalPadding1D(layers.Layer):
    """Capa de padding causal compatible con Keras Functional API"""
    
    def __init__(self, padding_size, **kwargs):
        super().__init__(**kwargs)
        self.padding_size = int(padding_size)
    
    def call(self, x):
        return tf.pad(x, [[0, 0], [self.padding_size, 0], [0, 0]])
    
    def compute_output_shape(self, input_shape):
        b, t, c = input_shape
        t_out = None if t is None else t + self.padding_size
        return (b, t_out, c)
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'padding_size': self.padding_size})
        return cfg

class SqueezeExcitation1D(layers.Layer):
    """Squeeze-and-Excitation ligero compatible con XLA/JIT"""
    
    def __init__(self, channels, se_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.se_ratio = se_ratio
        self.hidden = max(1, channels // se_ratio)
        
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(self.hidden, activation='relu', use_bias=False)
        self.fc2 = layers.Dense(channels, activation='sigmoid', use_bias=False)
        self.reshape = layers.Reshape((1, channels))
        self.mul = layers.Multiply()
    
    def call(self, x):
        # Squeeze
        s = self.pool(x)
        # Excitation
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.reshape(s)
        # Scale
        return self.mul([x, s])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'se_ratio': self.se_ratio
        })
        return config

class TemporalAlignment(layers.Layer):
    """Capa para alinear temporalmente tensores en conexiones residuales"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        x, residual = inputs
        # Obtener la longitud mínima
        x_len = tf.shape(x)[1]
        res_len = tf.shape(residual)[1]
        min_len = tf.minimum(x_len, res_len)
        
        # Recortar a la longitud mínima
        x_aligned = x[:, :min_len, :]
        res_aligned = residual[:, :min_len, :]
        
        return x_aligned, res_aligned

def build_tcn(input_shape,
              num_classes=2,
              num_filters=68,
              kernel_size=7,
              dropout_rate=0.25,
              num_blocks=8,
              time_step=True,
              one_hot=True,
              hpc=False,
              separable=True,
              use_squeeze_excitation=False,
              use_gelu=True):
    
    if one_hot and num_classes != 2:
        raise ValueError("one_hot=True requiere num_classes=2.")
    if not one_hot and num_classes != 1:
        raise ValueError("one_hot=False requiere num_classes=1.")

    # define the input layer
    dtype = 'float32' if hpc else None
    inputs = layers.Input(shape=input_shape, dtype=dtype)
    x = inputs

    # Activación a usar
    activation = 'gelu' if use_gelu else 'relu'

    # build TCN blocks
    for i in range(num_blocks):
        dilation = 2 ** i
        residual = x

        # first causal conv + norm + dropout
        if separable:
            # Padding causal manual para SeparableConv1D usando capa personalizada
            pad_len = dilation * (kernel_size - 1)
            x = CausalPadding1D(pad_len, name=f"pad1_block{i+1}")(x)
            x = layers.SeparableConv1D(filters=num_filters,
                                       kernel_size=kernel_size,
                                       dilation_rate=dilation,
                                       padding='valid',  # Cambiado de 'same' a 'valid'
                                       depth_multiplier=1,
                                       use_bias=False,
                                       depthwise_initializer='he_normal',
                                       pointwise_initializer='he_normal',
                                       name=f"sepconv1_block{i+1}")(x)
        else:
            x = layers.Conv1D(filters=num_filters,
                          kernel_size=kernel_size,
                          dilation_rate=dilation,
                          padding='causal',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name=f"conv2_block{i+1}")(x)
        x = layers.LayerNormalization(dtype="float32", name=f"ln1_block{i+1}")(x)
        x = layers.SpatialDropout1D(rate=dropout_rate,
                                    name=f"drop1_block{i+1}")(x)

        # second causal conv + norm + activation + dropout
        if separable:
            # Padding causal manual para SeparableConv1D usando capa personalizada
            pad_len = dilation * (kernel_size - 1)
            x = CausalPadding1D(pad_len, name=f"pad2_block{i+1}")(x)
            x = layers.SeparableConv1D(filters=num_filters,
                                       kernel_size=kernel_size,
                                       dilation_rate=dilation,
                                       padding='valid',  # Cambiado de 'same' a 'valid'
                                       depth_multiplier=1,
                                       use_bias=False,
                                       depthwise_initializer='he_normal',
                                       pointwise_initializer='he_normal',
                                       name=f"sepconv2_block{i+1}")(x)
        else:
            x = layers.Conv1D(filters=num_filters,
                          kernel_size=kernel_size,
                          dilation_rate=dilation,
                          padding='causal',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name=f"conv1_block{i+1}")(x)
        
        x = layers.LayerNormalization(dtype="float32", name=f"ln2_block{i+1}")(x)
        x = layers.Activation(activation, name=f"{activation}_block{i+1}")(x)
        x = layers.SpatialDropout1D(rate=dropout_rate,
                                    name=f"drop2_block{i+1}")(x)

        # Squeeze-and-Excitation opcional
        if use_squeeze_excitation:
            x = SqueezeExcitation1D(num_filters, name=f"se_block{i+1}")(x)

        # skip connection con ajuste de longitud temporal
        if i == 0:
            if separable:
                skip = layers.SeparableConv1D(filters=num_filters,
                                              kernel_size=1,
                                              padding='same',
                                              depth_multiplier=1,
                                              use_bias=False,
                                              depthwise_initializer='he_normal',
                                              pointwise_initializer='he_normal',
                                              name="sepconv_skip")(residual)
            else:
                skip = layers.Conv1D(filters=num_filters,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 use_bias=False,
                                 name="conv_skip")(residual)
        else:
            skip = residual
        
        # Ajustar longitudes para conexión residual usando capa personalizada
        # x_aligned, skip_aligned = TemporalAlignment(name=f"align_block{i+1}")([x, skip])
        # x = layers.Add(name=f"add_block{i+1}")([x_aligned, skip_aligned])
        x = layers.Add(name=f"add_block{i+1}")([x, skip])

    # classification head
    if time_step:
        # Frame-by-frame classification
        x = layers.Dense(units=num_classes,
                         kernel_initializer='he_normal',
                         use_bias=False,
                         name="fc")(x)
        if one_hot:
            outputs = layers.Softmax(name="softmax")(x)
        else:
            # Para clasificación binaria en modo time_step
            outputs = layers.Dense(1, activation='sigmoid', 
                                 kernel_initializer='he_normal',
                                 name='output')(x)
    else:
        # Window-level classification
        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dense(units=num_classes,
                         kernel_initializer='he_normal',
                         use_bias=False,
                         name="fc")(x)
        if one_hot:
            outputs = layers.Softmax(name="softmax")(x)
        else:
            # Para clasificación binaria en modo window
            outputs = layers.Dense(1, activation='sigmoid',
                                 kernel_initializer='he_normal', 
                                 name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="tcn_eegnet")
    
    # Calcular y mostrar información del modelo incluyendo campo receptivo
    total_params = model.count_params()
    receptive_field = calculate_receptive_field(num_blocks, kernel_size)
    
    print(f"🧠 TCN Enhanced creada:")
    print(f"├── Parámetros totales: {total_params:,}")
    print(f"├── Campo receptivo: {receptive_field} muestras ({receptive_field/256:.2f}s a 256Hz)")
    print(f"├── Filtros: {num_filters}")
    print(f"├── Bloques: {num_blocks}")
    print(f"├── Separable: {separable}")
    print(f"├── Squeeze-Excitation: {use_squeeze_excitation}")
    print(f"├── Activación: {activation}")
    print(f"├── Modo: {'Frame-by-frame' if time_step else 'Por ventana'}")
    print(f"└── Salida: {'One-hot' if one_hot else 'Binario'}")
    
    return model

def calculate_receptive_field(num_blocks, kernel_size):
    """
    Calcula el campo receptivo total de la TCN
    
    Para TCN con dilataciones exponenciales: 2^0, 2^1, 2^2, ...
    RF = 1 + sum((kernel_size - 1) * dilation_rate for each block)
    
    Args:
        num_blocks: Número de bloques TCN
        kernel_size: Tamaño del kernel de convolución
    
    Returns:
        int: Campo receptivo total en muestras
    """
    receptive_field = 1
    for i in range(num_blocks):
        dilation_rate = 2 ** i
        receptive_field += (kernel_size - 1) * dilation_rate
    
    # Nota: Cada bloque TCN tiene 2 convoluciones, pero la segunda 
    # no aumenta el campo receptivo debido al residual connection
    return receptive_field

def analyze_receptive_field_for_eeg(sample_rate=256):
    """
    Analiza diferentes configuraciones de TCN para determinar 
    el campo receptivo óptimo para detección de convulsiones EEG
    
    Args:
        sample_rate: Frecuencia de muestreo en Hz (default: 256)
    """
    print("📊 Análisis de Campo Receptivo para Detección de Convulsiones EEG")
    print("=" * 70)
    print(f"Frecuencia de muestreo: {sample_rate} Hz")
    print("")
    
    configs = [
        {'blocks': 3, 'kernel': 7, 'name': 'Básica (rápida)'},
        {'blocks': 4, 'kernel': 7, 'name': 'Estándar'},
        {'blocks': 5, 'kernel': 7, 'name': 'Mejorada'},
        {'blocks': 6, 'kernel': 7, 'name': 'Avanzada'},
        {'blocks': 7, 'kernel': 7, 'name': 'Completa'},
        {'blocks': 4, 'kernel': 11, 'name': 'Kernel grande'},
        {'blocks': 8, 'kernel': 3, 'name': 'Muchos bloques'},
    ]
    
    print(f"{'Configuración':<20} | {'RF (muestras)':<12} | {'Tiempo (s)':<10} | {'Recomendación'}")
    print("-" * 70)
    
    for config in configs:
        rf = calculate_receptive_field(config['blocks'], config['kernel'])
        time_seconds = rf / sample_rate
        
        # Determinar recomendación basada en literatura EEG
        if time_seconds < 1.0:
            recommendation = "⚠️  Muy pequeño"
        elif time_seconds < 2.0:
            recommendation = "🔶 Mínimo"
        elif time_seconds < 5.0:
            recommendation = "✅ Óptimo"
        elif time_seconds < 8.0:
            recommendation = "🟡 Grande"
        else:
            recommendation = "🔴 Excesivo"
        
        print(f"{config['name']:<20} | {rf:<12} | {time_seconds:<10.2f} | {recommendation}")
    
    print("\n🧠 Recomendaciones para Convulsiones EEG:")
    print("├── ⚠️  < 1s: Insuficiente para capturar patrones pre-ictales")
    print("├── 🔶 1-2s: Mínimo para detección básica")
    print("├── ✅ 2-5s: Óptimo para balance precisión/eficiencia")
    print("├── 🟡 5-8s: Puede capturar más contexto pero riesgo overfitting")
    print("└── 🔴 > 8s: Excesivo, probable overfitting y alta latencia")
    
    print(f"\n💡 Para tu aplicación:")
    print(f"├── Tiempo objetivo: 2-4 segundos ({2*sample_rate}-{4*sample_rate} muestras)")
    print(f"├── Configuración recomendada: 7 bloques, kernel=7")
    print(f"└── Campo receptivo resultante: ~{calculate_receptive_field(7, 7)} muestras ({calculate_receptive_field(7, 7)/sample_rate:.2f}s)")
    
    return calculate_receptive_field(7, 7)  # Retornar configuración recomendada