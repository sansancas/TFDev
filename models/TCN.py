import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# ====================================================================================
# TCN OPTIMIZADA PARA DETECCIÓN DE CONVULSIONES EEG - TENSORFLOW/KERAS
# ====================================================================================

class CausalConv1D(layers.Layer):
    """Convolución causal mejorada con padding adaptativo"""
    
    def __init__(self, filters, kernel_size, dilation_rate=1, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.pad = (kernel_size - 1) * dilation_rate
        
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='valid',
            use_bias=use_bias,
            kernel_initializer='he_normal'
        )
    
    def call(self, x):
        # Padding causal manual
        x = tf.pad(x, [[0, 0], [self.pad, 0], [0, 0]])
        return self.conv(x)

class SeparableConv1D(layers.Layer):
    """SeparableConv1D optimizada para EEG con mejor eficiencia"""
    
    def __init__(self, filters, kernel_size, dilation_rate=1, use_bias=False, 
                 depth_multiplier=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.depth_multiplier = depth_multiplier
        self.pad = dilation_rate * (kernel_size - 1)
        
        # Depthwise convolution
        self.depthwise = layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            dilation_rate=dilation_rate,
            padding='valid',
            use_bias=use_bias,
            depthwise_initializer='he_normal'
        )
        
        # Pointwise convolution
        self.pointwise = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            use_bias=use_bias,
            kernel_initializer='he_normal'
        )
        
        # BatchNorm para estabilidad en EEG
        self.bn_dw = layers.BatchNormalization()
        self.bn_pw = layers.BatchNormalization()
    
    def call(self, x, training=None):
        # Padding causal
        x = tf.pad(x, [[0, 0], [self.pad, 0], [0, 0]])
        
        # Depthwise convolution
        x = self.depthwise(x)
        x = self.bn_dw(x, training=training)
        x = tf.nn.relu(x)
        
        # Pointwise convolution
        x = self.pointwise(x)
        x = self.bn_pw(x, training=training)
        
        return x

class SqueezeExcitation1D(layers.Layer):
    """Squeeze-and-Excitation optimizado para señales EEG"""
    
    def __init__(self, channels, se_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.se_ratio = se_ratio
        self.hidden = max(1, channels // se_ratio)
        
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(self.hidden, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        self.dropout = layers.Dropout(0.1)
        self.reshape = layers.Reshape((1, channels))
        self.multiply = layers.Multiply()
    
    def call(self, x, training=None):
        # Squeeze
        s = self.pool(x)
        
        # Excitation
        s = self.fc1(s)
        s = self.dropout(s, training=training)
        s = self.fc2(s)
        s = self.reshape(s)
        
        # Scale
        return self.multiply([x, s])

class EnhancedTCNBlock(layers.Layer):
    """Bloque TCN mejorado para detección de convulsiones EEG"""
    
    def __init__(self, channels, kernel_size, dilation_rate, dropout=0.25,
                 separable=True, use_se=True, use_residual_scaling=True, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.separable = separable
        self.use_se = use_se
        self.use_residual_scaling = use_residual_scaling
        
        # Capas convolutivas
        if separable:
            self.conv1 = SeparableConv1D(channels, kernel_size, dilation_rate, use_bias=False)
            self.conv2 = SeparableConv1D(channels, kernel_size, dilation_rate, use_bias=False)
        else:
            self.conv1 = CausalConv1D(channels, kernel_size, dilation_rate, use_bias=False)
            self.conv2 = CausalConv1D(channels, kernel_size, dilation_rate, use_bias=False)
        
        # Normalización y dropout
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        
        # Squeeze-and-Excitation
        if use_se:
            self.se = SqueezeExcitation1D(channels)
        
        # Residual scaling
        if use_residual_scaling:
            self.residual_scale = self.add_weight(
                name='residual_scale',
                shape=(1,),
                initializer='zeros',
                trainable=True
            )
            # Inicializar con 0.1
            self.residual_scale.assign([0.1])
        
        # Projection layer para dilataciones altas
        self.projection = None
        if dilation_rate > 4:
            self.projection = layers.Conv1D(channels, 1, use_bias=False, kernel_initializer='he_normal')
    
    def call(self, x, training=None):
        residual = x
        
        # Primera convolución
        x = self.conv1(x, training=training) if self.separable else self.conv1(x)
        x = self.ln1(x, training=training)
        x = tf.keras.activations.gelu(x)  # GELU en lugar de ReLU
        x = self.drop1(x, training=training)
        
        # Segunda convolución
        x = self.conv2(x, training=training) if self.separable else self.conv2(x)
        x = self.ln2(x, training=training)
        x = tf.keras.activations.gelu(x)
        x = self.drop2(x, training=training)
        
        # Squeeze-and-Excitation
        if self.use_se:
            x = self.se(x, training=training)
        
        # Ajustar longitudes para residual connection
        if x.shape[1] != residual.shape[1]:
            T = min(x.shape[1], residual.shape[1])
            x = x[:, :T, :]
            residual = residual[:, :T, :]
        
        # Proyección del residual si es necesario
        if self.projection is not None:
            residual = self.projection(residual)
        
        # Conexión residual con escalado
        if self.use_residual_scaling:
            return residual + self.residual_scale * x
        else:
            return residual + x

def build_optimized_seizure_tcn(input_dim=22,
                               num_classes=2,
                               num_filters=96,
                               kernel_size=7,
                               num_blocks=8,
                               time_step=True,
                               one_hot=True,
                               separable=True,
                               dropout=0.3,
                               use_se=True,
                               use_residual_scaling=True,
                               use_adaptive_dilation=True,
                               use_multiscale=True,
                               class_weights=None):
    """
    TCN optimizada específicamente para detección de convulsiones EEG
    ✅ FIXED: Maneja correctamente one_hot y binary modes
    """
    
    # Input layer
    inputs = layers.Input(shape=(None, input_dim), name='input')
    x = inputs
    
    # Proyección de entrada mejorada con normalización
    x = layers.Dense(num_filters, kernel_initializer='he_normal', name='in_fc')(x)
    x = layers.LayerNormalization(name='in_ln')(x)
    x = layers.Activation('gelu', name='in_gelu')(x)
    x = layers.Dropout(0.1, name='in_dropout')(x)
    
    # Bloques TCN con dilataciones adaptativas
    if use_adaptive_dilation:
        # Dilataciones que crecen más lentamente para mejor captura temporal
        dilations = [2**min(i, 6) for i in range(num_blocks)]  # Cap at 64
    else:
        dilations = [2**i for i in range(num_blocks)]
    
    for i, dilation in enumerate(dilations):
        x = EnhancedTCNBlock(
            channels=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout=dropout,
            separable=separable,
            use_se=use_se,
            use_residual_scaling=use_residual_scaling,
            name=f'tcn_block_{i+1}'
        )(x)
    
    # Características multi-escala opcionales
    if use_multiscale:
        # Diferentes escalas temporales
        multiscale_features = []
        for k in [3, 5, 7, 11]:
            ms_feat = layers.Conv1D(
                num_filters//4, 
                kernel_size=k, 
                padding='same',
                activation='relu',
                name=f'multiscale_conv_{k}'
            )(x)
            multiscale_features.append(ms_feat)
        
        # Concatenar y fusionar
        ms_concat = layers.Concatenate(axis=-1, name='multiscale_concat')(multiscale_features)
        ms_fused = layers.Conv1D(num_filters, 1, name='multiscale_fusion')(ms_concat)
        x = layers.Add(name='multiscale_residual')([x, ms_fused])
    
    # ✅ FIXED: Cabeza de clasificación que maneja correctamente ambos modos
    if time_step:
        # Predicción frame-by-frame
        x = layers.Dense(num_filters//2, kernel_initializer='he_normal', name='head_dense1')(x)
        x = layers.LayerNormalization(name='head_ln')(x)
        x = layers.Activation('gelu', name='head_gelu')(x)
        x = layers.Dropout(dropout, name='head_dropout')(x)
        
        # ✅ CRITICAL FIX: Output shape basado en el modo
        if one_hot:
            # One-hot: output (batch, time, 2) para 2 clases
            x = layers.Dense(num_classes, kernel_initializer='he_normal', name='head_output')(x)
            outputs = layers.Softmax(name='softmax')(x)
        else:
            # Binary: output (batch, time, 1) para clasificación binaria
            outputs = layers.Dense(1, activation='sigmoid', name='head_output')(x)
    else:
        # ✅ FIXED: Predicción por ventana - NO TEMPORAL
        x_avg = layers.GlobalAveragePooling1D(name='gap')(x)
        x_max = layers.GlobalMaxPooling1D(name='gmp')(x)
        x_combined = layers.Concatenate(name='pool_concat')([x_avg, x_max])
        
        x_combined = layers.Dense(num_filters, kernel_initializer='he_normal', name='head_dense1')(x_combined)
        x_combined = layers.LayerNormalization(name='head_ln')(x_combined)
        x_combined = layers.Activation('gelu', name='head_gelu')(x_combined)
        x_combined = layers.Dropout(dropout, name='head_dropout')(x_combined)
        
        # ✅ CRITICAL FIX: Output shape basado en el modo - NO TEMPORAL
        if one_hot:
            # One-hot: output (batch, 2) para 2 clases
            x_combined = layers.Dense(num_classes, kernel_initializer='he_normal', name='head_dense2')(x_combined)
            outputs = layers.Softmax(name='softmax')(x_combined)
        else:
            # Binary: output (batch, 1) para clasificación binaria
            outputs = layers.Dense(1, activation='sigmoid', name='head_output')(x_combined)
    
    # Crear modelo
    model = models.Model(inputs=inputs, outputs=outputs, name='optimized_seizure_tcn')
    
    # Información del modelo
    total_params = model.count_params()
    receptive_field = calculate_receptive_field(num_blocks, kernel_size, use_adaptive_dilation)
    
    print(f"🧠 OptimizedSeizureTCN (TensorFlow) creada:")
    print(f"├── Parámetros totales: {total_params:,}")
    print(f"├── Campo receptivo: {receptive_field} muestras ({receptive_field/256:.2f}s)")
    print(f"├── Canales entrada: {input_dim}")
    print(f"├── Filtros: {num_filters}")
    print(f"├── Bloques TCN: {num_blocks}")
    print(f"├── Modo: {'Frame-by-frame' if time_step else 'Por ventana'}")
    print(f"└── Salida: {'One-hot' if one_hot else 'Binario'}")
    
    return model

def calculate_receptive_field(num_blocks, kernel_size, use_adaptive_dilation=True):
    """Calcula el campo receptivo total de la red"""
    rf = 1
    for i in range(num_blocks):
        if use_adaptive_dilation:
            dilation = 2**min(i, 6)
        else:
            dilation = 2**i
        rf += (kernel_size - 1) * dilation
    return rf

# Funciones de utilidad adicionales
def create_seizure_tcn(input_channels=22, window_length_samples=1280, **kwargs):
    """Factory function para crear TCN optimizada para convulsiones"""
    
    # ✅ FIXED: Configuración que detecta automáticamente el modo correcto
    default_config = {
        'input_dim': input_channels,
        'num_classes': 2 if kwargs.get('one_hot', True) else 1,  # Auto-detect classes
        'num_filters': 32,  # Reducido para evitar overfitting
        'kernel_size': 7,
        'num_blocks': 4,    # Reducido para ser más eficiente
        'time_step': kwargs.get('time_step', True),
        'one_hot': kwargs.get('one_hot', True),
        'dropout': 0.3,
        'use_se': True,
        'use_multiscale': False,  # Disabled por defecto para simplicidad
        'separable': True,
        'use_adaptive_dilation': True,
        'use_residual_scaling': True
    }
    
    # Actualizar con parámetros proporcionados
    default_config.update(kwargs)
    
    # ✅ CRITICAL: Forzar num_classes correcto basado en one_hot
    if not default_config['one_hot']:
        default_config['num_classes'] = 1
    
    return build_optimized_seizure_tcn(**default_config)
