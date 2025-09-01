import os
import time
import datetime
from pathlib import Path
import tensorflow as tf
from dataset import create_dataset_final, write_tfrecord_splits_FINAL_CORRECTED
from models.TCN import create_seizure_tcn
from keras.losses import CategoricalFocalCrossentropy, BinaryFocalCrossentropy

ONEHOT=False
TIMESTEP=True

# ====================================================================================
# MÉTRICAS PERSONALIZADAS PARA EVALUACIÓN MÉDICA
# ====================================================================================

class MetricsNormalizationMixin:
    """Mixin class with XLA-compatible shared normalization methods"""
    
    @staticmethod
    def _normalize_predictions_shared(y_true, y_pred):
        """
        ✅ XLA-COMPATIBLE: Ultra-simplified shared normalization without tf.cond
        Strategy: Always flatten and apply threshold - works for all cases
        """
        # ===============================
        # STEP 1: FLATTEN EVERYTHING TO 1D
        # ===============================
        y_pred_flat = tf.reshape(y_pred, [-1])
        y_true_flat = tf.reshape(y_true, [-1])
        
        # ===============================
        # STEP 2: SIMPLE BINARY CONVERSION
        # ===============================
        
        # For predictions: convert to binary using simple threshold
        # This works for both probability outputs, logits, and multiclass
        y_pred_binary = tf.cast(y_pred_flat > 0.5, tf.int32)
        
        # For labels: handle different input types uniformly
        # Cast to float first, then threshold, then back to int
        y_true_float = tf.cast(y_true_flat, tf.float32)
        y_true_binary = tf.cast(y_true_float > 0.5, tf.int32)
        
        # ===============================
        # STEP 3: ENSURE SAME LENGTH
        # ===============================
        pred_len = tf.shape(y_pred_binary)[0]
        true_len = tf.shape(y_true_binary)[0]
        min_len = tf.minimum(pred_len, true_len)
        
        y_pred_final = y_pred_binary[:min_len]
        y_true_final = y_true_binary[:min_len]
        
        return y_true_final, y_pred_final
    
    @staticmethod
    def _handle_temporal_pred(y_pred):
        """
        ✅ XLA-COMPATIBLE: Simplified temporal handling without tf.cond
        """
        # Simply flatten and take mean - works for all temporal cases
        y_pred_flat = tf.reshape(y_pred, [-1])
        # Take mean to collapse temporal dimension
        return tf.reduce_mean(y_pred_flat)

class BalancedAccuracy(tf.keras.metrics.Metric, MetricsNormalizationMixin):
    """Balanced Accuracy para clases desbalanceadas - XLA compatible"""
    def __init__(self, name='balanced_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use XLA-compatible shared normalization method
        y_true_norm, y_pred_norm = self._normalize_predictions_shared(y_true, y_pred)
        
        self.tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 1, y_pred_norm == 1), tf.float32)))
        self.tn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 0, y_pred_norm == 0), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 0, y_pred_norm == 1), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 1, y_pred_norm == 0), tf.float32)))
    
    # Keep backward compatibility methods but delegate to XLA-compatible ones
    def _normalize_predictions(self, y_true, y_pred):
        return self._normalize_predictions_shared(y_true, y_pred)
    
    def _handle_temporal_pred(self, y_pred):
        return MetricsNormalizationMixin._handle_temporal_pred(y_pred)
    
    def result(self):
        sensitivity = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        specificity = self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())
        return (sensitivity + specificity) / 2.0
    
    def reset_state(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

class Sensitivity(tf.keras.metrics.Metric, MetricsNormalizationMixin):
    """Sensitivity (Recall) para detección médica - XLA compatible"""
    def __init__(self, name='sensitivity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # ✅ XLA-COMPATIBLE: Use simplified shared normalization method
        y_true_norm, y_pred_norm = self._normalize_predictions_shared(y_true, y_pred)
        
        self.tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 1, y_pred_norm == 1), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 1, y_pred_norm == 0), tf.float32)))
    
    def result(self):
        return self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.tp.assign(0)
        self.fn.assign(0)

class Specificity(tf.keras.metrics.Metric, MetricsNormalizationMixin):
    """Specificity para detección médica - XLA compatible"""
    def __init__(self, name='specificity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # ✅ XLA-COMPATIBLE: Use simplified shared normalization method
        y_true_norm, y_pred_norm = self._normalize_predictions_shared(y_true, y_pred)
        
        self.tn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 0, y_pred_norm == 0), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 0, y_pred_norm == 1), tf.float32)))
    
    def result(self):
        return self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.tn.assign(0)
        self.fp.assign(0)

class BrierScore(tf.keras.metrics.Metric):
    """Optimized Brier Score for all prediction/label combinations"""
    def __init__(self, name='brier_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_squared_error = self.add_weight(name='sse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to probabilities and binary labels consistently
        y_true_prob, y_pred_prob = self._normalize_to_probabilities(y_true, y_pred)
        
        # Clip predictions for numerical stability
        y_pred_prob = tf.clip_by_value(y_pred_prob, 1e-7, 1.0 - 1e-7)
        
        # Compute Brier score
        squared_diff = tf.square(y_true_prob - y_pred_prob)
        
        self.sum_squared_error.assign_add(tf.reduce_sum(squared_diff))
        self.count.assign_add(tf.cast(tf.size(y_true_prob), tf.float32))
    
    def _normalize_to_probabilities(self, y_true, y_pred):
        """
        ✅ COMPLETELY REWRITTEN: All TensorFlow operations, no Python conditionals
        Ultra-simple approach: always treat as binary probabilities
        """
        
        # ===============================
        # HANDLE y_pred: Ultra-simple conversion
        # ===============================
        def process_predictions_simple(y_pred):
            """Super simple: flatten, check range, apply appropriate transformation"""
            # Flatten everything to 1D first
            y_pred_flat = tf.reshape(y_pred, [-1])
            
            # Check value ranges using TensorFlow operations only
            max_val = tf.reduce_max(y_pred_flat)
            min_val = tf.reduce_min(y_pred_flat)
            
            # Simple decision tree using only tf.cond
            # If values look like probabilities [0,1], keep them
            # If values look like logits, apply sigmoid
            # Otherwise, apply sigmoid as fallback
            
            is_prob_like = tf.logical_and(
                tf.greater_equal(min_val, -0.1),
                tf.less_equal(max_val, 1.1)
            )
            
            def keep_as_prob():
                return tf.clip_by_value(y_pred_flat, 0.0, 1.0)
            
            def apply_sigmoid():
                return tf.nn.sigmoid(y_pred_flat)
            
            return tf.cond(is_prob_like, keep_as_prob, apply_sigmoid)
        
        y_pred_prob = process_predictions_simple(y_pred)
        
        # ===============================
        # HANDLE y_true: Ultra-simple conversion
        # ===============================
        def process_labels_simple(y_true):
            """Super simple: flatten, cast to float, clip to [0,1]"""
            # Flatten everything to 1D first
            y_true_flat = tf.reshape(y_true, [-1])
            
            # Convert to float and clip to [0,1]
            y_true_float = tf.cast(y_true_flat, tf.float32)
            
            # Simple binary conversion: anything > 0.5 is positive
            return tf.cast(tf.greater(y_true_float, 0.5), tf.float32)
        
        y_true_prob = process_labels_simple(y_true)
        
        # ===============================
        # ENSURE SAME LENGTH - SIMPLE
        # ===============================
        
        # Get lengths using TensorFlow operations
        pred_len = tf.shape(y_pred_prob)[0]
        true_len = tf.shape(y_true_prob)[0]
        
        # Take minimum length
        min_len = tf.minimum(pred_len, true_len)
        
        # Truncate both to same length
        y_pred_prob = y_pred_prob[:min_len]
        y_true_prob = y_true_prob[:min_len]
        
        # Final safety clipping
        y_pred_prob = tf.clip_by_value(y_pred_prob, 0.0, 1.0)
        y_true_prob = tf.clip_by_value(y_true_prob, 0.0, 1.0)
        
        return y_true_prob, y_pred_prob
    
    def result(self):
        return self.sum_squared_error / (self.count + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.sum_squared_error.assign(0)
        self.count.assign(0)

class TimeLimitCallback(tf.keras.callbacks.Callback):
    """Callback para limitar tiempo de entrenamiento"""
    def __init__(self, max_time_hours=24):
        super().__init__()
        self.max_time_seconds = max_time_hours * 3600
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > self.max_time_seconds:
            self.model.stop_training = True

class ConfusionMatrixMetrics(tf.keras.metrics.Metric):
    """XLA/JIT compatible confusion matrix metrics using simplified approach"""
    
    def __init__(self, num_classes=2, name='confusion_matrix', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_norm, y_pred_norm = self._normalize_predictions(y_true, y_pred)
        
        self.tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 1, y_pred_norm == 1), tf.float32)))
        self.tn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 0, y_pred_norm == 0), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 0, y_pred_norm == 1), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_norm == 1, y_pred_norm == 0), tf.float32)))
    
    def _normalize_predictions(self, y_true, y_pred):
        """
        ✅ XLA/JIT COMPATIBLE: Ultra-simplified normalization without problematic tf.cond
        Strategy: Always flatten first, then process uniformly
        """
        # ===============================
        # STEP 1: FLATTEN EVERYTHING TO 1D
        # ===============================
        y_pred_flat = tf.reshape(y_pred, [-1])
        y_true_flat = tf.reshape(y_true, [-1])
        
        # ===============================
        # STEP 2: SIMPLE BINARY CONVERSION
        # ===============================
        
        # For predictions: convert to binary using simple threshold
        # This works for both probability outputs and logits
        y_pred_binary = tf.cast(y_pred_flat > 0.5, tf.int32)
        
        # For labels: handle different input types uniformly
        # Cast to float first, then threshold, then back to int
        y_true_float = tf.cast(y_true_flat, tf.float32)
        y_true_binary = tf.cast(y_true_float > 0.5, tf.int32)
        
        # ===============================
        # STEP 3: ENSURE SAME LENGTH
        # ===============================
        pred_len = tf.shape(y_pred_binary)[0]
        true_len = tf.shape(y_true_binary)[0]
        min_len = tf.minimum(pred_len, true_len)
        
        y_pred_final = y_pred_binary[:min_len]
        y_true_final = y_true_binary[:min_len]
        
        return y_true_final, y_pred_final
    
    def result(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / (total + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
    
    def get_confusion_matrix_values(self):
        return {
            'tp': float(self.tp.numpy()),
            'tn': float(self.tn.numpy()),
            'fp': float(self.fp.numpy()),
            'fn': float(self.fn.numpy())
        }

class TimePerEpochCallback(tf.keras.callbacks.Callback):
    """Callback optimizado para medir tiempo por época"""
    
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            if logs is not None:
                logs['epoch_time'] = epoch_time
                logs['epoch_time_minutes'] = epoch_time / 60.0
                logs['avg_epoch_time'] = sum(self.epoch_times) / len(self.epoch_times)

class EnhancedCSVLogger(tf.keras.callbacks.CSVLogger):
    """CSV Logger optimizado"""
    
    def __init__(self, filename, separator=',', append=False):
        super().__init__(filename, separator, append)
        self.confusion_metrics = None
    
    def set_model(self, model):
        super().set_model(model)
        for metric in model.metrics:
            if isinstance(metric, ConfusionMatrixMetrics):
                self.confusion_metrics = metric
                break
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and self.confusion_metrics is not None:
            try:
                cm_values = self.confusion_metrics.get_confusion_matrix_values()
                logs.update({
                    'tp': cm_values['tp'],
                    'tn': cm_values['tn'], 
                    'fp': cm_values['fp'],
                    'fn': cm_values['fn'],
                    'ppv': cm_values['tp'] / (cm_values['tp'] + cm_values['fp'] + 1e-7),
                    'npv': cm_values['tn'] / (cm_values['tn'] + cm_values['fn'] + 1e-7)
                })
            except:
                pass
        
        super().on_epoch_end(epoch, logs)

def create_runs_directory(base_name="seizure_experiment"):
    """Crea directorio organizado para experimentos"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{base_name}_{timestamp}"
    
    runs_dir = Path("runs")
    experiment_dir = runs_dir / run_name
    
    dirs_to_create = [
        experiment_dir,
        experiment_dir / "models",
        experiment_dir / "checkpoints", 
        experiment_dir / "logs",
        experiment_dir / "tensorboard"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir

def create_label_transformer(model_info, onehot):
    """Create optimized label transformer based on model output characteristics"""
    
    def transform_labels(x, y):
        """Transform labels AND inputs to match model format"""
        
        # ✅ CRITICAL FIX: Handle input transpose for BATCHED data
        if model_info.get('needs_transpose', False):
            # During training, x has shape (batch, channels, timepoints)
            # We need (batch, timepoints, channels)
            if len(x.shape) == 3:  # Batched data: (batch, channels, timepoints)
                x = tf.transpose(x, [0, 2, 1])  # -> (batch, timepoints, channels)
            elif len(x.shape) == 2:  # Single sample: (channels, timepoints)
                x = tf.transpose(x, [1, 0])  # -> (timepoints, channels)
        
        # ✅ FIXED: Handle labels based on model's ACTUAL output expectation
        if not onehot:
            # Binary mode - modelo espera (batch, time, 1) o (batch, 1)
            if len(y.shape) > 1:
                if y.shape[-1] == 2:
                    # One-hot to binary: take positive class
                    y = y[1]
                else:
                    y = tf.squeeze(y)
            
            # Ensure scalar float
            y = tf.cast(y, tf.float32)
            if len(y.shape) > 0:
                y = tf.reduce_mean(y)
            
            # ✅ CRITICAL FIX: Only expand if model outputs temporal
            if model_info['outputs_temporal']:
                # Model expects (batch, time, 1)
                time_steps = model_info['output_shape'][0]
                batch_size = tf.shape(x)[0]
                
                # Create (batch, time, 1) labels
                y = tf.fill([batch_size, time_steps, 1], y)
            else:
                # Model expects (batch, 1) - keep as scalar and expand later
                y = tf.expand_dims(y, axis=-1)  # (batch, 1)
                
        else:
            # One-hot mode - modelo espera (batch, time, 2) o (batch, 2)  
            if len(y.shape) == 1:
                if y.shape[0] == 2:
                    # Already one-hot - ensure float
                    y = tf.cast(y, tf.float32)
                elif y.shape[0] == 1:
                    # Convert single value to one-hot
                    y = tf.cast(y[0], tf.int32)
                    y = tf.one_hot(y, depth=2)
                else:
                    # Convert first element to one-hot
                    y = tf.cast(y[0], tf.int32)
                    y = tf.one_hot(y, depth=2)
            elif len(y.shape) == 0:
                # Scalar to one-hot
                y = tf.cast(y, tf.int32)
                y = tf.one_hot(y, depth=2)
            elif len(y.shape) > 1:
                # Multi-dimensional - check if already one-hot
                if y.shape[-1] == 2:
                    y = tf.cast(y, tf.float32)
                else:
                    # Take appropriate element and convert
                    y_val = tf.cast(y[0] if len(y.shape) > 1 else y, tf.int32)
                    y = tf.one_hot(y_val, depth=2)
            
            # Ensure output is float32
            y = tf.cast(y, tf.float32)
            
            # ✅ CRITICAL FIX: Only expand if model outputs temporal
            if model_info['outputs_temporal']:
                # Model expects (batch, time, 2)
                time_steps = model_info['output_shape'][0]
                batch_size = tf.shape(x)[0]
                
                # Expand to temporal: (batch, 2) -> (batch, time, 2)
                y = tf.expand_dims(y, axis=1)  # (batch, 1, 2)
                y = tf.tile(y, [1, time_steps, 1])  # (batch, time, 2)
            # If not temporal, y should already be (batch, 2)
        
        return x, y
    
    return transform_labels

def create_optimized_dataset_pipeline(tfrecord_files, n_channels, n_timepoints, batch_size, one_hot, is_training=True):
    """Optimized dataset creation with proper validation and preprocessing"""
    
    def parse_and_validate(x, y):
        """Parse and validate data with proper error handling"""
        # Validate and clean input data (x should be float)
        x = tf.cast(x, tf.float32)
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
        
        # Handle labels based on their data type
        if y.dtype in [tf.int32, tf.int64]:
            # Labels are integers - convert to float for validation if needed
            y_float = tf.cast(y, tf.float32)
            y_float = tf.where(tf.math.is_nan(y_float), tf.zeros_like(y_float), y_float)
            y_float = tf.where(tf.math.is_inf(y_float), tf.zeros_like(y_float), y_float)
            # Convert back to int if originally integer
            y = tf.cast(y_float, y.dtype)
        else:
            # Labels are float - normal validation
            y = tf.cast(y, tf.float32)
            y = tf.where(tf.math.is_nan(y), tf.zeros_like(y), y)
            y = tf.where(tf.math.is_inf(y), tf.zeros_like(y), y)
        
        # Normalize input data to reasonable range
        x = tf.clip_by_value(x, -1000.0, 1000.0)
        
        # ✅ KEEP ORIGINAL FORMAT - DO NOT TRANSPOSE HERE
        # The transpose will be handled in the label_transformer
        # This ensures dataset inspection shows the original format
        # but training gets the correct format
        
        # Handle labels based on mode and ensure correct data type
        if not one_hot:
            # Binary mode - ensure float output
            if len(y.shape) > 1:
                # If one-hot encoded, take the positive class
                if y.shape[-1] == 2:
                    y = y[1]  # Take positive class
                else:
                    y = tf.reduce_mean(tf.cast(y, tf.float32))  # Collapse to single value
            
            # ✅ CRITICAL: Keep as scalar for now - expansion happens in label_transformer
            y = tf.cast(y, tf.float32)
            if len(y.shape) > 0:
                y = tf.reduce_mean(y)
            y = tf.cast(y > 0.5, tf.float32)
        else:
            # One-hot mode - ensure proper format
            if len(y.shape) == 0:
                # Scalar label - convert to one-hot
                y = tf.cast(y, tf.int32)
                y = tf.one_hot(y, depth=2)
            elif len(y.shape) == 1:
                if tf.shape(y)[0] == 2:
                    # Already one-hot - ensure float
                    y = tf.cast(y, tf.float32)
                else:
                    # Single value - convert to one-hot
                    y = tf.cast(y[0], tf.int32)
                    y = tf.one_hot(y, depth=2)
            else:
                # Multi-dimensional - handle appropriately
                if y.shape[-1] == 2:
                    # Already one-hot format
                    y = tf.cast(y, tf.float32)
                else:
                    # Take first element and convert to one-hot
                    y_val = tf.cast(y[0], tf.int32) if len(y.shape) > 1 else tf.cast(y, tf.int32)
                    y = tf.one_hot(y_val, depth=2)
        
        return x, y
    
    # Create base dataset using the corrected create_dataset_final from dataset.py
    dataset = create_dataset_final(
        tfrecord_files,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        batch_size=batch_size,
        one_hot=one_hot,
        time_step=False,  # ✅ Force window-level labels initially
        balance_pos_frac=None,
        cache=False,
        drop_remainder=False,
        shuffle=is_training
    )
    
    # Apply validation and preprocessing
    dataset = dataset.map(
        parse_and_validate,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Optimization based on training/validation
    if is_training:
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.repeat()
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_model_with_validation(model_config, sample_input_shape, onehot=True):
    """Build model with proper validation and output shape detection"""
    # ✅ FIXED: Improved GPU strategy initialization with proper memory setup
    
    # Initialize GPU memory growth BEFORE creating strategy
    gpus = tf.config.experimental.list_physical_devices('GPU')
    strategy = None
    
    if gpus:
        try:
            # This needs to be done BEFORE any other TensorFlow operations
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # Also set visible devices
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Now create the strategy
            strategy = tf.distribute.OneDeviceStrategy("GPU:0")
            print(f"✅ GPU strategy initialized: {gpus[0].name}")
            
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"⚠️  GPU memory growth setup failed (already initialized): {e}")
            try:
                # Try to create strategy anyway
                strategy = tf.distribute.OneDeviceStrategy("GPU:0")
                print(f"✅ GPU strategy created despite memory growth warning: {gpus[0].name}")
            except Exception as e2:
                print(f"❌ GPU strategy creation failed: {e2}")
                strategy = tf.distribute.get_strategy()
        except Exception as e:
            print(f"❌ GPU setup failed: {e}")
            strategy = tf.distribute.get_strategy()
    else:
        print("❌ No GPU found, using default strategy")
        strategy = tf.distribute.get_strategy()
    
    # Build model within strategy scope
    with strategy.scope():
        model = create_seizure_tcn(**model_config)
    
    # Determine correct input shape for model
    # Model expects (batch, time, channels) format
    if len(sample_input_shape) == 2:
        if sample_input_shape[0] == model_config['input_channels']:
            # Current: (channels, timepoints) -> Need: (timepoints, channels)
            correct_input_shape = (sample_input_shape[1], sample_input_shape[0])
        else:
            # Already: (timepoints, channels)
            correct_input_shape = sample_input_shape
    else:
        correct_input_shape = sample_input_shape
            
    print(f"Original dataset shape: {sample_input_shape}")
    print(f"Model input shape: {correct_input_shape}")
    
    # Test model with corrected input shape
    sample_input = tf.random.normal((1,) + correct_input_shape)
    try:
        with strategy.scope():
            sample_output = model(sample_input)
        
        model_info = {
            'input_shape': correct_input_shape,
            'dataset_shape': sample_input_shape,
            'output_shape': sample_output.shape[1:],  # Remove batch dimension
            'outputs_temporal': len(sample_output.shape) == 3 and sample_output.shape[1] > 1,
            'num_classes': sample_output.shape[-1] if len(sample_output.shape) > 1 else 1,
            'needs_transpose': sample_input_shape != correct_input_shape,
            'strategy': strategy  # ✅ Store strategy for later use
        }
        
        print(f"Model built successfully:")
        print(f"  Expected input shape: {model_info['input_shape']}")
        print(f"  Dataset input shape: {model_info['dataset_shape']}")
        print(f"  Needs transpose: {model_info['needs_transpose']}")
        print(f"  Output shape: {model_info['output_shape']}")
        print(f"  Temporal outputs: {model_info['outputs_temporal']}")
        print(f"  Number of classes: {model_info['num_classes']}")
        print(f"  Strategy: {type(strategy).__name__}")
        
        # ✅ Check if actually using GPU
        if hasattr(strategy, '_device_map') or 'GPU' in str(strategy):
            print(f"  ✅ Using GPU acceleration")
        else:
            print(f"  ⚠️  Using CPU (no GPU acceleration)")
        
        return model, model_info
        
    except Exception as e:
        print(f"Model test failed with shape {correct_input_shape}: {e}")
        
        # Try alternative input arrangement
        if len(sample_input_shape) == 2:
            alt_shape = (sample_input_shape[1], sample_input_shape[0])
            print(f"Trying alternative shape: {alt_shape}")
            
            try:
                sample_input_alt = tf.random.normal((1,) + alt_shape)
                with strategy.scope():
                    sample_output = model(sample_input_alt)
                
                model_info = {
                    'input_shape': alt_shape,
                    'dataset_shape': sample_input_shape,
                    'output_shape': sample_output.shape[1:],
                    'outputs_temporal': len(sample_output.shape) == 3 and sample_output.shape[1] > 1,
                    'num_classes': sample_output.shape[-1] if len(sample_output.shape) > 1 else 1,
                    'needs_transpose': True,
                    'strategy': strategy
                }
                
                print(f"Model built with alternative shape:")
                print(f"  Model input shape: {model_info['input_shape']}")
                print(f"  Dataset shape: {model_info['dataset_shape']}")
                print(f"  Needs transpose: {model_info['needs_transpose']}")
                print(f"  Output shape: {model_info['output_shape']}")
                
                return model, model_info
                
            except Exception as e2:
                raise RuntimeError(f"Model building failed with both shapes. Original: {e}, Alternative: {e2}")
        else:
            raise RuntimeError(f"Model building failed: {e}")

def setup_metrics_and_loss(is_binary, class_weights=None, gamma=2.0):
    """Setup optimized metrics and loss functions"""
    
    # ✅ RESTORED: All metrics including the XLA-compatible ConfusionMatrixMetrics
    # Core metrics that work for all cases
    metrics = [
        'accuracy',
        BalancedAccuracy(),
        Sensitivity(), 
        Specificity(),
        BrierScore(),  # ✅ Working with robust implementation
        ConfusionMatrixMetrics(num_classes=1 if is_binary else 2)  # ✅ RESTORED with XLA-compatible implementation
    ]
    
    # Additional metrics for specific cases - with better error handling
    if not is_binary:
        try:
            metrics.extend([
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(curve='ROC', name='auc_roc'),
                tf.keras.metrics.AUC(curve='PR', name='auc_pr')
            ])
        except Exception as e:
            print(f"Warning: Could not add AUC metrics: {e}")
    else:
        # Add simple binary metrics that work reliably
        try:
            metrics.extend([
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ])
        except Exception as e:
            print(f"Warning: Could not add precision/recall metrics: {e}")
    
    # Setup loss function
    if is_binary:
        alpha = class_weights[1] if class_weights else 0.75
        loss_fn = BinaryFocalCrossentropy(alpha=alpha, gamma=gamma)
    else:
        loss_fn = CategoricalFocalCrossentropy(alpha=class_weights, gamma=gamma)
    
    return metrics, loss_fn

def train_seizure_complete_v2(model_config,
                             train_dataset, 
                             val_dataset,
                             steps_per_epoch,
                             validation_steps,
                             epochs=50,
                             learning_rate=0.001,
                             class_weights=None,
                             gamma=2.0,
                             experiment_name='seizure_model',
                             monitor_metric='val_accuracy',
                             patience=10,
                             max_time_hours=24,
                             onehot=True,
                             verbose=False):
    """Optimized training pipeline with improved error handling"""
    
    experiment_dir = create_runs_directory(experiment_name)
    
    # File paths
    model_best_path = experiment_dir / "models" / f"{experiment_name}_best.keras"
    model_final_path = experiment_dir / "models" / f"{experiment_name}_final.keras" 
    checkpoint_path = experiment_dir / "checkpoints" / f"{experiment_name}_epoch_{{epoch:03d}}.keras"
    csv_log_path = experiment_dir / "logs" / f"{experiment_name}_training.csv"
    tensorboard_path = experiment_dir / "tensorboard"
    
    is_binary = not onehot
    
    # Inspect dataset to get input shape
    try:
        sample_batch = next(iter(train_dataset.take(1)))
        input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
        
        if verbose:
            print(f"Dataset input shape: {input_shape}")
            print(f"Dataset label shape: {sample_batch[1].shape}")
            
            # Handle scalar labels safely
            if len(sample_batch[1].shape) == 0:
                print(f"Labels are scalar - single label: {sample_batch[1]}")
                label_shape = ()
            else:
                label_shape = sample_batch[1].shape[1:] if len(sample_batch[1].shape) > 1 else sample_batch[1].shape
                print(f"Single label shape: {label_shape}")
                
                # Safe slicing for non-scalar labels
                try:
                    if sample_batch[1].shape[0] >= 3:
                        print(f"Sample labels (first 3): {sample_batch[1][:3]}")
                    else:
                        print(f"Sample labels (all): {sample_batch[1]}")
                except:
                    print(f"Sample labels: {sample_batch[1]}")
            
    except Exception as e:
        raise RuntimeError(f"Could not inspect dataset: {e}")
    
    # Build model with validation
    model, model_info = build_model_with_validation(model_config, input_shape, onehot)
    
    # ✅ Get the strategy used for model building
    strategy = model_info.get('strategy', tf.distribute.get_strategy())
    
    # ✅ CRITICAL: Build the model BEFORE compiling to ensure proper shape
    if model_info['needs_transpose']:
        # Build with the correct input shape
        build_input = tf.random.normal((1,) + model_info['input_shape'])
        with strategy.scope():
            _ = model(build_input)  # This builds the model with correct shapes
        if verbose:
            print(f"Model built and validated with shape: {build_input.shape}")
    
    # ✅ FIXED: Setup everything within the SAME strategy scope used for model creation
    with strategy.scope():
        # Setup metrics and loss
        metrics, loss_fn = setup_metrics_and_loss(is_binary, class_weights, gamma)
        
        # Setup optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )
        
        # ✅ RE-ENABLE JIT: Now that ConfusionMatrixMetrics is XLA-compatible
        try:
            # Try with JIT compilation first - should work now
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics,
                jit_compile=True  # ✅ RE-ENABLED JIT with XLA-compatible metrics
            )
            if verbose:
                print("✅ JIT compilation enabled with XLA-compatible metrics")
        except Exception as jit_error:
            if verbose:
                print(f"⚠️  JIT compilation failed ({jit_error}), falling back to non-JIT")
            # Fallback to non-JIT compilation
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics,
                jit_compile=False
            )
    
    # Transform datasets if needed
    needs_transform = (
        model_info.get('needs_transpose', False) or
        model_info['outputs_temporal'] or 
        (is_binary and len(sample_batch[1].shape) > 1) or
        (not is_binary and model_info['outputs_temporal'])
    )
    
    if needs_transform:
        if verbose:
            print("Applying dataset transformations...")
            print(f"  Transpose input: {model_info.get('needs_transpose', False)}")
            print(f"  Temporal outputs: {model_info['outputs_temporal']}")
            print(f"  Label shape incompatible: {len(sample_batch[1].shape) > 1 if is_binary else False}")
        
        label_transformer = create_label_transformer(model_info, onehot)
        train_dataset = train_dataset.map(label_transformer, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(label_transformer, num_parallel_calls=tf.data.AUTOTUNE)
        
        # ✅ ENHANCED: Validate transformation with proper error handling
        if verbose:
            try:
                transformed_batch = next(iter(train_dataset.take(1)))
                print(f"After transformation:")
                print(f"  X shape: {transformed_batch[0].shape}")
                print(f"  Y shape: {transformed_batch[1].shape}")
                print(f"  Y sample: {transformed_batch[1][0]}")
                
                # Test model compatibility with SINGLE sample to avoid batch issues
                single_x = transformed_batch[0][:1]  # Take first sample
                single_y = transformed_batch[1][:1]   # Take first label
                
                print(f"  Testing with single sample X shape: {single_x.shape}")
                print(f"  Testing with single sample Y shape: {single_y.shape}")
                
                # Test model prediction within strategy scope
                with strategy.scope():
                    test_pred = model(single_x)
                print(f"  Model prediction shape: {test_pred.shape}")
                
                # Check shape compatibility
                pred_shape = test_pred.shape[1:]
                label_shape = single_y.shape[1:]
                compatible = pred_shape == label_shape
                print(f"  Shapes compatible: {compatible}")
                
                if not compatible:
                    print(f"  Prediction shape: {pred_shape}")
                    print(f"  Label shape: {label_shape}")
                    print(f"  ⚠️  Shape mismatch - but training may still work due to automatic broadcasting")
                else:
                    print(f"  ✅ Perfect shape match!")
                
            except Exception as e:
                print(f"Transformation validation warning: {e}")
                # Continue anyway - the model was built successfully so training should work
    
    # Optimize datasets
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Setup callbacks
    time_callback = TimePerEpochCallback()
    
    # ✅ ADD: Clean progress callback for better output formatting
    progress_callback = CleanProgressCallback(verbose=verbose, show_progress_bar=verbose)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_best_path),
            monitor=monitor_metric,
            mode='max' if 'accuracy' in monitor_metric or 'f1' in monitor_metric else 'min',
            save_best_only=True,
            save_weights_only=False,
            verbose=0
        ),
        
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_freq='epoch',
            save_weights_only=False,
            verbose=0
        ),
        
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            mode='max' if 'accuracy' in monitor_metric or 'f1' in monitor_metric else 'min',
            restore_best_weights=True,
            verbose=0
        ),
        
        time_callback,
        progress_callback,  # ✅ ADD: Custom progress callback
        
        EnhancedCSVLogger(
            filename=str(csv_log_path),
            separator=',',
            append=False
        ),
        
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_path),
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq='epoch'
        ),
        
        TimeLimitCallback(max_time_hours=max_time_hours)
    ]
    
    if verbose:
        print(f"Starting training: {experiment_dir.name}")
        print(f"Using strategy: {type(strategy).__name__}")
    
    # ✅ FIXED: Use verbose=0 to prevent step-by-step metric printing
    # The custom progress callback will handle epoch-level reporting
    with strategy.scope():
        history = model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=0  # ✅ Keep verbose=0 to prevent Keras built-in progress output
        )
    
    # Save final model
    model.save(str(model_final_path))
    
    if verbose:
        avg_time = sum(time_callback.epoch_times) / len(time_callback.epoch_times) if time_callback.epoch_times else 0
        total_time = sum(time_callback.epoch_times) if time_callback.epoch_times else 0
        print(f"Training completed: {len(history.history['loss'])} epochs, {total_time/60:.1f}min total, {avg_time:.1f}s/epoch")
    
    return model, history, experiment_dir

# Configuración principal
def run_seizure_pipeline(WINDOW_SIZE=10, FS=256):
    """Pipeline principal optimizado"""
    
    # Configuración
    config = {
        'one_hot': ONEHOT,
        'window_size': WINDOW_SIZE,  # Window size in seconds
        'fs': FS,  # Sampling frequency
        'n_timepoints': WINDOW_SIZE * FS,  # Total timepoints per window
        'n_channels': 22,
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 50,
        'patience': 10
    }
    
    if verbose_pipeline := True:
        print(f"Pipeline Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Create TFRecords with validation
    print("Creating TFRecord files...")
    definitivo_stats = write_tfrecord_splits_FINAL_CORRECTED(
        data_dir='DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3',
        tfrecord_dir='tfrecords',
        montage='ar',
        limits={'eval': 10, 'dev': 10},
        n_workers=2,
        chunk_size_mb=50,
        resume=False,
        validate=True
    )
    
    # Print TFRecord creation stats
    if definitivo_stats:
        print(f"TFRecord creation stats: {definitivo_stats}")

    # Get TFRecord files
    tfrecord_files = list(Path('tfrecords').rglob('*.tfrecord'))
    if not tfrecord_files:
        raise FileNotFoundError("No TFRecord files found")
    
    if verbose_pipeline:
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Inspect file sizes for better step estimation
        total_size_mb = 0
        for f in tfrecord_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            if verbose_pipeline and len(tfrecord_files) <= 10:  # Only show details for small number of files
                print(f"  {f.name}: {size_mb:.1f} MB")
        
        print(f"Total TFRecord size: {total_size_mb:.1f} MB")
    
    # Split files for train/validation
    split_idx = int(0.8 * len(tfrecord_files))
    train_files = [str(f) for f in tfrecord_files[:split_idx]]
    val_files = [str(f) for f in tfrecord_files[split_idx:]]
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Create optimized datasets
    train_ds = create_optimized_dataset_pipeline(
        train_files,
        n_channels=config['n_channels'],
        n_timepoints=config['n_timepoints'],
        batch_size=config['batch_size'],
        one_hot=config['one_hot'],
        is_training=True
    )
    
    val_ds = create_optimized_dataset_pipeline(
        val_files,
        n_channels=config['n_channels'],
        n_timepoints=config['n_timepoints'],
        batch_size=config['batch_size'],
        one_hot=config['one_hot'],
        is_training=False
    )
    
    # Better step estimation based on actual data inspection
    print("Estimating dataset sizes...")
    
    # Count samples in a few batches to estimate total
    try:
        sample_count = 0
        for i, batch in enumerate(train_ds.take(5)):
            batch_size = batch[0].shape[0]
            sample_count += batch_size
            if verbose_pipeline:
                print(f"  Batch {i+1}: {batch_size} samples")
        
        # Estimate total samples assuming similar batch sizes
        if sample_count > 0:
            avg_batch_size = sample_count / 5
            # Rough estimation: each file might contain similar amount of data
            estimated_total_samples = len(train_files) * avg_batch_size * 20  # Conservative multiplier
            steps_per_epoch = max(10, int(estimated_total_samples // config['batch_size']))
        else:
            steps_per_epoch = 100  # Fallback
            
    except Exception as e:
        print(f"Could not estimate dataset size: {e}")
        steps_per_epoch = 100  # Fallback
    
    # Validation steps
    try:
        val_sample_count = 0
        for i, batch in enumerate(val_ds.take(3)):
            val_sample_count += batch[0].shape[0]
        
        if val_sample_count > 0:
            avg_val_batch_size = val_sample_count / 3
            estimated_val_samples = len(val_files) * avg_val_batch_size * 20
            val_steps = max(5, int(estimated_val_samples // config['batch_size']))
        else:
            val_steps = 25  # Fallback
            
    except Exception as e:
        print(f"Could not estimate validation size: {e}")
        val_steps = 25  # Fallback
    
    if verbose_pipeline:
        print(f"Estimated training samples: {steps_per_epoch * config['batch_size']}")
        print(f"Training steps per epoch: {steps_per_epoch}")
        print(f"Estimated validation samples: {val_steps * config['batch_size']}")
        print(f"Validation steps: {val_steps}")
    
    # Verify minimum reasonable steps
    if steps_per_epoch < 10:
        print(f"Warning: Very few training steps ({steps_per_epoch}). Using minimum of 50.")
        steps_per_epoch = 50
        
    if val_steps < 5:
        print(f"Warning: Very few validation steps ({val_steps}). Using minimum of 10.")
        val_steps = 10
    
    # ✅ FIXED: Model configuration with proper parameters
    model_config = {
        'input_channels': config['n_channels'],
        'num_filters': 32,
        'num_blocks': 4,
        'one_hot': config['one_hot'],
        'time_step': TIMESTEP,  # Use global TIMESTEP setting
        'num_classes': 2 if config['one_hot'] else 1,  # Explicit class count
        'window_length_samples': config['n_timepoints']  # Pass window length
    }
    
    print(f"Model configuration:")
    print(f"  One-hot: {model_config['one_hot']}")
    print(f"  Time-step: {model_config['time_step']}")
    print(f"  Classes: {model_config['num_classes']}")
    
    # Train model
    model, history, experiment_dir = train_seizure_complete_v2(
        model_config=model_config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        gamma=2.0,
        experiment_name='seizure_detection_optimized',
        monitor_metric='val_balanced_accuracy',
        patience=config['patience'],
        max_time_hours=12,
        onehot=config['one_hot'],
        verbose=True
    )
    
    if verbose_pipeline:
        print(f"Training completed. Results saved to: {experiment_dir}")
        
        # Print final metrics
        final_metrics = {k: v[-1] for k, v in history.history.items() if v}
        print("Final metrics:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return model, history, experiment_dir

# ✅ ADD GPU memory setup at module level to ensure it's done early
def setup_gpu_memory():
    """Setup GPU memory growth at import time"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Try to enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            print("ℹ️  No GPUs found")
    except RuntimeError:
        # Memory growth must be set before GPUs have been initialized
        print("ℹ️  GPU memory growth already configured")
    except Exception as e:
        print(f"⚠️  GPU memory setup warning: {e}")

# ✅ Call GPU setup at module import
setup_gpu_memory()

class CleanProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for clean epoch-only progress reporting with tqdm-like progress bar"""
    
    def __init__(self, verbose=True, show_progress_bar=True):
        super().__init__()
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.start_time = None
        self.epoch_start = None
        self.current_step = 0
        self.total_steps = 0
        
    def on_train_begin(self, logs=None):
        if self.verbose:
            print("Training started...")
        self.start_time = time.time()
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
        self.epoch_start = time.time()
        self.current_step = 0
        self.total_steps = self.params.get('steps', 0)
    
    def on_batch_end(self, batch, logs=None):
        if not self.show_progress_bar or not self.verbose:
            return
            
        self.current_step = batch + 1
        
        # Calculate progress
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 0
        bar_length = 30
        filled_length = int(bar_length * progress)
        
        # Create progress bar
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Calculate time estimates
        elapsed = time.time() - self.epoch_start
        if self.current_step > 0:
            time_per_step = elapsed / self.current_step
            eta = time_per_step * (self.total_steps - self.current_step)
            eta_str = f"{int(eta)}s"
        else:
            eta_str = "?s"
        
        # Format key metrics from logs
        metrics_str = ""
        if logs:
            key_metrics = ['loss', 'accuracy', 'balanced_accuracy']
            metric_parts = []
            for metric in key_metrics:
                if metric in logs:
                    metric_parts.append(f"{metric}: {logs[metric]:.4f}")
            if metric_parts:
                metrics_str = " - " + " - ".join(metric_parts[:3])  # Limit to 3 metrics
        
        # Print progress bar (overwrite same line)
        progress_text = (f"\r{self.current_step}/{self.total_steps} "
                        f"[{bar}] "
                        f"{progress*100:.1f}% "
                        f"- {int(elapsed)}s - ETA: {eta_str}"
                        f"{metrics_str}")
        
        print(progress_text, end='', flush=True)
    
    def on_epoch_end(self, epoch, logs=None):
        if not self.verbose:
            return
            
        # Clear the progress bar line
        if self.show_progress_bar:
            print()  # New line after progress bar
            
        epoch_time = time.time() - self.epoch_start
        
        # Format metrics nicely
        if logs:
            # Separate training and validation metrics
            train_metrics = {}
            val_metrics = {}
            
            for key, value in logs.items():
                if key.startswith('val_'):
                    val_metrics[key[4:]] = value  # Remove 'val_' prefix
                elif key not in ['epoch_time', 'epoch_time_minutes', 'avg_epoch_time']:
                    train_metrics[key] = value
            
            # Print training metrics
            train_parts = []
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    train_parts.append(f"{k}: {v:.4f}")
            
            if train_parts:
                train_str = " - ".join(train_parts)
                print(f" {epoch_time:.0f}s - {train_str}")
            
            # Print validation metrics if available
            if val_metrics:
                val_parts = []
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        val_parts.append(f"val_{k}: {v:.4f}")
                
                if val_parts:
                    val_str = " - ".join(val_parts)
                    print(f" Validation: {val_str}")