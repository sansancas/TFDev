import os
import re
import glob
import time
import math
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Optional
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.losses import BinaryFocalCrossentropy, CategoricalFocalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy , Tversky#, Loss, Reduction
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import absl.logging as absl
from tqdm.keras import TqdmCallback
from dataset import create_dataset_final_v2, write_tfrecord_splits_FINAL_CORRECTED, DEFAULT_EEG_FEATURES

ENABLE_XLA = True                    # Master flag for XLA/JIT compilation
JIT_COMPILE_MODEL = False             # JIT compile individual model layers (model.compile jit_compile=)
JIT_GLOBAL = True                    # Global TensorFlow XLA setting

# Apply XLA settings
if ENABLE_XLA and JIT_GLOBAL:
    tf.config.optimizer.set_jit(True)
    print("XLA/JIT enabled globally")
else:
    tf.config.optimizer.set_jit(False)
    print("XLA/JIT disabled globally")

from tensorflow.keras.callbacks import Callback, CSVLogger, TerminateOnNaN, SwapEMAWeights, TensorBoard, BackupAndRestore
from models.TCN import build_tcn
from models.Hybrid import build_hybrid
from models.Transformer import build_transformer
from dataset import make_balanced_stream
# from tensorflow_addons.callbacks import TQDMProgressBar

USE_GRU = True
USE_SE = True
PREPROCESS = {
    'bandpass': (0.5, 40.),   # (low, high) Hz
    'notch': 60.,             # Hz
    'resample': 256,          # Hz target
}
WINDOW_SEC = 5.0             # Window length in seconds
BATCH_SIZE = 12
EPOCHS = 100
USE_DROPOUT = True
DROPOUT_RATE = 0.2
LEARNING_RATE = 2e-4
MIN_LR_FRACTION = 0.05
MIN_LR = LEARNING_RATE * MIN_LR_FRACTION
WARMUP_RATIO = 0.1
LIMITS={'train': 450, 'dev': 100, 'eval': 100}
TIME_LIMIT_H = 48
FRAME_HOP_SEC = 2.5
SWEEP=False
HPC = False
ONEHOT = False
TIME_STEP = False
TRANSPOSE = True
NOTCH = True
BANDPASS = True
NORMALIZE = True
NUM_CLASSES = 2  if ONEHOT else 1
WRITE = True
CUT = True
FULL_REC = './records2' if ONEHOT else './bin_records2'
CUT_REC = './records_cut2' if ONEHOT else './bin_records_cut2'
TFRECORD_DIR = CUT_REC if CUT else FULL_REC
RUNS_DIR = Path("./runs")
RUN_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = RUNS_DIR / f"eeg_seizures_{RUN_STAMP}"

MODEL = 'TCN'  
# Use focal losses if True; otherwise standard cross-entropy losses
FOCAL = True
TVERSKY = False
CLASS_WEIGHTS = False
BALANCED_STREAM = None #0.2

# Dataset window mode configuration:
# - "default": hard window label
# - "soft": soft label = frac of positive frames
# - "features": window-level EEG features concatenated as extra channels (only window-level)
WINDOW_MODE = "features"
# Feature names to use when WINDOW_MODE=="features" (subset of dataset.DEFAULT_EEG_FEATURES)
# Lighter, standard set for seizure studies (time-domain + key spectral):

# FEATURE_NAMES = list(DEFAULT_EEG_FEATURES)  # all

FEATURE_NAMES = [
    "rms_eeg",              # energía global (mantén solo uno de rms/std/mad/activity)
    "line_length",          # complejidad y contenido HF
    "hjorth_mobility",      # dinámica de primer orden
    "hjorth_complexity",    # irregularidad (2º orden / 1º orden)
    "bp_rel_theta",         # composición espectral lenta
    "bp_rel_alpha",         # ritmo de reposo / cambios de activación
    "bp_rel_beta",          # activación / rápida
    "ratio_theta_beta",     # equilibrio lentitud vs rápida (un único ratio discriminativo)
    "spectral_entropy",     # organización vs aleatoriedad
    "sef95",                # desplazamiento espectral global
    "tkeo_mean",            # energía instantánea / bursts
    "zcr",                  # cruces por cero (HF / fragmentación)
]

# If True and WINDOW_MODE=="features" and TIME_STEP==False, the dataset will return (eeg, feats) separately
# and models will use a dual-input path for better efficiency.
FEATURES_AS_VECTOR = True

# ===== DMD / Koopman-aux features (dataset-side) =====
ENABLE_DMD_FEATURES = True
DMD_OPTS = {"n_modes": 8, "subsample": 2}
INCLUDE_LABEL_FEATS = False # Append [y_soft, y_hard] to features vector for inspection


# ===== Domain ID (for inter-patient generalization / DA) =====
RETURN_DOMAIN_ID = True # If True, dataset returns domain input (patient/record)
DOMAIN_FROM = "patient" # "patient" | "record"
DOMAIN_TO_INT = True
DOMAIN_NUM_BUCKETS = 4096
INCLUDE_PATIENTS = None # e.g., [b"pt_001", b"pt_002"]
EXCLUDE_PATIENTS = None # e.g., [b"heldout_pt"]
USE_DOMAIN_INPUT_IN_MODEL = True

# ===================== Model hyperparams =====================
TCN_KERNEL_SIZE = 7
TCN_BLOCKS = 7
TCN_FILTERS = 64
SE_RATIO = 16


NUM_ATTENTION_HEADS=4
RNN_UNITS=64


EMBED_DIMENSION=128
TRANS_LAYERS=4
ATTENTION_HEADS=4
MLP_DIMENSION=256


# ===== Regularizers / Aux heads =====
KOOPMAN_LATENT_DIM = 64
KOOPMAN_LOSS_WEIGHT = 0.1
USE_RECONSTRUCTION_HEAD = True
RECON_WEIGHT = 0.05
RECON_TARGET = "signal" # for Transformer we reconstruct projection embedding
BOTTLENECK_DIM = 128 #128
EXPAND_DIM = 256
# ---------------------- Run configuration snapshot ----------------------
def save_run_config(run_dir: Path, extra: dict | None = None):
    cfg = {
        "MODEL": MODEL,
        "FOCAL": FOCAL,
        "CLASS_WEIGHTS": CLASS_WEIGHTS,
        "BALANCED_STREAM": BALANCED_STREAM,
        "PREPROCESS": PREPROCESS,
        "WINDOW_SEC": WINDOW_SEC,
        "FRAME_HOP_SEC": FRAME_HOP_SEC,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "MIN_LR_FRACTION": MIN_LR_FRACTION,
        "WARMUP_RATIO": WARMUP_RATIO,
        "TIME_LIMIT_H": TIME_LIMIT_H,
        "LIMITS": LIMITS,
        "HPC": HPC,
        "ONEHOT": ONEHOT,
        "TIME_STEP": TIME_STEP,
        "USE_SE": USE_SE,
        "USE_DROPOUT": USE_DROPOUT,
        "DROPOUT_RATE": DROPOUT_RATE,
        "WINDOW_MODE": WINDOW_MODE,
        "FEATURE_NAMES": FEATURE_NAMES,
        "FEATURES_AS_VECTOR": FEATURES_AS_VECTOR,
        "ENABLE_DMD_FEATURES": ENABLE_DMD_FEATURES,
        "DMD_OPTS": DMD_OPTS,
        "INCLUDE_LABEL_FEATS": INCLUDE_LABEL_FEATS,
        "RETURN_DOMAIN_ID": RETURN_DOMAIN_ID,
        "DOMAIN_FROM": DOMAIN_FROM,
        "DOMAIN_TO_INT": DOMAIN_TO_INT,
        "DOMAIN_NUM_BUCKETS": DOMAIN_NUM_BUCKETS,
        "INCLUDE_PATIENTS": INCLUDE_PATIENTS,
        "EXCLUDE_PATIENTS": EXCLUDE_PATIENTS,
        "TCN_KERNEL_SIZE": TCN_KERNEL_SIZE,
        "TCN_BLOCKS": TCN_BLOCKS,
        "TCN_FILTERS": TCN_FILTERS,
        "SE_RATIO": SE_RATIO,
        "NUM_ATTENTION_HEADS": NUM_ATTENTION_HEADS,
        "RNN_UNITS": RNN_UNITS,
        "EMBED_DIMENSION": EMBED_DIMENSION,
        "TRANS_LAYERS": TRANS_LAYERS,
        "ATTENTION_HEADS": ATTENTION_HEADS,
        "MLP_DIMENSION": MLP_DIMENSION,
        "CUT": CUT,
        "TFRECORD_DIR": str(TFRECORD_DIR),
        "KOOPMAN_LATENT_DIM": KOOPMAN_LATENT_DIM,
        "KOOPMAN_LOSS_WEIGHT": KOOPMAN_LOSS_WEIGHT,
        "USE_RECONSTRUCTION_HEAD": USE_RECONSTRUCTION_HEAD,
        "RECON_WEIGHT": RECON_WEIGHT,
        "RECON_TARGET": RECON_TARGET,
        "BOTTLENECK_DIM": BOTTLENECK_DIM,
        "EXPAND_DIM": EXPAND_DIM,
        "RUN_DIR": str(run_dir),
    }
    if extra:
        cfg.update(extra)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

def save_model_summary(model, run_dir: Path):
    buf = []
    model.summary(print_fn=lambda s: buf.append(s))
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "artifacts" / "model_summary.txt", "w") as f:
        f.write("\n".join(buf))
    
    # ---------------------- Receptive Field utilities ----------------------
def tcn_receptive_field(kernel_size: int, num_blocks: int, dilation_base: int = 2, convs_per_block: int = 2) -> int:
    """Compute receptive field (in time steps) for a stack of dilated TCN blocks.
    Each block contains `convs_per_block` conv layers (here 2), both with the same dilation d=base**i.
    RF ≈ 1 + (k-1) * convs_per_block * Σ base**i, i=0..num_blocks-1.
    """
    return 1 + (kernel_size - 1) * convs_per_block * sum(dilation_base ** i for i in range(num_blocks))

def print_receptive_field_tcn(t_steps: int, fs: int, kernel_size: int, num_blocks: int, run_dir: Path | None = None):
    rf_steps = tcn_receptive_field(kernel_size, num_blocks)
    rf_sec = rf_steps / float(fs)
    win_sec = t_steps / float(fs)
    msg = (
        f"TCN receptive field: {rf_steps} steps (~{rf_sec:.3f}s) over window {t_steps} steps (~{win_sec:.3f}s).\n"
        f"Blocks={num_blocks}, kernel_size={kernel_size}, fs={fs}Hz"
    )
    print(msg)
    if run_dir is not None:
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(run_dir / "artifacts" / "receptive_field.txt", "w") as f:
            f.write(msg + "\n")

def infer_tcn_hparams_from_model(model) -> tuple[int, int]:
    """Infer (kernel_size, num_blocks) from a TCN model by scanning Conv1D/SeparableConv1D blocks.
    Falls back to (7, 7) if not detected.
    """
    ks = None
    blocks = set()
    for l in model.layers:
        name = getattr(l, 'name', '')
        if not hasattr(l, 'kernel_size'):
            continue
        if 'block' in name and (name.startswith('conv') or name.startswith('sepconv')):
            # kernel_size is tuple like (k,)
            try:
                ks = int(l.kernel_size[0])
            except Exception:
                pass
            m = re.search(r'block(\d+)', name)
            if m:
                try:
                    blocks.add(int(m.group(1)))
                except Exception:
                    pass
    num_blocks = max(blocks) if blocks else 7
    if ks is None:
        ks = 7
    return ks, num_blocks


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 1=INFO, 2=WARNING, 3=ERROR
# si quieres, suprime mensajes de oneDNN también
os.environ.setdefault("ONE_DNN_VERBOSE", "0")

# habilita logging interactivo de Keras (progbar con \r), si existe en tu versión
try:
    if hasattr(tf.keras.utils, "enable_interactive_logging"):
        tf.keras.utils.enable_interactive_logging()
    # y baja la verbosidad de loggers Python/absl
    tf.get_logger().setLevel(logging.ERROR)
    try:
        
        absl.set_verbosity(absl.ERROR)
    except Exception:
        pass
except Exception:
    pass

if HPC:
    mixed_precision.set_global_policy('float32')
    print("Using float32 policy for HPC")
else:
    mixed_precision.set_global_policy('mixed_float16')
    print("Using mixed_float16")

np.random.seed(42)
tf.random.set_seed(42)

class LRLogger(Callback):
    def __init__(self, log_dir, every_steps=10):
        super().__init__()
        self.every = int(every_steps)
        self.fw = tf.summary.create_file_writer(str(Path(log_dir) / "tb" / "lr"))

    def _current_lr(self):
        opt = self.model.optimizer
        lr = opt.learning_rate
        step_var = opt.iterations  # MirroredVariable
        if callable(lr):  # schedule
            lr_t = lr(step_var)  # pass the tensor-var directly
        else:
            lr_t = tf.convert_to_tensor(lr, dtype=tf.float32)
        return float(K.get_value(lr_t))

    def on_train_batch_end(self, batch, logs=None):
        step_var = self.model.optimizer.iterations  # int64 tensor
        step = int(K.get_value(step_var))           # Python int for modulo
        if step % self.every == 0:
            opt = self.model.optimizer
            lr = opt.learning_rate
            lr_t = lr(step_var) if callable(lr) else tf.convert_to_tensor(lr, tf.float32)
            with self.fw.as_default():
                tf.summary.scalar("learning_rate", lr_t, step=step_var)

class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_seconds):
        super().__init__()
        self.max_seconds = max_seconds

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            print(f"\nTiempo límite alcanzado ({elapsed:.0f}s > {self.max_seconds}s), deteniendo entrenamiento.")
            self.model.stop_training = True

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.epoch_times = []
        print("Training start:", time.strftime("%X"))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        total_time  = time.time() - self.train_start
        print(f"→ Epoch {epoch+1} took {epoch_time:.1f}s, total elapsed {total_time:.1f}s")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start
        print("Training finished at", time.strftime("%X"), f"(total {total_time:.1f}s)")

class CSVLoggerWithEpochTime(tf.keras.callbacks.CSVLogger):
    """CSVLogger that also writes epoch_time seconds per epoch if available via a TimeHistory callback.
    It appends a 'epoch_time' column to the CSV.
    """
    def __init__(self, filename, time_cb: TimeHistory | None = None, **kwargs):
        super().__init__(filename, **kwargs)
        self._time_cb = time_cb
        self._wrote_header = False
    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        if self._time_cb and len(self._time_cb.epoch_times) > epoch:
            logs['epoch_time'] = float(self._time_cb.epoch_times[epoch])
        else:
            logs['epoch_time'] = None
        return super().on_epoch_end(epoch, logs)

# --- util: seleccionar la columna de una clase cuando y_true/y_pred son one-hot ---
def _select_col(y_true, y_pred, class_index: int):
    """Devuelve (y_true, y_pred) de la clase pedida.
    - one-hot (..,2): selecciona la columna class_index
    - binario (..,1): usa tal cual si class_index==1; invierte si class_index==0
    Funciona tanto en modo ventana como time-step (se aplanará internamente en métricas Keras).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    last_rank = y_pred.shape.rank
    last_dim = (y_pred.shape[-1] if last_rank is not None else None)

    if last_dim == 2:
        y_true_c = y_true[..., class_index]
        y_pred_c = y_pred[..., class_index]
        return y_true_c, y_pred_c

    # Binario o desconocido
    if last_dim == 1:
        y_true_b = tf.squeeze(y_true, axis=-1)
        y_pred_b = tf.squeeze(y_pred, axis=-1)
    else:
        # ya vector
        y_true_b = y_true
        y_pred_b = y_pred

    # If labels might be soft (window_mode=="soft"), downstream metrics expect binary ground truth.
    # We'll threshold y_true at 0.5 here.
    y_true_bin = tf.cast(y_true_b >= 0.5, tf.float32)
    if int(class_index) == 0:
        return (1.0 - y_true_bin), (1.0 - y_pred_b)
    else:
        return y_true_bin, y_pred_b

# --- AUC por clase (PR o ROC) ---
class AUCClass(tf.keras.metrics.AUC):
    def __init__(self, class_index: int, curve="PR", name=None, **kw):
        super().__init__(curve=curve, name=name or f"{curve.lower()}_auc_c{class_index}", **kw)
        self.class_index = int(class_index)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        return super().update_state(y_true, y_pred, sample_weight)

# --- contadores TP/TN/FP/FN por clase (umbral 0.5) ---
class TruePositivesClass(tf.keras.metrics.Metric):
    def __init__(self, class_index: int, threshold=0.5, name=None, **kw):
        super().__init__(name=name or f"tp_c{class_index}", **kw)
        self.class_index = int(class_index); self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros", dtype=self.dtype)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        y_hat = tf.cast(y_pred >= self.threshold, self.dtype)
        y_true = tf.cast(y_true, self.dtype)
        val = y_hat * y_true
        if sample_weight is not None:
            val = tf.cast(sample_weight, self.dtype) * val
        self.tp.assign_add(tf.reduce_sum(val))
    def result(self): return self.tp
    def reset_states(self): self.tp.assign(0.0)

class TrueNegativesClass(tf.keras.metrics.Metric):
    def __init__(self, class_index: int, threshold=0.5, name=None, **kw):
        super().__init__(name=name or f"tn_c{class_index}", **kw)
        self.class_index = int(class_index); self.threshold = float(threshold)
        self.tn = self.add_weight(name="tn", shape=(), initializer="zeros", dtype=self.dtype)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        y_hat = tf.cast(y_pred >= self.threshold, self.dtype)
        y_true = tf.cast(y_true, self.dtype)
        val = (1.0 - y_hat) * (1.0 - y_true)
        if sample_weight is not None:
            val = tf.cast(sample_weight, self.dtype) * val
        self.tn.assign_add(tf.reduce_sum(val))
    def result(self): return self.tn
    def reset_states(self): self.tn.assign(0.0)

class FalsePositivesClass(tf.keras.metrics.Metric):
    def __init__(self, class_index: int, threshold=0.5, name=None, **kw):
        super().__init__(name=name or f"fp_c{class_index}", **kw)
        self.class_index = int(class_index); self.threshold = float(threshold)
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros", dtype=self.dtype)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        y_hat = tf.cast(y_pred >= self.threshold, self.dtype)
        y_true = tf.cast(y_true, self.dtype)
        val = y_hat * (1.0 - y_true)
        if sample_weight is not None:
            val = tf.cast(sample_weight, self.dtype) * val
        self.fp.assign_add(tf.reduce_sum(val))
    def result(self): return self.fp
    def reset_states(self): self.fp.assign(0.0)

class FalseNegativesClass(tf.keras.metrics.Metric):
    def __init__(self, class_index: int, threshold=0.5, name=None, **kw):
        super().__init__(name=name or f"fn_c{class_index}", **kw)
        self.class_index = int(class_index); self.threshold = float(threshold)
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros", dtype=self.dtype)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        y_hat = tf.cast(y_pred >= self.threshold, self.dtype)
        y_true = tf.cast(y_true, self.dtype)
        val = (1.0 - y_hat) * y_true
        if sample_weight is not None:
            val = tf.cast(sample_weight, self.dtype) * val
        self.fn.assign_add(tf.reduce_sum(val))
    def result(self): return self.fn
    def reset_states(self): self.fn.assign(0.0)

# --- Precision y Recall por clase (Keras ya ofrece class_id, pero este wrapper respeta sample_weight binario) ---
class PrecisionClass(tf.keras.metrics.Precision):
    def __init__(self, class_index: int, name=None, **kw):
        super().__init__(name=name or f"precision_c{class_index}", **kw)
        self.class_index = int(class_index)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        return super().update_state(y_true, y_pred, sample_weight)

class RecallClass(tf.keras.metrics.Recall):
    def __init__(self, class_index: int, name=None, **kw):
        super().__init__(name=name or f"recall_c{class_index}", **kw)
        self.class_index = int(class_index)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        return super().update_state(y_true, y_pred, sample_weight)

# --- métricas “At …” por clase (basadas en curvas internas) ---
class PrecisionAtRecallClass(tf.keras.metrics.PrecisionAtRecall):
    def __init__(self, class_index: int, recall=0.90, name=None, **kw):
        super().__init__(recall=recall, name=name or f"precision_at_recall{int(100*recall)}_c{class_index}", **kw)
        self.class_index = int(class_index)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        return super().update_state(y_true, y_pred, sample_weight)

class RecallAtPrecisionClass(tf.keras.metrics.RecallAtPrecision):
    def __init__(self, class_index: int, precision=0.90, name=None, **kw):
        super().__init__(precision=precision, name=name or f"recall_at_precision{int(100*precision)}_c{class_index}", **kw)
        self.class_index = int(class_index)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        return super().update_state(y_true, y_pred, sample_weight)

class SensitivityAtSpecificityClass(tf.keras.metrics.SensitivityAtSpecificity):
    def __init__(self, class_index: int, specificity=0.995, name=None, **kw):
        super().__init__(specificity=specificity, name=name or f"sensitivity_at_spec{int(1000*specificity)}_c{class_index}", **kw)
        self.class_index = int(class_index)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        return super().update_state(y_true, y_pred, sample_weight)

class SpecificityAtSensitivityClass(tf.keras.metrics.SpecificityAtSensitivity):
    def __init__(self, class_index: int, sensitivity=0.90, name=None, **kw):
        super().__init__(sensitivity=sensitivity, name=name or f"specificity_at_sens{int(100*sensitivity)}_c{class_index}", **kw)
        self.class_index = int(class_index)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        return super().update_state(y_true, y_pred, sample_weight)

# --- F1 por clase con umbral fijo (rápido y estable) ---
class F1Class(tf.keras.metrics.Metric):
    def __init__(self, class_index: int, threshold=0.5, name=None, **kw):
        super().__init__(name=name or f"f1_c{class_index}", **kw)
        self.class_index = int(class_index)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros", dtype=self.dtype)
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros", dtype=self.dtype)
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros", dtype=self.dtype)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        y_true = tf.cast(y_true, tf.float32)
        y_hat = tf.cast(y_pred >= self.threshold, tf.float32)
        tp = y_hat * y_true
        fp = y_hat * (1.0 - y_true)
        fn = (1.0 - y_hat) * y_true
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            tp = sw * tp; fp = sw * fp; fn = sw * fn
        self.tp.assign_add(tf.reduce_sum(tp))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))
    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall    = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        return tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
    def reset_states(self):
        for v in (self.tp, self.fp, self.fn): v.assign(0.0)

# --- Balanced Accuracy y Brier (útiles para balance/calibración, independientes de la clase) ---
class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, class_index: int = 1, threshold=0.5, name="balanced_accuracy", **kw):
        super().__init__(name=name, **kw)
        self.class_index = int(class_index)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros", dtype=self.dtype)
        self.tn = self.add_weight(name="tn", shape=(), initializer="zeros", dtype=self.dtype)
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros", dtype=self.dtype)
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros", dtype=self.dtype)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        y_true = tf.cast(y_true, tf.float32)
        y_hat  = tf.cast(y_pred >= self.threshold, tf.float32)
        tp = y_hat * y_true
        tn = (1.0 - y_hat) * (1.0 - y_true)
        fp = y_hat * (1.0 - y_true)
        fn = (1.0 - y_hat) * y_true
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            tp = sw * tp; tn = sw * tn; fp = sw * fp; fn = sw * fn
        self.tp.assign_add(tf.reduce_sum(tp))
        self.tn.assign_add(tf.reduce_sum(tn))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))
    def result(self):
        tpr = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        tnr = tf.math.divide_no_nan(self.tn, self.tn + self.fp)
        return 0.5 * (tpr + tnr)
    def reset_states(self):
        for v in (self.tp, self.tn, self.fp, self.fn): v.assign(0.0)

class BrierPos(tf.keras.metrics.Metric):
    def __init__(self, class_index: int = 1, name="brier_pos", **kw):
        super().__init__(name=name, **kw)
        self.class_index = int(class_index)
        self.total = self.add_weight(name="total", shape=(), initializer="zeros", dtype=self.dtype)
        self.count = self.add_weight(name="count", shape=(), initializer="zeros", dtype=self.dtype)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _select_col(y_true, y_pred, self.class_index)
        err = tf.square(y_pred - y_true)
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            err = sw * err
            n = tf.reduce_sum(sw)
        else:
            n = tf.cast(tf.size(err), tf.float32)
        self.total.assign_add(tf.reduce_sum(err))
        self.count.assign_add(n)
    def result(self): return tf.math.divide_no_nan(self.total, self.count)
    def reset_states(self): self.total.assign(0.0); self.count.assign(0.0)

class ParetoCheckpointMulti(tf.keras.callbacks.Callback):
    """
    Guarda el modelo cuando el vector de métricas 'metrics' (con direcciones en 'directions')
    domina al mejor hasta ahora (tolerancias 'eps'). Opcionalmente aplica 'constraints'
    (p.ej. ("val_fa_per_24h", "<=", 2.0)). Mantiene UN mejor vector (no archivo Pareto).
    """
    def __init__(self,
                 filepath: str,
                 metrics: list,                 # p.ej. ["val_recall_seizure","val_recall_background"]
                 directions: list,              # p.ej. ["max","max"] o ["max","min",...]
                 eps: list = None,              # tolerancia por métrica (misma longitud que metrics)
                 constraints: list = None,      # lista de (key, op, value), p.ej. [("val_fa_per_24h","<=",2.0)]
                 save_weights_only: bool = False,
                 verbose: int = 1):
        super().__init__()
        assert len(metrics) == len(directions), "metrics y directions deben tener misma longitud"
        self.filepath = filepath
        self.metrics = metrics
        self.directions = [d.lower() for d in directions]
        self.eps = eps if eps is not None else [1e-4] * len(metrics)
        self.constraints = constraints or []
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self._best_vec = None   # lista de floats en el orden de self.metrics

    @staticmethod
    def _ok_constraint(val, op, target):
        if val is None or not math.isfinite(float(val)):
            return False
        if   op == ">=": return val >= target
        elif op == "<=": return val <= target
        elif op ==  ">": return val >  target
        elif op ==  "<": return val <  target
        elif op == "==": return abs(val - target) < 1e-12
        else: raise ValueError(f"Operador no soportado: {op}")

    def _dominates(self, cand, best):
        """cand domina a best si es >= (o <=) con tolerancia en todas y mejora estricta en al menos una."""
        better_or_equal_all = True
        strictly_better = False
        for i, (c, b, dir_i, eps_i) in enumerate(zip(cand, best, self.directions, self.eps)):
            if dir_i == "max":
                if c < b - eps_i:   # peor que best
                    better_or_equal_all = False; break
                if c > b + eps_i:   # claramente mejor en esta
                    strictly_better = True
            elif dir_i == "min":
                if c > b + eps_i:
                    better_or_equal_all = False; break
                if c < b - eps_i:
                    strictly_better = True
            else:
                raise ValueError(f"dirección inválida: {dir_i}")
        return better_or_equal_all and strictly_better

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # 1) Chequear restricciones (si las hay)
        for key, op, target in self.constraints:
            v = logs.get(key, None)
            if not self._ok_constraint(float(v) if v is not None else None, op, float(target)):
                return  # no guarda si no cumple la puerta

        # 2) Construir vector candidato
        cand = []
        for k in self.metrics:
            v = logs.get(k, None)
            if v is None or not math.isfinite(float(v)):
                return  # alguna métrica no disponible todavía
            cand.append(float(v))

        # 3) Primera vez: guardar
        if self._best_vec is None:
            self._best_vec = cand
            path = self.filepath.format(epoch=epoch+1, **{m: v for m, v in zip(self.metrics, cand)})
            if self.save_weights_only:
                self.model.save_weights(path)
            else:
                self.model.save(path)
            if self.verbose:
                print(f"\n[pareto_multi] saved (init): {path}  " +
                      " ".join([f"{m}={v:.4f}" for m, v in zip(self.metrics, cand)]))
            return

        # 4) Comparar y guardar si domina
        if self._dominates(cand, self._best_vec):
            self._best_vec = cand
            path = self.filepath.format(epoch=epoch+1, **{m: v for m, v in zip(self.metrics, cand)})
            if self.save_weights_only:
                self.model.save_weights(path)
            else:
                self.model.save(path)
            if self.verbose:
                print(f"\n[pareto_multi] saved: {path}  " +
                      " ".join([f"{m}={v:.4f}" for m, v in zip(self.metrics, cand)]))

# =================================================================================================================
# Loss Functions
# =================================================================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt)
    def wrapped(y_true, y_pred):
        return tf.reduce_mean(loss_fn(y_true, y_pred))
    return wrapped

# =================================================================================================================
# Pipeline Functions
# =================================================================================================================

def estimate_class_weight(ds: tf.data.Dataset, onehot: bool, time_step: bool, max_batches: int = 100) -> Optional[Dict[int, float]]:
    """Approximate class weights from a dataset batch stream (window-level recommended).
    Returns {0: w0, 1: w1} so that expected weight ~ 1. If time_step=True, returns None.
    """
    if time_step:
        # Using class_weight with per-timestep labels is not directly supported
        return None
    pos = 0
    neg = 0
    count_batches = 0
    for x, y in ds.take(max_batches):  # type: ignore
        yb = y.numpy()
        # shapes: (B,1) or (B,2) or (B,)
        if onehot:
            if yb.ndim >= 2:
                yb = yb[..., -1]
        yb = yb.reshape(-1)
        pos += int((yb > 0.5).sum())
        neg += int((yb <= 0.5).sum())
        count_batches += 1
    total = pos + neg
    if total == 0:
        return None
    p = pos / float(total)
    p = min(max(p, 1e-6), 1 - 1e-6)
    # weights so that E[w] = 1: w1*p + w0*(1-p) = 1, with w1 = 0.5/p, w0 = 0.5/(1-p)
    w1 = 0.5 / p
    w0 = 0.5 / (1.0 - p)
    return {0: float(w0), 1: float(w1)}

def inspect_dataset(ds: tf.data.Dataset, num_batches: int | None = 3, positive_label: int = 1):
    """Ligero y compatible con todos los modos (time-step/window, one-hot/binario, soft):
    - Itera pocas batches (por defecto 3)
    - Inferencia de formas para contar positivos y total
    - En soft labels, umbraliza a 0.5
    """
    total_items = 0
    total_pos = 0

    it = ds if num_batches is None else ds.take(num_batches)
    for eeg_batch, y_batch in it:
        y = y_batch
        y_np = y.numpy()
        # Handle one-hot → class indices
        if y_np.ndim >= 1 and y_np.shape[-1] == 2:
            y_np = np.argmax(y_np, axis=-1)
        else:
            # squeeze last dim if (..,1)
            if y_np.ndim >= 1 and y_np.shape[-1] == 1:
                y_np = np.squeeze(y_np, axis=-1)
            # soft labels (float in [0,1]) → binarize at 0.5
            if y_np.dtype != np.int64 and y_np.dtype != np.int32:
                y_np = (y_np >= 0.5).astype(np.int32)

        # Now y_np is integers 0/1 possibly with time axis; count all elements
        total_items += y_np.size
        total_pos += int((y_np == positive_label).sum())

    prev = (total_pos / total_items) if total_items else 0.0
    print(f"\nTotal: {total_pos}/{total_items} positives → {prev:.2%} prevalence")

def _flatten_inputs(inputs):
    return tf.nest.flatten(inputs)


def autodetect_feat_dim_from_dataset(ds):
    """Peek one element of `ds` and return (feat_dim_or_None, has_domain_bool).
    - EEG is assumed to be rank-3 (B,T,C)
    - Feature vector is rank-2 (B,F)
    - Domain id is rank-1 (B,)
    This function performs NO mapping; it only reads shapes.
    """
    if ds is None:
        return None, False
    for elem in ds.take(1):
        # elem can be (inputs, y) or ((inputs...), y)
        if isinstance(elem, (tuple, list)) and len(elem) == 2:
            inputs, _ = elem
        else:
            # unexpected, just bail out gracefully
            return None, False
        flat = _flatten_inputs(inputs)
        eeg = next((t for t in flat if getattr(t, 'shape', None) is not None and len(t.shape) == 3), None)
        feats = next((t for t in flat if getattr(t, 'shape', None) is not None and len(t.shape) == 2), None)
        domain = next((t for t in flat if getattr(t, 'shape', None) is not None and len(t.shape) == 1), None)
        feat_dim = int(feats.shape[-1]) if feats is not None and feats.shape.rank == 2 else None
        return feat_dim, (domain is not None)
    return None, False

def pipeline(data_dir: str, n_channels: int = 22, n_timepoints: int = 256, write_ds=True, limits=None):

    if write_ds:
        write_tfrecord_splits_FINAL_CORRECTED(data_dir, TFRECORD_DIR, montage='ar', resample_fs=PREPROCESS['resample'], limits=limits, window_sec=WINDOW_SEC, hop_sec=FRAME_HOP_SEC)
    # respetar el parámetro n_channels recibido
    # Save run configuration snapshot
    save_run_config(RUN_DIR, extra={
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "data_dir": data_dir,
    })

    # Prefer sharded per-EDF datasets if present, else fallback to monolith
    train_glob = os.path.join(TFRECORD_DIR, 'train', '*.tfrecord')
    val_glob   = os.path.join(TFRECORD_DIR, 'val', '*.tfrecord')
    train_files = sorted(glob.glob(train_glob))
    val_files   = sorted(glob.glob(val_glob))
    if len(train_files) == 0:
        mono_train = os.path.join(TFRECORD_DIR, 'train.tfrecord')
        if os.path.exists(mono_train):
            train_files = [mono_train]
    if len(val_files) == 0:
        mono_val = os.path.join(TFRECORD_DIR, 'val.tfrecord')
        if os.path.exists(mono_val):
            val_files = [mono_val]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No TFRecord files found for training in '{TFRECORD_DIR}'. Looked for {train_glob} and train.tfrecord")
    if len(val_files) == 0:
        raise FileNotFoundError(f"No TFRecord files found for validation in '{TFRECORD_DIR}'. Looked for {val_glob} and val.tfrecord")

    train_paths = train_files
    val_paths   = val_files

    def _make_train_ds(paths):
        return create_dataset_final_v2(
            train_paths,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            batch_size=BATCH_SIZE,
            one_hot=ONEHOT,
            time_step=TIME_STEP,
            cache=False,
            drop_remainder=False,
            shuffle=True,
            shuffle_buffer=4096,
            window_mode=WINDOW_MODE,
            include_label_feats=INCLUDE_LABEL_FEATS,
            sample_rate=PREPROCESS['resample'],
            feature_names=FEATURE_NAMES,
            return_feature_vector=FEATURES_AS_VECTOR,
            prefetch=True,
            balance_strategy='undersample' if BALANCED_STREAM else 'none',
            enable_dmd_features=ENABLE_DMD_FEATURES,
            dmd_n_modes=DMD_OPTS.get('n_modes', 8),
            dmd_subsample=DMD_OPTS.get('subsample', 2),
            return_domain_id=RETURN_DOMAIN_ID,
            domain_from=DOMAIN_FROM,
            domain_to_int=DOMAIN_TO_INT,
            domain_num_buckets=DOMAIN_NUM_BUCKETS,
            include_patients=INCLUDE_PATIENTS,
            exclude_patients=EXCLUDE_PATIENTS,
            balance_pos_frac=BALANCED_STREAM if BALANCED_STREAM else None,
            use_domain_in_output=False,
        
        )
    def _make_count_ds(paths):
        # Para estimar steps y class weights no aplicamos balance
        return create_dataset_final_v2(
            train_paths,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            batch_size=BATCH_SIZE,
            one_hot=ONEHOT,
            time_step=TIME_STEP,
            cache=False,
            drop_remainder=False,
            shuffle=False,
            shuffle_buffer=4096,
            window_mode=WINDOW_MODE,
            include_label_feats=INCLUDE_LABEL_FEATS,
            sample_rate=PREPROCESS['resample'],
            feature_names=FEATURE_NAMES,
            return_feature_vector=FEATURES_AS_VECTOR,
            prefetch=False,
            balance_strategy='undersample' if BALANCED_STREAM else 'none',
            enable_dmd_features=ENABLE_DMD_FEATURES,
            dmd_n_modes=DMD_OPTS.get('n_modes', 8),
            dmd_subsample=DMD_OPTS.get('subsample', 2),
            return_domain_id=RETURN_DOMAIN_ID,
            domain_from=DOMAIN_FROM,
            domain_to_int=DOMAIN_TO_INT,
            domain_num_buckets=DOMAIN_NUM_BUCKETS,
            include_patients=INCLUDE_PATIENTS,
            exclude_patients=EXCLUDE_PATIENTS,
            balance_pos_frac=None,
            use_domain_in_output=False,
        )
    # Count steps safely (cardinality may be UNKNOWN due to flat_map)
    _count_ds = _make_count_ds(train_paths)
    card = tf.data.experimental.cardinality(_count_ds)
    if card == tf.data.experimental.UNKNOWN_CARDINALITY:
        steps_per_epoch = sum(1 for _ in _count_ds)
    else:
        steps_per_epoch = int(card.numpy())
    print("steps_per_epoch:", steps_per_epoch)

    # Rebuild clean datasets for fit()
    train_ds = _make_train_ds(train_paths).repeat()

    est_epochs_budget = 20
    est_total_steps = int(est_epochs_budget * steps_per_epoch)
    warmup_steps    = max(1, int(WARMUP_RATIO * est_total_steps))
    decay_steps     = max(1, est_total_steps - warmup_steps)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=decay_steps,
        alpha=MIN_LR_FRACTION,
        warmup_target=LEARNING_RATE,
        warmup_steps=warmup_steps
    )

    if BALANCED_STREAM is not None and BALANCED_STREAM > 0.0:
        print(f"Internal balancing active (strategy=rejection) target pos frac {BALANCED_STREAM}")

    val_ds = create_dataset_final_v2(
            val_paths,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            batch_size=BATCH_SIZE,
            one_hot=ONEHOT,
            time_step=TIME_STEP,
            cache=False,
            drop_remainder=False,
            shuffle=False,
            shuffle_buffer=4096,
            window_mode=WINDOW_MODE,
            include_label_feats=INCLUDE_LABEL_FEATS,
            sample_rate=PREPROCESS['resample'],
            feature_names=FEATURE_NAMES,
            return_feature_vector=FEATURES_AS_VECTOR,
            prefetch=True,
            balance_strategy='undersample' if BALANCED_STREAM else 'none',
            enable_dmd_features=ENABLE_DMD_FEATURES,
            dmd_n_modes=DMD_OPTS.get('n_modes', 8),
            dmd_subsample=DMD_OPTS.get('subsample', 2),
            return_domain_id=RETURN_DOMAIN_ID,
            domain_from=DOMAIN_FROM,
            domain_to_int=DOMAIN_TO_INT,
            domain_num_buckets=DOMAIN_NUM_BUCKETS,
            include_patients=INCLUDE_PATIENTS,
            exclude_patients=EXCLUDE_PATIENTS,
            balance_pos_frac=None,
            use_domain_in_output=False,
    )

    inspect_dataset(_count_ds, num_batches=1000)
    
    inspect_dataset(val_ds, num_batches=1000)

    feat_dim_detected, had_domain = autodetect_feat_dim_from_dataset(train_ds)

    # Compute effective input channels and whether we have a separate features vector
    eff_channels = n_channels
    use_feat_vec = (FEATURES_AS_VECTOR and WINDOW_MODE == "features" and not TIME_STEP)
    feat_input_dim = feat_input_dim = int(feat_dim_detected) if feat_dim_detected is not None else None
    if WINDOW_MODE == "features" and not TIME_STEP and not FEATURES_AS_VECTOR:
        # concatenated to channels per time step
        eff_channels = n_channels + len(FEATURE_NAMES)

    if CLASS_WEIGHTS:
        class_weight = estimate_class_weight(_count_ds, onehot=ONEHOT, time_step=TIME_STEP, max_batches=100)
        if class_weight is not None:
            print("Using estimated class weights:", class_weight)
        else:
            print("Could not estimate class weights, proceeding without them.")

    strategy = tf.distribute.OneDeviceStrategy("GPU:0")
    with strategy.scope():
        # Loss selection respects FOCAL flag
        if ONEHOT and not TVERSKY:
            loss_fn = (CategoricalFocalCrossentropy(alpha=[0.25, 0.55], gamma=1.5, label_smoothing=0.0, from_logits=False)
                       if FOCAL else
                       CategoricalCrossentropy(from_logits=False))
        elif not TVERSKY:
            loss_fn = (BinaryFocalCrossentropy(alpha=0.75, gamma=3)
                       if FOCAL else
                       BinaryCrossentropy(from_logits=False))
        else:
            print("Tversky")
            loss_fn = Tversky(alpha=0.5, beta=0.5)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-3,
            use_ema=True,
            ema_momentum=0.999,
            ema_overwrite_frequency=None
        )
        common_kwargs = dict(
            feat_input_dim=feat_input_dim if (FEATURES_AS_VECTOR and WINDOW_MODE.startswith('soft') or WINDOW_MODE=='features') and not TIME_STEP else None,
            koopman_latent_dim=KOOPMAN_LATENT_DIM,
            koopman_loss_weight=KOOPMAN_LOSS_WEIGHT,
            use_reconstruction_head=USE_RECONSTRUCTION_HEAD,
            recon_weight=RECON_WEIGHT,
            recon_target=RECON_TARGET,
            bottleneck_dim=BOTTLENECK_DIM,
            expand_dim=EXPAND_DIM,
        )

        # Build model inside strategy scope
        match MODEL:
                case 'HYB':
                    model = build_hybrid(
                        input_shape=(n_timepoints, eff_channels),
                        num_classes=NUM_CLASSES,
                        one_hot=ONEHOT,
                        time_step=TIME_STEP,
                        feat_input_dim=feat_input_dim,
                        conv_type="conv",             # "conv" o "separable"
                        num_filters=TCN_FILTERS,
                        kernel_size=TCN_KERNEL_SIZE,
                        se_ratio=SE_RATIO,
                        dropout_rate=DROPOUT_RATE,
                        num_heads=NUM_ATTENTION_HEADS,
                        rnn_units=RNN_UNITS,
                        use_se_after_cnn=True,
                        use_se_after_rnn=True,
                        use_between_attention=True,
                        use_final_attention=True
                    )
                case 'TRANS':
                    model = build_transformer(
                        input_shape=(n_timepoints, eff_channels),
                        num_classes=NUM_CLASSES,
                        embed_dim=EMBED_DIMENSION,
                        num_layers=TRANS_LAYERS,
                        num_heads=ATTENTION_HEADS,
                        mlp_dim=MLP_DIMENSION,
                        dropout_rate=DROPOUT_RATE,
                        time_step_classification=TIME_STEP,
                        one_hot=ONEHOT,
                        use_se=USE_SE,
                        se_ratio=SE_RATIO,
                        **common_kwargs,
                    )
                case _:
                    model = build_tcn(
                        input_shape=(n_timepoints, eff_channels),
                        num_classes=NUM_CLASSES,
                        num_filters=TCN_FILTERS,
                        kernel_size=TCN_KERNEL_SIZE,
                        dropout_rate=DROPOUT_RATE,
                        num_blocks=TCN_BLOCKS,
                        time_step_classification=TIME_STEP,
                        one_hot=ONEHOT,
                        hpc=HPC,
                        separable=False,
                        se_ratio=SE_RATIO,
                        cycle_dilations=(1,2,4,8),
                        use_attention_pool_win=True,
                        **common_kwargs,
                    )

        # Print receptive field info for TCN-like models
        if MODEL == 'TCN':
            t_steps = n_timepoints
            fs = PREPROCESS['resample']
            print_receptive_field_tcn(t_steps, fs, TCN_KERNEL_SIZE, TCN_BLOCKS, run_dir=RUN_DIR)

        # Metrics: simpler set for binary (ONEHOT=False), richer for one-hot multiclass
        if not ONEHOT:
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                AUCClass(class_index=1, curve="PR",  name="pr_auc"),
                AUCClass(class_index=1, curve="ROC", name="roc_auc"),
                PrecisionClass(1, name="precision"),
                RecallClass(1,    name="recall"),
                F1Class(1,        name="f1"),
                # positive-class confusion-matrix counters
                TruePositivesClass(1, name="tp"),
                TrueNegativesClass(1, name="tn"),
                FalsePositivesClass(1, name="fp"),
                FalseNegativesClass(1, name="fn"),
                PrecisionAtRecallClass(1, recall=0.90, name="precision_at_recall90"),
                RecallAtPrecisionClass(1, precision=0.90, name="recall_at_precision90"),
                SensitivityAtSpecificityClass(1, specificity=0.995, name="sens_at_spec99p5"),
                SpecificityAtSensitivityClass(1, sensitivity=0.90, name="spec_at_sens90"),
                BrierPos(class_index=1, name="brier_pos_seizure"),
                BalancedAccuracy(class_index=1, name="balanced_accuracy"),
            ]
        else:
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                AUCClass(class_index=1, curve="PR",  name="pr_auc_seizure"),
                AUCClass(class_index=1, curve="ROC", name="roc_auc_seizure"),
                AUCClass(class_index=0, curve="PR",  name="pr_auc_background"),
                AUCClass(class_index=0, curve="ROC", name="roc_auc_background"),
                PrecisionClass(1, name="precision_seizure"),
                RecallClass(1,    name="recall_seizure"),
                PrecisionClass(0, name="precision_background"),
                RecallClass(0,    name="recall_background"),
                F1Class(1, name="f1_seizure"),
                F1Class(0, name="f1_background"),
                PrecisionAtRecallClass(1, recall=0.90, name="precision_at_recall90_c1_s"),
                RecallAtPrecisionClass(1, precision=0.90, name="recall_at_precision90_c1_s"),
                SensitivityAtSpecificityClass(1, specificity=0.995, name="sens_at_spec99p5_c1_s"),
                SpecificityAtSensitivityClass(1, sensitivity=0.90, name="spec_at_sens90_c1_s"),
                PrecisionAtRecallClass(0, recall=0.90, name="precision_at_recall90_c1_b"),
                RecallAtPrecisionClass(0, precision=0.90, name="recall_at_precision90_c1_b"),
                SensitivityAtSpecificityClass(0, specificity=0.995, name="sens_at_spec99p5_c1_b"),
                SpecificityAtSensitivityClass(0, sensitivity=0.90, name="spec_at_sens90_c1_b"),
                TruePositivesClass(1, name="tp_seizure"),
                TrueNegativesClass(1, name="tn_seizure"),
                FalsePositivesClass(1, name="fp_seizure"),
                FalseNegativesClass(1, name="fn_seizure"),
                TruePositivesClass(0, name="tp_background"),
                TrueNegativesClass(0, name="tn_background"),
                FalsePositivesClass(0, name="fp_background"),
                FalseNegativesClass(0, name="fn_background"),
                BrierPos(class_index=1, name="brier_pos_seizure"),
                BalancedAccuracy(class_index=1, name="balanced_accuracy"),
            ]
        model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics,
                jit_compile=JIT_COMPILE_MODEL
            )

    # Save model summary after build
    save_model_summary(model, RUN_DIR)

    # Pareto/selector: in binary mode, monitor a single primary metric; in one-hot, keep dual metrics
    if not ONEHOT:
        pareto_primary = ParetoCheckpointMulti(
                filepath=str(RUN_DIR / "pareto_primary_ep{epoch:03d}.keras"),
                metrics=["val_pr_auc"],
                directions=["max"],
                eps=[1e-4],
                constraints=[],
                save_weights_only=False, verbose=1
            )
    else:
        pareto_primary = ParetoCheckpointMulti(
                filepath=str(RUN_DIR / "pareto_both_recalls_ep{epoch:03d}.keras"),
                metrics=["val_recall_seizure", "val_recall_background"],
                directions=["max", "max"],
                eps=[1e-4, 1e-4],
                constraints=[],
                save_weights_only=False, verbose=1
            )

    if not ONEHOT:
        pareto_auc = ParetoCheckpointMulti(
                filepath=str(RUN_DIR / "pareto_auc_ep{epoch:03d}.keras"),
                metrics=["val_pr_auc"],
                directions=["max"],
                eps=[1e-4],
                save_weights_only=False, verbose=1
            )
    else:
        pareto_auc = ParetoCheckpointMulti(
                filepath=str(RUN_DIR / "pareto_auc_both_ep{epoch:03d}.keras"),
                metrics=["val_pr_auc_seizure", "val_pr_auc_background"],
                directions=["max", "max"],
                eps=[1e-4, 1e-4],
                save_weights_only=False, verbose=1
            )

    if not ONEHOT:
        pareto_f1 = ParetoCheckpointMulti(
                filepath=str(RUN_DIR / "pareto_f1_ep{epoch:03d}.keras"),
                metrics=["val_f1"],
                directions=["max"],
                eps=[1e-4],
                save_weights_only=False, verbose=1
            )
    else:
        pareto_f1 = ParetoCheckpointMulti(
                filepath=str(RUN_DIR / "pareto_f1_both_ep{epoch:03d}.keras"),
                metrics=["val_f1_seizure", "val_f1_background"],
                directions=["max", "max"],
                eps=[1e-4, 1e-4],
                save_weights_only=False, verbose=1
            )

    best_by_brier = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(RUN_DIR / "best_by_brier.keras"),
            monitor="val_brier_pos_seizure", # nombre correcto de la métrica
            mode="min",
            save_best_only=True,
            verbose=1
        )

    best_by_frame_recall_ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(RUN_DIR / ("best_by_recall.keras" if not ONEHOT else "best_by_seizure_recall.keras")),
            monitor=("val_recall" if not ONEHOT else "val_recall_seizure"),
            mode="max",
            save_best_only=True,
            verbose=1
        )

        # # keep this only if you compiled with tf.keras.metrics.AUC(curve="PR", name="pr_auc")
    best_by_pr_auc_ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(RUN_DIR / "best_by_pr_auc.keras"),
            monitor="val_pr_auc",         # from model.evaluate on validation_data
            mode="max",
            save_best_only=True,
            verbose=1
        )

    # per-epoch weights for forensic/ablations (doesn't select, just saves) – avoid formatting with unknown metrics
    per_epoch_weights = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(RUN_DIR / "weights" / "ep{epoch:03d}.weights.h5"),
            save_weights_only=True,
            save_best_only=False,
            monitor=("val_recall" if not ONEHOT else "val_recall_seizure"),
            mode="max",
            save_freq="epoch",
            verbose=0
        )

        # --- Early Stopping (choose ONE primary criterion) ---
        # Option B: stop by frame recall (if you still prefer frame-level behavior)
    earlystop_frame = tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="min",
            patience=15,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1
        )

        # --- Housekeeping / instrumentation ---
    backup_cb = tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(RUN_DIR / "bak"),
            save_freq="epoch",
        )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=str(RUN_DIR / "tb_train"),
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq="epoch",
        )
    # define time tracker BEFORE CSV logger that references it
    time_cb = TimeHistory()
    cb_csvlogger = CSVLoggerWithEpochTime(str(RUN_DIR / "training_log.csv"), time_cb)
    # optional: a minimal CSV just for epoch_time if the main CSV has issues
    class EpochTimeCSV(tf.keras.callbacks.Callback):
        def __init__(self, path, time_cb):
            super().__init__()
            self.path = Path(path)
            self.time_cb = time_cb
            self._wrote_header = False
        def on_epoch_end(self, epoch, logs=None):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'a') as f:
                if not self._wrote_header:
                    f.write('epoch,epoch_time\n')
                    self._wrote_header = True
                et = (self.time_cb.epoch_times[epoch] if self.time_cb and len(self.time_cb.epoch_times) > epoch else '')
                f.write(f"{epoch},{et}\n")
    cb_epoch_time_fallback = EpochTimeCSV(RUN_DIR / 'training_epoch_times.csv', time_cb)
    terminate_nan_cb = tf.keras.callbacks.TerminateOnNaN()
    cb_timelimit = TimeLimitCallback(max_seconds=TIME_LIMIT_H*3600)
    lr_logger = LRLogger(RUN_DIR, every_steps=20)
        # pr_logger = PRCurveLogger(val_ds, RUN_DIR, max_batches=800)  # frame-PR diagnostic
    ema_swap = SwapEMAWeights(swap_on_epoch=True)

        # --- FINAL ORDER: things that MODIFY logs (calibrator) must come BEFORE ckpts/earlystop ---
    all_cbs = [
            # 1) metric writers / modifiers (populate logs first)
            ema_swap,           # if it swaps weights for evaluation, keep it before val/eop logging
            # pr_calibrator,      # <-- injects val_event_recall, val_fa_per_24h, etc. into logs

            # 2) selectors (consume logs to save/stop)
            # best_by_event_ckpt,         # uses val_event_recall
            best_by_frame_recall_ckpt,  # uses val_recall_seizure (optional)
            pareto_primary,
            best_by_brier,
            pareto_f1,
            pareto_auc,
            per_epoch_weights,
            # Choose ONE early stop:
            # earlystop_event,            # preferred
            earlystop_frame,          # (disable if using earlystop_event)

            # 3) infra/logging
            backup_cb,
            tensorboard_cb,
            cb_csvlogger,
            lr_logger,#pr_logger,
            terminate_nan_cb,
            cb_timelimit,
            time_cb,
            cb_epoch_time_fallback,
        ]
        # train_ds = train_ds.take(steps_per_epoch)
    if CLASS_WEIGHTS:
        history = model.fit(
                train_ds, 
                validation_data=val_ds, 
                epochs=EPOCHS,
                steps_per_epoch=steps_per_epoch,
                callbacks=all_cbs + [TqdmCallback(verbose=1)],              
                verbose=2,
                class_weight=class_weight
            )
    else:
        history = model.fit(
                train_ds, 
                validation_data=val_ds, 
                epochs=EPOCHS,
                steps_per_epoch=steps_per_epoch,
                callbacks=all_cbs + [TqdmCallback(verbose=1)],              
                verbose=2
            )
    model.save('comp_model.keras')
    print("Model saved to final_model.keras")
    return model, history
        
# =================================================================================================================
# Main execution    
# ================================================================================================================    

if __name__ == "__main__":
    # Example usage
    data_root = '../DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3'
    tn_timepoints = int(WINDOW_SEC * PREPROCESS['resample'])
    print(tn_timepoints)
    model, history = pipeline(data_root, n_timepoints=tn_timepoints, n_channels=22, write_ds=WRITE, limits=LIMITS)
    print("Pipeline completed successfully.")