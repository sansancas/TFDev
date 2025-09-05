# shukeda_optimized.py
# Pipeline EEG — estable, serializable y con métricas/ETA ligeros
# - ONEHOT/BINARIO y TIME_STEP on/off
# - Métricas nativas + custom (BA, Brier, Spec@t, Sens@Spec, Spec@Sens, TP/TN/FP/FN@t)
# - LR Warmup + Cosine, ¡serializable!
# - Checkpoints por múltiples monitores (solo pesos)
# - Entrenamiento con repeat() + steps explícitos
# - Verbose=2 con LightETA (estado/ETA cada N segundos)

import os, re, gc, glob, time, math, logging
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf

# ========================== CONFIG GLOBAL ====================================
ONEHOT        = False          # True: salida one-hot (2 clases); False: binaria (1 salida)
TIME_STEP     = True           # True: salida (B,T,C/1); False: (B,C/1)
WINDOW_SEC    = 5.0
FRAME_HOP_SEC = 3.0
FS_TARGET     = 256
BATCH_SIZE    = 8
EPOCHS        = 100
LEARNING_RATE = 2e-4
WARMUP_RATIO  = 0.1            # fracción de pasos para warmup
MIN_LR_FRAC   = 0.05           # mínimo relativo en CosineDecay
WEIGHT_DECAY  = 1e-3
TIME_LIMIT_H  = 48
LIMITS        = {'train': 30, 'dev': 2, 'eval': 2}
WRITE_TFREC   = True
BALANCE_POS_FRAC = None        # si tu create_dataset_final lo soporta (p.ej. 0.5)

# Métricas: umbrales fijos y objetivos de operación
THRESHOLDS              = [0.30, 0.50, 0.70]         # crea recall@t, specificity@t, TP/TN/FP/FN@t, BA@t
SENS_AT_SPEC_TARGETS    = [0.90, 0.95]    # sensibilidad a especificidad objetivo
SPEC_AT_SENS_TARGETS    = [0.90]          # especificidad a sensibilidad objetivo

# Monitores de checkpoints
PRIMARY_MONITOR         = "val_auc_pr"
SECONDARY_MONITORS      = []  # si vacío, se autollenan según listas de arriba

# Dirs (ajusta a tu estructura)
FULL_REC = '../records2' if ONEHOT else '../bin_records2'
CUT_REC  = '../records_cut2' if ONEHOT else '../bin_records_cut2'
TFRECORD_DIR = CUT_REC
RUNS_DIR = Path("./runs")
RUN_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = RUNS_DIR / f"eeg_seizures_{RUN_STAMP}"

# ======================= IMPORTA DATASET Y MODELO ============================
from dataset import write_tfrecord_splits_FINAL_CORRECTED, create_dataset_final
from models.TCN import build_tcn

# ====================== SILENCIAR / MIXED PRECISION ==========================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try: tf.get_logger().setLevel(logging.ERROR)
except Exception: pass

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Determinismo “razonable”
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # evita pequeñas diferencias numéricas
np.random.seed(42); tf.random.set_seed(42)

# Desactivar barra interactiva (usamos verbose=2 + LightETA)
tf.keras.utils.disable_interactive_logging()

# ====================== UTILIDADES DE FORMAS =================================
def _norm_shapes(y_true, y_pred, is_onehot: bool):
    """
    Devuelve y_true, y_pred en tf.float32 y compatibles.
    - Si ONEHOT: colapsa a prob/etiqueta de clase positiva (índice 1).
    - Si TIME_STEP: replica y_true cuando venga sin eje temporal.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Manejo del eje temporal si aparece en y_pred
    if len(y_pred.shape) == 3:  # (B,T,C/1)
        if len(y_true.shape) == 1:
            y_true = tf.expand_dims(y_true, 1)                 # (B,1)
            y_true = tf.tile(y_true, [1, tf.shape(y_pred)[1]]) # (B,T)
        elif len(y_true.shape) == 2 and (is_onehot and y_true.shape[-1] == 2):
            y_true = tf.expand_dims(y_true, 1)                      # (B,1,2)
            y_true = tf.tile(y_true, [1, tf.shape(y_pred)[1], 1])   # (B,T,2)
        if y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, -1)                    # (B,T)

    if len(y_pred.shape) == 2 and y_pred.shape[-1] == 1:
        y_pred = tf.squeeze(y_pred, -1)                        # (B,)

    # Selección de clase positiva
    if is_onehot:
        if len(y_true.shape) >= 2 and y_true.shape[-1] == 2:
            y_true = y_true[..., 1]
        if len(y_pred.shape) >= 2 and y_pred.shape[-1] == 2:
            y_pred = y_pred[..., 1]

    return y_true, y_pred

def _binarize01(t):  # por si etiquetas llegan como floats
    return tf.cast(t >= 0.5, tf.float32)

# ================= MÉTRICAS PERSONALIZADAS / WRAPPERS ========================
class BalancedAccuracy(tf.keras.metrics.Metric):
    """BA = 0.5*(TPR + TNR) a umbral fijo (threshold)."""
    def __init__(self, threshold=0.5, name="balanced_accuracy", **kw):
        super().__init__(name=name, **kw)
        self.th = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros", dtype="float32")
        self.tn = self.add_weight(name="tn", shape=(), initializer="zeros", dtype="float32")
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros", dtype="float32")
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _norm_shapes(y_true, y_pred, is_onehot=ONEHOT)
        y_true = _binarize01(y_true)
        y_hat = tf.cast(y_pred >= self.th, tf.float32)
        tp = y_hat * y_true
        tn = (1.0 - y_hat) * (1.0 - y_true)
        fp = y_hat * (1.0 - y_true)
        fn = (1.0 - y_hat) * y_true
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            tp, tn, fp, fn = sw*tp, sw*tn, sw*fp, sw*fn
        self.tp.assign_add(tf.reduce_sum(tp))
        self.tn.assign_add(tf.reduce_sum(tn))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self):
        tpr = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        tnr = tf.math.divide_no_nan(self.tn, self.tn + self.fp)
        return 0.5 * (tpr + tnr)

    def reset_states(self):
        for v in (self.tp, self.tn, self.fp, self.fn):
            v.assign(0.0)

class BrierScore(tf.keras.metrics.Metric):
    """Brier score (prob. clase positiva vs etiqueta binaria 0/1)."""
    def __init__(self, name="brier_score", **kw):
        super().__init__(name=name, **kw)
        self.sum_err = self.add_weight(name="sum_err", shape=(), initializer="zeros", dtype="float32")
        self.count   = self.add_weight(name="count",   shape=(), initializer="zeros", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _norm_shapes(y_true, y_pred, is_onehot=ONEHOT)
        y_true = _binarize01(y_true)
        err = tf.square(y_pred - y_true)
        if sample_weight is not None:
            err = err * tf.cast(sample_weight, tf.float32)
        self.sum_err.assign_add(tf.reduce_sum(err))
        self.count.assign_add(tf.cast(tf.size(err), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.sum_err, self.count)

    def reset_states(self):
        self.sum_err.assign(0.0)
        self.count.assign(0.0)

class SpecificityAtThreshold(tf.keras.metrics.Metric):
    """Especificidad (TNR) a umbral fijo."""
    def __init__(self, threshold=0.5, name=None, **kw):
        name = name or f"specificity@t={threshold:.2f}"
        super().__init__(name=name, **kw)
        self.th = float(threshold)
        self.tn = self.add_weight(name="tn", shape=(), initializer="zeros", dtype="float32")
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _norm_shapes(y_true, y_pred, is_onehot=ONEHOT)
        y_true = _binarize01(y_true)
        y_hat  = tf.cast(y_pred >= self.th, tf.float32)
        tn = (1.0 - y_hat) * (1.0 - y_true)
        fp = y_hat * (1.0 - y_true)
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            tn, fp = sw*tn, sw*fp
        self.tn.assign_add(tf.reduce_sum(tn))
        self.fp.assign_add(tf.reduce_sum(fp))

    def result(self):  # TNR
        return tf.math.divide_no_nan(self.tn, self.tn + self.fp)

    def reset_states(self):
        self.tn.assign(0.0); self.fp.assign(0.0)

class ShapeAwareWrapper(tf.keras.metrics.Metric):
    """
    Envuelve una métrica Keras aplicando _norm_shapes primero.
    Útil para Recall/Precision con thresholds, TP/TN/FP/FN y
    SensitivityAtSpecificity / SpecificityAtSensitivity.
    """
    def __init__(self, inner_metric, name=None):
        super().__init__(name=name or inner_metric.name)
        self.inner = inner_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _norm_shapes(y_true, y_pred, is_onehot=ONEHOT)
        return self.inner.update_state(y_true, y_pred, sample_weight)

    def result(self): return self.inner.result()
    def reset_states(self): return self.inner.reset_states()

# ======================== PÉRDIDAS ===========================================
try:
    from tensorflow.keras.losses import BinaryFocalCrossentropy, CategoricalFocalCrossentropy
    HAVE_FOCAL = True
except Exception:
    HAVE_FOCAL = False

def make_loss(is_onehot: bool):
    if is_onehot:
        def loss(y_true, y_pred):
            if TIME_STEP and len(y_pred.shape) == 3 and len(y_true.shape) == 2:
                y_true2 = tf.expand_dims(y_true, 1)
                y_true2 = tf.tile(y_true2, [1, tf.shape(y_pred)[1], 1])
            else:
                y_true2 = y_true
            if HAVE_FOCAL:
                return CategoricalFocalCrossentropy(alpha=[0.25, 0.75], gamma=2.0, label_smoothing=0.1)(y_true2, y_pred)
            else:
                return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)(y_true2, y_pred)
        return loss
    else:
        def loss(y_true, y_pred):
            if TIME_STEP and len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
                y_pred2 = tf.squeeze(y_pred, -1)
            else:
                y_pred2 = y_pred
            if TIME_STEP and len(y_true.shape) == 1 and len(y_pred2.shape) == 2:
                y_true2 = tf.expand_dims(y_true, 1)
                y_true2 = tf.tile(y_true2, [1, tf.shape(y_pred2)[1]])
            else:
                y_true2 = y_true
            if HAVE_FOCAL:
                return BinaryFocalCrossentropy(alpha=0.75, gamma=2.0, label_smoothing=0.1)(y_true2, y_pred2)
            else:
                return tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)(y_true2, y_pred2)
        return loss

# ======================== NOMBRES DE MÉTRICAS =================================
def _mn_sens_at_spec(s): return f"sens@spec>={s:.2f}"
def _mn_spec_at_sens(s): return f"spec@sens>={s:.2f}"
def _mn_recall_t(t):     return f"recall@t={t:.2f}"
def _mn_spec_t(t):       return f"specificity@t={t:.2f}"
def _mn_tp_t(t):         return f"TP@t={t:.2f}"
def _mn_tn_t(t):         return f"TN@t={t:.2f}"
def _mn_fp_t(t):         return f"FP@t={t:.2f}"
def _mn_fn_t(t):         return f"FN@t={t:.2f}"
def _mn_ba_t(t):         return f"balanced_accuracy@t={t:.2f}"
def _mn_prec_t(t):       return f"precision@t={t:.2f}" 

# ======================== MÉTRICAS ===========================================
def make_metrics(is_onehot: bool):
    mets = []
    # Accuracy apropiada según salida
    if is_onehot: mets += [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    else:         mets += [tf.keras.metrics.BinaryAccuracy(name="accuracy")]

    # PR / ROC
    mets += [tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
             tf.keras.metrics.AUC(curve="ROC", name="auc_roc")]

    # Versiones a umbral fijo + cuentas de la matriz de confusión
    for t in THRESHOLDS:
        mets += [ShapeAwareWrapper(tf.keras.metrics.Recall(thresholds=[t]),    name=_mn_recall_t(t))]
        mets += [SpecificityAtThreshold(threshold=t,                          name=_mn_spec_t(t))]
        mets += [ShapeAwareWrapper(tf.keras.metrics.Precision(thresholds=[t]), name=_mn_prec_t(t))]  # <---
        mets += [
            ShapeAwareWrapper(tf.keras.metrics.TruePositives(thresholds=[t]),  name=_mn_tp_t(t)),
            ShapeAwareWrapper(tf.keras.metrics.TrueNegatives(thresholds=[t]),  name=_mn_tn_t(t)),
            ShapeAwareWrapper(tf.keras.metrics.FalsePositives(thresholds=[t]), name=_mn_fp_t(t)),
            ShapeAwareWrapper(tf.keras.metrics.FalseNegatives(thresholds=[t]), name=_mn_fn_t(t)),
        ]
        mets += [BalancedAccuracy(threshold=t, name=_mn_ba_t(t))]

    # Objetivos (sin fijar umbral; barrido interno)
    for s in SENS_AT_SPEC_TARGETS:
        mets += [ShapeAwareWrapper(tf.keras.metrics.SensitivityAtSpecificity(specificity=s), name=_mn_sens_at_spec(s))]
    for s in SPEC_AT_SENS_TARGETS:
        mets += [ShapeAwareWrapper(tf.keras.metrics.SpecificityAtSensitivity(sensitivity=s), name=_mn_spec_at_sens(s))]

    # Alias BA principal (usa el primer umbral)
    mets += [BalancedAccuracy(threshold=THRESHOLDS[0], name="balanced_accuracy")]

    # Calibración
    mets += [BrierScore(name="brier_score")]
    return mets

# ============== LR schedule: Warmup + CosineDecay serializable ================
@tf.keras.utils.register_keras_serializable(package="sched")
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_ratio=0.1, min_lr_frac=0.05, name=None):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(max(1, total_steps))
        self.warmup_ratio = float(warmup_ratio)
        self.min_lr_frac = float(min_lr_frac)

        self.warmup_steps = int(max(1, self.warmup_ratio * self.total_steps))
        self.decay_steps  = int(max(1, self.total_steps - self.warmup_steps))
        self.min_lr = self.base_lr * self.min_lr_frac

        self.cosine = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.base_lr,
            decay_steps=self.decay_steps,
            alpha=self.min_lr / self.base_lr
        )
        self._name = name or "WarmupCosineSchedule"

    def __call__(self, step):
        step = tf.cast(step, tf.int32)
        def _warm():
            return self.base_lr * tf.cast(step, tf.float32) / float(self.warmup_steps)
        def _cos():
            return self.cosine(tf.cast(step - self.warmup_steps, tf.float32))
        return tf.cond(step < self.warmup_steps, _warm, _cos)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_ratio": self.warmup_ratio,
            "min_lr_frac": self.min_lr_frac,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# =================== UTILIDADES DATA/LOGGING/CALLBACKS =======================
def cardinality_safe(ds, fallback_scan=True, max_scan=200000):
    card = int(tf.data.experimental.cardinality(ds).numpy())
    if card < 0 and fallback_scan:  # UNKNOWN_CARDINALITY
        card = sum(1 for _ in ds.take(max_scan))
    return max(1, card)

class SafeTimeCSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename): super().__init__(); self.filename = filename; self.f=None; self.start=None; self.epoch_start=None
    def on_train_begin(self, logs=None):
        self.start = time.time(); self.f = open(self.filename, "w", buffering=1)
        self.f.write("epoch,epoch_time_seconds,total_time_seconds\n")
    def on_epoch_begin(self, epoch, logs=None): self.epoch_start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        et = time.time() - self.epoch_start; tt = time.time() - self.start
        self.f.write(f"{epoch},{et:.3f},{tt:.3f}\n")
    def on_train_end(self, logs=None):
        if self.f: self.f.close()

class SafeTimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_seconds, grace_seconds=300):
        super().__init__(); self.max_seconds=max_seconds; self.grace=grace_seconds; self.start=None
    def on_train_begin(self, logs=None):
        self.start=time.time(); print(f"⏰ Límite de tiempo: {self.max_seconds/3600:.1f} h")
    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start > self.max_seconds:
            print("⏰ Tiempo agotado; deteniendo entrenamiento tras cerrar época."); self.model.stop_training=True

class SafeBackupAndRestore(tf.keras.callbacks.Callback):
    def __init__(self, backup_dir: Path):
        super().__init__(); self.dir = Path(backup_dir); self.dir.mkdir(parents=True, exist_ok=True)
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0 and epoch > 0:
            path = self.dir / f"backup_epoch_{epoch:03d}.weights.h5"
            try:
                self.model.save_weights(str(path))
                for p in sorted(self.dir.glob("backup_epoch_*.weights.h5"))[:-3]:
                    p.unlink(missing_ok=True)
            except Exception as e:
                print(f"⚠️ Backup warning @epoch {epoch}: {e}")

class ThresholdReportCSV(tf.keras.callbacks.Callback):
    def __init__(self, run_dir: Path, thresholds, split_keys=("","val_"), filename="reports/threshold_report.csv"):
        super().__init__()
        self.thresholds = [float(t) for t in thresholds]
        self.split_keys = split_keys  # ("", "val_") → train y val
        self.path = Path(run_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

    @staticmethod
    def _get(d, k): 
        v = d.get(k); 
        return float(v) if v is not None else float("nan")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        rows, best_val = [], {"f1": -1, "t": None}
        for prefix in self.split_keys:  # "" → train, "val_" → val
            split = "val" if prefix == "val_" else "train"
            for t in self.thresholds:
                k = lambda name: f"{prefix}{name}@t={t:.2f}"
                rec = self._get(logs, k("recall"))
                spe = self._get(logs, k("specificity"))
                pre = self._get(logs, k("precision"))
                tp  = self._get(logs, k("TP"))
                tn  = self._get(logs, k("TN"))
                fp  = self._get(logs, k("FP"))
                fn  = self._get(logs, k("FN"))
                ba  = self._get(logs, f"{prefix}balanced_accuracy@t={t:.2f}")
                # F1 a partir de precision/recall si existen
                denom = (pre + rec) if math.isfinite(pre) and math.isfinite(rec) and (pre+rec)>0 else float("nan")
                f1 = (2*pre*rec/denom) if math.isfinite(denom) else float("nan")
                rows.append({
                    "epoch": epoch+1, "split": split, "t": t,
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "recall": rec, "specificity": spe, "precision": pre,
                    "balanced_accuracy": ba, "f1": f1,
                    # globales del epoch (mismo para todos los t, pero útiles)
                    "auc_pr": self._get(logs, f"{prefix}auc_pr"),
                    "auc_roc": self._get(logs, f"{prefix}auc_roc"),
                    "brier": self._get(logs, f"{prefix}brier_score"),
                    "loss": self._get(logs, f"{prefix}loss"),
                })
                if split == "val" and math.isfinite(f1) and f1 > best_val["f1"]:
                    best_val = {"f1": f1, "t": t}

        # escribe CSV (append)
        write_header = not self._header_written and not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header: w.writeheader(); self._header_written = True
            w.writerows(rows)

        # print resumen del mejor F1 en val
        if best_val["t"] is not None:
            print(f"\n🔎 Epoch {epoch+1}: mejor F1(val)={best_val['f1']:.4f} @ t={best_val['t']:.2f}  → {self.path}")

def _sanitize_monitor_for_filename(monitor: str) -> str:
    s = monitor.lower().replace("val_", "")
    s = re.sub(r"[^a-z0-9]+", "_", s); s = re.sub(r"_+", "_", s).strip("_")
    return s

def make_callbacks(run_dir: Path, primary_monitor="val_auc_pr", secondary_monitors=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    cbs = [
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
        SafeTimeCSVLogger(str(run_dir / "training_log_with_times.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor=primary_monitor, mode="max", patience=15, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tb"), write_graph=False, write_images=False),
        SafeTimeLimitCallback(TIME_LIMIT_H * 3600, grace_seconds=300),
        SafeBackupAndRestore(run_dir / "backups"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_by_primary.weights.h5"),
            monitor=primary_monitor, mode="max",
            save_best_only=True, save_weights_only=True
        ),
    ]
    if secondary_monitors:
        for m in secondary_monitors:
            cbs.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / f"best_by_{_sanitize_monitor_for_filename(m)}.weights.h5"),
                monitor=m, mode="max", save_best_only=True, save_weights_only=True
            ))
    print(f"✅ Callbacks: {len(cbs)} | primary: {primary_monitor} | secondary: {secondary_monitors or []}")
    return cbs

# === Callback de ETA ligero (imprime cada N s una sola línea por época) ======
class LightETA(tf.keras.callbacks.Callback):
    def __init__(self, train_steps, print_every_sec=20, track=("loss", "accuracy")):
        super().__init__()
        self.train_steps = int(train_steps) if train_steps else None
        self.print_every = float(print_every_sec)
        self.track = tuple(track)
        self._last = 0.0; self._epoch_start = 0.0; self._batch = 0
        self._ema = {k: None for k in self.track}

    def _get_lr(self):
        try:
            it = int(self.model.optimizer.iterations.numpy())
            lr = self.model.optimizer.learning_rate
            if callable(getattr(lr, "__call__", None)):
                return float(lr(it).numpy())
            return float(tf.keras.backend.get_value(lr))
        except Exception:
            try: return float(tf.keras.backend.get_value(self.model.optimizer.lr))
            except Exception: return float("nan")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time(); self._batch = 0
        for k in self._ema: self._ema[k] = None
        print(f"\nEpoch {epoch+1}/{self.params.get('epochs','?')} ...", flush=True)
        self._last = self._epoch_start

    def on_train_batch_end(self, batch, logs=None):
        self._batch += 1
        now = time.time()
        if logs:
            for k in self.track:
                v = logs.get(k)
                if v is not None:
                    prev = self._ema[k]
                    self._ema[k] = v if prev is None else (0.9*prev + 0.1*v)
        if now - self._last >= self.print_every and self.train_steps:
            elapsed = now - self._epoch_start
            done = min(self._batch, self.train_steps)
            speed = done / elapsed if elapsed > 0 else 0.0
            remain = (self.train_steps - done) / speed if speed > 0 else float("inf")
            lr = self._get_lr()
            mets = " ".join(f"{k}={self._ema[k]:.4f}" for k in self.track if self._ema[k] is not None)
            print(
                f"\r  {done:>5}/{self.train_steps:<5} "
                f"({done/self.train_steps:5.1%})  "
                f"ETA {remain/60:5.1f}m  "
                f"spd {speed:4.1f} it/s  "
                f"lr {lr:.2e}  {mets}",
                end="", flush=True
            )
            self._last = now

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start
        mets = " ".join(f"{k}={logs.get(k):.4f}" for k in self.track if logs and k in logs)
        print(f"\r  done in {elapsed:5.1f}s  {mets}".ljust(100), flush=True)

# ========================= DATASETS ==========================================
def make_datasets(tfrecord_dir: str, n_channels: int, n_timepoints: int):
    train_glob = os.path.join(tfrecord_dir, 'train', '*.tfrecord')
    val_glob   = os.path.join(tfrecord_dir, 'val',   '*.tfrecord')

    def _resolve(pat):
        if '*' in pat:
            files = sorted(glob.glob(pat)); return files or []
        return [pat] if os.path.exists(pat) else []

    train_files = _resolve(train_glob) or _resolve(os.path.join(tfrecord_dir, 'train.tfrecord'))
    val_files   = _resolve(val_glob)   or _resolve(os.path.join(tfrecord_dir, 'val.tfrecord'))

    if not train_files:
        raise FileNotFoundError(f"No se encontraron TFRecords de entrenamiento en {tfrecord_dir}")

    train_ds = create_dataset_final(
        train_files, n_channels=n_channels, n_timepoints=n_timepoints,
        batch_size=BATCH_SIZE, time_step=TIME_STEP, one_hot=ONEHOT,
        shuffle=True, balance_pos_frac=BALANCE_POS_FRAC
    )
    val_ds = create_dataset_final(
        val_files or train_files, n_channels=n_channels, n_timepoints=n_timepoints,
        batch_size=BATCH_SIZE, time_step=TIME_STEP, one_hot=ONEHOT,
        shuffle=False, balance_pos_frac=None
    )
    return train_ds, val_ds

def inspect_dataset(ds, batches=None, pos_label=1):
    tot, pos = 0, 0
    it = ds if batches is None else ds.take(batches)
    for _, y in it:
        y = y.numpy()
        if y.ndim > 1 and y.shape[-1] > 1:  # one-hot
            y = np.argmax(y, axis=-1)
        tot += y.size
        pos += int((y == pos_label).sum())
    prev = (pos / tot) if tot else 0.0
    print(f"Dataset windows: {tot} | positivos: {pos} ({prev:.2%})")

# ========================= PIPELINE PRINCIPAL ================================
def pipeline(data_root: str, n_channels=22):
    # TFRecords
    if WRITE_TFREC:
        print("🎯 PIPELINE DEFINITIVO CORREGIDO:", TFRECORD_DIR)
        write_tfrecord_splits_FINAL_CORRECTED(
            data_root, TFRECORD_DIR, montage='ar',
            resample_fs=FS_TARGET, limits=LIMITS,
            window_sec=WINDOW_SEC, hop_sec=FRAME_HOP_SEC
        )

    n_timepoints = int(WINDOW_SEC * FS_TARGET)
    train_ds, val_ds = make_datasets(TFRECORD_DIR, n_channels, n_timepoints)

    # Sanidad rápida
    inspect_dataset(train_ds, batches=10)
    inspect_dataset(val_ds,   batches=10)

    # Dataset infinito + pasos explícitos (robusto)
    # Creamos datasets "contables" sin shuffle para cardinalidad
    count_train_ds = create_dataset_final(
        glob.glob(os.path.join(TFRECORD_DIR, 'train', '*.tfrecord')) or [os.path.join(TFRECORD_DIR, 'train.tfrecord')],
        n_channels=n_channels, n_timepoints=n_timepoints,
        batch_size=BATCH_SIZE, time_step=TIME_STEP, one_hot=ONEHOT,
        shuffle=False, balance_pos_frac=None
    )
    count_val_ds = create_dataset_final(
        glob.glob(os.path.join(TFRECORD_DIR, 'val', '*.tfrecord')) or [os.path.join(TFRECORD_DIR, 'val.tfrecord')],
        n_channels=n_channels, n_timepoints=n_timepoints,
        batch_size=BATCH_SIZE, time_step=TIME_STEP, one_hot=ONEHOT,
        shuffle=False, balance_pos_frac=None
    )
    steps_per_epoch  = cardinality_safe(count_train_ds)
    val_steps        = cardinality_safe(count_val_ds)

    train_ds = train_ds.repeat()
    val_ds   = val_ds.repeat()

    # LR schedule
    total_steps = max(1, EPOCHS * steps_per_epoch)
    lr_sched = WarmupCosineSchedule(
        base_lr=LEARNING_RATE, total_steps=total_steps,
        warmup_ratio=WARMUP_RATIO, min_lr_frac=MIN_LR_FRAC
    )

    # Optimizer AdamW
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=WEIGHT_DECAY, use_ema=False)
    except TypeError:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=WEIGHT_DECAY)

    # Modelo
    num_classes = 2 if ONEHOT else 1
    model = build_tcn(
        input_shape=(n_timepoints, n_channels),
        num_classes=num_classes,
        one_hot=ONEHOT, time_step=TIME_STEP,
        separable=True, use_squeeze_excitation=True, use_gelu=True
    )

    # Métricas y pérdida
    metrics = make_metrics(ONEHOT)
    loss_fn = make_loss(ONEHOT)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
        jit_compile=True
    )

    # Monitores secundarios automáticos si no se definieron explícitamente
    sec_mons = list(SECONDARY_MONITORS)
    if not sec_mons:
        for s in SENS_AT_SPEC_TARGETS:
            sec_mons.append(f"val_{_mn_sens_at_spec(s)}")
        for s in SPEC_AT_SENS_TARGETS:
            sec_mons.append(f"val_{_mn_spec_at_sens(s)}")
        for t in THRESHOLDS[:1]:
            sec_mons.extend([
                f"val_{_mn_recall_t(t)}",
                f"val_{_mn_spec_t(t)}",
                f"val_{_mn_tp_t(t)}",
                f"val_{_mn_tn_t(t)}",
                f"val_{_mn_fp_t(t)}",
                f"val_{_mn_fn_t(t)}",
            ])

    callbacks = make_callbacks(
        RUN_DIR,
        primary_monitor=PRIMARY_MONITOR if PRIMARY_MONITOR.startswith("val_") else f"val_{PRIMARY_MONITOR}",
        secondary_monitors=sec_mons
    )

    # ETA ligero (imprime cada 20 s una línea dentro de la época)
    eta_cb = LightETA(train_steps=steps_per_epoch, print_every_sec=20, track=("loss","accuracy"))

    print(f"🚀 Entrenando — modo: {'ONEHOT' if ONEHOT else 'BINARIO'} | TIME_STEP={TIME_STEP}")
    report_cb = ThresholdReportCSV(RUN_DIR, thresholds=THRESHOLDS, filename="reports/threshold_report.csv")
    hist = model.fit(
        train_ds, validation_data=val_ds,
        steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks + [eta_cb, report_cb],
        verbose=2
    )

    # Guardados finales (portables)
    final_base = RUN_DIR / "final_model"
    model.save(str(final_base) + ".keras", include_optimizer=False)  # arquitectura + pesos
    model.save_weights(str(final_base) + ".weights.h5")              # solo pesos
    print(f"✅ Guardado: {final_base}.keras  y  {final_base}.weights.h5")
    return model, hist

# ================================ MAIN =======================================
if __name__ == "__main__":
    data_root = 'DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3'
    pipeline(data_root=data_root, n_channels=22)
