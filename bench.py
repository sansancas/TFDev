from __future__ import annotations

"""
TensorFlow Bench/Eval Script

What it does:
- Loads a saved Keras model from a configurable path (edit MODEL_PATH or RUN_DIR below)
- Builds an eval dataset from TFRecords in the 'eval' split (following pipe_og/dataset logic)
- Runs predictions and computes consolidated metrics, including False Alarms per Hour (FA/h)
- Saves predictions.csv, eval_metrics.csv, and plots (ROC, PR, Confusion Matrix) into the run dir

Notes:
- No CLI args needed; edit the constants below.
- If RUN_DIR is set, we try to discover model + config there (preferring *.keras files).
- If MODEL_PATH is set directly to a .keras or SavedModel dir, we use that.
- We mirror naming and style from the PyTorch bench where reasonable.
"""

import os
import json
import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	average_precision_score,
	confusion_matrix,
)

from dataset import create_dataset_final_v2

# ===================== Custom Model Loading Functions =====================

def load_model_with_custom_objects(model_path: str):
	"""Load model providing custom objects for old models without decorators."""
	try:
		# Import custom classes - try both old and new locations
		custom_objects = {}
		
		# Try importing from models (new location with decorators)
		try:
			from models.Hybrid import FiLM1D as FiLM1D_Hybrid, AttentionPooling1D as AP1D_Hybrid
			from models.TCN import FiLM1D as FiLM1D_TCN
			from models.Transformer import (
				FiLM1D as FiLM1D_Trans, 
				MultiHeadSelfAttentionRoPE, 
				AttentionPooling1D as AP1D_Trans, 
				AddCLSToken
			)
			custom_objects.update({
				'FiLM1D': FiLM1D_Hybrid,
				'AttentionPooling1D': AP1D_Hybrid,
				'MultiHeadSelfAttentionRoPE': MultiHeadSelfAttentionRoPE,
				'AddCLSToken': AddCLSToken,
			})
		except ImportError:
			pass
		
		# Try importing from models (old location)
		try:
			from models.Hybrid import FiLM1D as FiLM1D_Old_H, AttentionPooling1D as AP1D_Old_H
			from models.TCN import FiLM1D as FiLM1D_Old_T
			if 'FiLM1D' not in custom_objects:
				custom_objects['FiLM1D'] = FiLM1D_Old_H
			if 'AttentionPooling1D' not in custom_objects:
				custom_objects['AttentionPooling1D'] = AP1D_Old_H
		except ImportError:
			pass
		
		print(f"Loading with custom_objects: {list(custom_objects.keys())}")
		return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
		
	except Exception as e:
		print(f"Custom objects loading failed: {e}")
		raise


def load_weights_only_fallback(model_path: str, cfg: dict, n_channels: int, n_timepoints: int):
	"""Fallback: reconstruct model architecture and load weights only."""
	try:
		# Convert to Path object for easier handling
		model_path = Path(model_path)
		
		# Try to reconstruct model from config
		model_type = cfg.get('MODEL', 'TCN')
		
		if model_type == 'TCN':
			from models.TCN import build_tcn
			model = build_tcn(
				input_shape=(n_timepoints, n_channels),
				num_classes=2 if cfg.get('ONEHOT', False) else 1,
				kernel_size=cfg.get('TCN_KERNEL_SIZE', 7),
				num_blocks=cfg.get('TCN_BLOCKS', 7),
				time_step_classification=cfg.get('TIME_STEP', False),
				one_hot=cfg.get('ONEHOT', False),
				hpc=cfg.get('HPC', False),
				separable=True,
				feat_input_dim=len(cfg.get('FEATURE_NAMES', [])) if cfg.get('FEATURES_AS_VECTOR', False) else None,
			)
		elif model_type == 'HYB':
			from models.Hybrid import build_hybrid
			model = build_hybrid(
				input_shape=(n_timepoints, n_channels),
				num_classes=2 if cfg.get('ONEHOT', False) else 1,
				one_hot=cfg.get('ONEHOT', False),
				time_step=cfg.get('TIME_STEP', False),
				feat_input_dim=len(cfg.get('FEATURE_NAMES', [])) if cfg.get('FEATURES_AS_VECTOR', False) else None,
			)
		elif model_type == 'TRANS':
			from models.Transformer import build_transformer
			model = build_transformer(
				input_shape=(n_timepoints, n_channels),
				num_classes=2 if cfg.get('ONEHOT', False) else 1,
				time_step_classification=cfg.get('TIME_STEP', False),
				one_hot=cfg.get('ONEHOT', False),
				hpc=cfg.get('HPC', False),
			)
		else:
			raise ValueError(f"Unknown model type: {model_type}")
		
		# Try to find and load weights
		weights_loaded = False
		
		# Strategy 3a: Look for corresponding .h5 weights in weights/ subdirectory
		if model_path.suffix == '.keras':
			# Extract epoch from filename if possible (e.g., pareto_f1_ep007.keras -> ep007)
			import re
			epoch_match = re.search(r'ep(\d+)', model_path.name)
			if epoch_match:
				epoch_num = epoch_match.group(1)
				weights_dir = model_path.parent / "weights"
				weights_file = weights_dir / f"ep{epoch_num}.weights.h5"
				if weights_file.exists():
					print(f"Found corresponding weights file: {weights_file}")
					model.load_weights(str(weights_file))
					weights_loaded = True
				else:
					print(f"No weights file found at: {weights_file}")
			
			# Strategy 3b: Look for any .h5 files in weights/ directory
			if not weights_loaded:
				weights_dir = model_path.parent / "weights"
				if weights_dir.exists():
					h5_files = sorted(weights_dir.glob("*.h5"))
					if h5_files:
						# Use the most recent weights file
						latest_weights = max(h5_files, key=lambda p: p.stat().st_mtime)
						print(f"Using latest weights file: {latest_weights}")
						model.load_weights(str(latest_weights))
						weights_loaded = True
		
		# Strategy 3c: Direct .h5 file
		elif model_path.suffix == '.h5':
			model.load_weights(str(model_path))
			weights_loaded = True
		
		# Strategy 3d: Weights directory structure
		else:
			weights_path = model_path / "variables" / "variables"
			if weights_path.exists():
				model.load_weights(str(weights_path))
				weights_loaded = True
		
		if not weights_loaded:
			raise FileNotFoundError(f"No compatible weights found for {model_path}")
		
		print("Successfully reconstructed model and loaded weights")
		return model
		
	except Exception as e:
		print(f"Weights-only loading failed: {e}")
		raise


# ===================== User-configurable section =====================
# Option A: Point to a run directory under ./runs (we'll auto-discover model & config)
RUN_DIR: Optional[str] = None  # e.g., "./runs/eeg_seizures_20250911-151631"

# Option B: Point directly to a Keras model file or SavedModel directory
MODEL_PATH: Optional[str] = 'runs/eeg_seizures_20250923-081338/pareto_auc_ep014.keras'  # e.g., "./runs/eeg_seizures_xxx/best.keras"

# If RUN_DIR is None, we'll pick the most recent run under RUNS_DIR
RUNS_DIR = Path("./runs")

# Where to save evaluation artifacts within the run directory
EVAL_SUBDIR_NAME = "eval_tf"

# Defaults if config is missing
DEFAULTS = {
	"FRAME_HOP_SEC": 2.5,
	"WINDOW_SEC": 5.0,
	"RESAMPLE_FS": 256,
	"ONEHOT": False,
	"TIME_STEP": False,
	"WINDOW_MODE": "features",  # "default" | "soft" | "features"
	"FEATURE_NAMES": None,       # if None, dataset will use its defaults
	"FEATURES_AS_VECTOR": True,
	"BATCH_SIZE": 64,
}

# Threshold configuration (edit here)
# If PREDICTION_THRESHOLD is None, we'll sweep thresholds and pick the best by THRESHOLD_SEARCH_METRIC
PREDICTION_THRESHOLD: Optional[float] = None   # e.g., 0.4 to fix, or None to auto-search
THRESHOLD_SEARCH_METRIC: str = "youden"           # one of: "f1", "balanced_accuracy", "youden", "precision", "recall"
THRESHOLD_SWEEP_STEPS: int = 501              # number of thresholds to evaluate in [0,1]


# ===================== Utilities =====================

def find_latest_run(runs_dir: Path) -> Optional[Path]:
	if not runs_dir.exists():
		return None
	runs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("eeg_seizures_")]
	return max(runs, default=None, key=lambda p: p.stat().st_mtime)


def load_run_config(run_dir: Path) -> dict:
	cfg_path = run_dir / "run_config.json"
	if cfg_path.exists():
		try:
			with open(cfg_path, "r") as f:
				return json.load(f)
		except Exception:
			pass
	# Fallback minimal config
	return {
		"FRAME_HOP_SEC": DEFAULTS["FRAME_HOP_SEC"],
		"WINDOW_SEC": DEFAULTS["WINDOW_SEC"],
		"PREPROCESS": {"resample": DEFAULTS["RESAMPLE_FS"]},
		"ONEHOT": DEFAULTS["ONEHOT"],
		"TIME_STEP": DEFAULTS["TIME_STEP"],
		"WINDOW_MODE": DEFAULTS["WINDOW_MODE"],
		"FEATURE_NAMES": DEFAULTS["FEATURE_NAMES"],
		"FEATURES_AS_VECTOR": DEFAULTS["FEATURES_AS_VECTOR"],
		"BATCH_SIZE": DEFAULTS["BATCH_SIZE"],
		# If not present, default TFRecord base dir
		"TFRECORD_DIR": str(Path("./bin_records_cut2").resolve()),
	}


def discover_model_path(run_dir: Path) -> Optional[Path]:
	"""Try to find a saved Keras model inside the run directory.
	Priority: files with '.keras' (prefer names containing 'best'), else a 'saved_model' dir.
	"""
	# First .keras files
	keras_files = sorted(run_dir.glob("*.keras")) + sorted((run_dir / "artifacts").glob("*.keras")) if (run_dir / "artifacts").exists() else sorted(run_dir.glob("*.keras"))
	if keras_files:
		# prefer those with 'best' or 'final' in name
		preferred = [p for p in keras_files if "best" in p.name.lower()] or [p for p in keras_files if "final" in p.name.lower()]
		return Path(preferred[0] if preferred else keras_files[0])

	# Then SavedModel directory
	for cand in [run_dir / "saved_model", run_dir / "model", run_dir / "exported_model"]:
		if cand.exists() and cand.is_dir():
			# SavedModel should contain saved_model.pb
			if (cand / "saved_model.pb").exists():
				return cand
	return None


def infer_shape_from_tfrecord(tfrecord_files: list[str], cfg: dict) -> Tuple[int, int]:
	"""Infer (n_channels, n_timepoints) from the first record if metadata is present.
	Fallback to (22, int(WINDOW_SEC * RESAMPLE_FS)).
	"""
	if not tfrecord_files:
		# fallback
		n_ch = 22
		n_tp = int(float(cfg.get("WINDOW_SEC", DEFAULTS["WINDOW_SEC"])) * float(cfg.get("PREPROCESS", {}).get("resample", DEFAULTS["RESAMPLE_FS"])) )
		return n_ch, n_tp

	feat = {
		"n_channels": tf.io.FixedLenFeature([], tf.int64, default_value=0),
		"n_timepoints": tf.io.FixedLenFeature([], tf.int64, default_value=0),
	}
	try:
		ds = tf.data.TFRecordDataset(tfrecord_files[:1])
		for raw in ds.take(1):
			ex = tf.io.parse_single_example(raw, feat)
			n_ch = int(ex["n_channels"].numpy())
			n_tp = int(ex["n_timepoints"].numpy())
			if n_ch > 0 and n_tp > 0:
				return n_ch, n_tp
	except Exception:
		pass

	# fallback
	n_ch = 22
	n_tp = int(float(cfg.get("WINDOW_SEC", DEFAULTS["WINDOW_SEC"])) * float(cfg.get("PREPROCESS", {}).get("resample", DEFAULTS["RESAMPLE_FS"])) )
	return n_ch, n_tp


def build_eval_dataset(cfg: dict, tfrecord_dir: Path, n_channels: int, n_timepoints: int, batch_size: int) -> tf.data.Dataset:
	eval_dir = tfrecord_dir / "eval"
	if not eval_dir.exists():
		# fallback to 'test' if 'eval' doesn't exist
		eval_dir = tfrecord_dir / "test"
	files = sorted(glob.glob(str(eval_dir / "*.tfrecord")))
	if not files:
		raise FileNotFoundError(f"No TFRecord files found in {eval_dir}")

	ds = create_dataset_final_v2(
		files,
		n_channels=n_channels,
		n_timepoints=n_timepoints,
		batch_size=batch_size,
		one_hot=bool(cfg.get("ONEHOT", DEFAULTS["ONEHOT"])),
		time_step=bool(cfg.get("TIME_STEP", DEFAULTS["TIME_STEP"])),
		cache=False,
		drop_remainder=False,
		shuffle=False,
		window_mode=str(cfg.get("WINDOW_MODE", DEFAULTS["WINDOW_MODE"])) ,
		# Keep feature vector identical to training: do NOT append label-derived features unless explicitly requested
		include_label_feats=bool(cfg.get("INCLUDE_LABEL_FEATS", False)),
		sample_rate=int(cfg.get("PREPROCESS", {}).get("resample", DEFAULTS["RESAMPLE_FS"])) ,
		feature_names=cfg.get("FEATURE_NAMES", DEFAULTS["FEATURE_NAMES"]) ,
		return_feature_vector=bool(cfg.get("FEATURES_AS_VECTOR", DEFAULTS["FEATURES_AS_VECTOR"])) ,
		prefetch=True,
	)
	return ds


def ensure_dir(p: Path) -> Path:
	p.mkdir(parents=True, exist_ok=True)
	return p


def save_png_svg(fig, out_dir: Path, name: str):
	out_dir = ensure_dir(out_dir)
	png = out_dir / f"{name}.png"
	svg = out_dir / f"{name}.svg"
	fig.tight_layout()
	fig.savefig(png, bbox_inches="tight", dpi=300)
	try:
		fig.savefig(svg, bbox_inches="tight")
	except Exception:
		pass
	plt.close(fig)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, frame_hop_sec: float, threshold: float = 0.5) -> dict:
	# Convert shapes
	y_true = y_true.astype(int).reshape(-1)
	# Accept prob either (N,) or (N,1)
	y_prob = y_prob.reshape(-1)
	y_pred = (y_prob >= float(threshold)).astype(int)

	tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
	total = tp + tn + fp + fn

	# Base metrics
	acc = accuracy_score(y_true, y_pred) if total > 0 else 0.0
	bacc = balanced_accuracy_score(y_true, y_pred) if total > 0 else 0.0
	prec = precision_score(y_true, y_pred, zero_division=0)
	rec = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
	spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
	f1 = f1_score(y_true, y_pred, zero_division=0)

	# Prob-based
	try:
		auroc = roc_auc_score(y_true, y_prob)
	except Exception:
		auroc = np.nan
	try:
		auprc = average_precision_score(y_true, y_prob)
	except Exception:
		auprc = np.nan

	# False alarms per hour (window-level approximation)
	hours = (len(y_true) * float(frame_hop_sec)) / 3600.0
	fa_per_hour = (fp / hours) if hours > 0 else np.nan

	return {
		"threshold": float(threshold),
		"accuracy": acc,
		"balanced_accuracy": bacc,
		"precision": prec,
		"recall": rec,
		"specificity": spec,
		"f1": f1,
		"auroc": auroc,
		"auprc": auprc,
		"false_alarms_per_hour": fa_per_hour,
		"tp": int(tp),
		"tn": int(tn),
		"fp": int(fp),
		"fn": int(fn),
		"total_windows": int(total),
		"window_hop_sec": float(frame_hop_sec),
	}


def _metric_value(row: dict, name: str) -> float:
	# helper to extract metric value for threshold selection
	name = name.lower()
	if name == "youden":
		return float(row.get("recall", 0.0)) + float(row.get("specificity", 0.0)) - 1.0
	return float(row.get(name, float("nan")))


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, frame_hop_sec: float,
					 metric: str = "f1", steps: int = 501) -> tuple[float, pd.DataFrame]:
	"""Evaluate metrics for thresholds in [0,1], return best threshold and full DataFrame."""
	thrs = np.linspace(0.0, 1.0, max(2, int(steps)))
	rows = []
	for thr in thrs:
		rows.append(compute_metrics(y_true, y_prob, frame_hop_sec, threshold=float(thr)))
	df = pd.DataFrame(rows)
	# choose best by requested metric (maximize)
	vals = df.apply(lambda r: _metric_value(r, metric), axis=1).to_numpy()
	# handle NaNs by treating as -inf
	vals = np.where(np.isnan(vals), -np.inf, vals)
	best_idx = int(np.argmax(vals))
	best_thr = float(df.loc[best_idx, "threshold"]) if len(df) else 0.5
	return best_thr, df


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, name: str = "confusion_matrix"):
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
	total = cm.sum()
	cm_norm = cm / total if total > 0 else cm

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
	sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
				xticklabels=["No Seizure", "Seizure"],
				yticklabels=["No Seizure", "Seizure"], ax=ax1, cbar_kws={"label": "Proportion"})
	ax1.set_title("Normalized Confusion Matrix")
	ax1.set_xlabel("Predicted")
	ax1.set_ylabel("True")

	# Add absolute counts text overlay
	for i in range(2):
		for j in range(2):
			ax1.text(j + 0.5, i + 0.85, f"({int(cm[i, j])})", ha="center", va="center", color="black")

	tn, fp, fn, tp = cm.ravel().astype(int)
	acc = (tp + tn) / total if total > 0 else 0
	prec = tp / (tp + fp) if (tp + fp) > 0 else 0
	rec = tp / (tp + fn) if (tp + fn) > 0 else 0
	spec = tn / (tn + fp) if (tn + fp) > 0 else 0
	f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

	text = (
		f"Accuracy: {acc:.3f}\n"
		f"Precision: {prec:.3f}\n"
		f"Recall (Sensitivity): {rec:.3f}\n"
		f"Specificity: {spec:.3f}\n"
		f"F1: {f1:.3f}\n\n"
		f"TP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}"
	)
	ax2.axis("off")
	ax2.text(0.01, 0.99, text, va="top", ha="left", fontsize=12, family="monospace")

	save_png_svg(fig, out_dir, name)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path, threshold: float | None = None):
	# ROC
	try:
		from sklearn.metrics import roc_curve, precision_recall_curve, auc
		fpr, tpr, _ = roc_curve(y_true, y_prob)
		roc_auc = auc(fpr, tpr)
		fig = plt.figure(figsize=(7, 6))
		plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
		plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.title("ROC Curve")
		# Mark operating point at chosen threshold
		if threshold is not None:
			# compute single-point FPR/TPR at threshold
			y_pred_thr = (y_prob >= float(threshold)).astype(int)
			from sklearn.metrics import confusion_matrix
			tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thr, labels=[0, 1]).ravel()
			fpr_thr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
			tpr_thr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
			plt.scatter([fpr_thr], [tpr_thr], color="red", zorder=5, label=f"thr={threshold:.3f}")
			plt.axvline(fpr_thr, color="red", linestyle=":", alpha=0.7)
			plt.axhline(tpr_thr, color="red", linestyle=":", alpha=0.7)
		plt.legend(loc="lower right")
		save_png_svg(fig, out_dir, "roc_curve")
	except Exception:
		pass

	# PR
	try:
		precision, recall, _ = precision_recall_curve(y_true, y_prob)
		ap = average_precision_score(y_true, y_prob)
		fig = plt.figure(figsize=(7, 6))
		plt.plot(recall, precision, label=f"AP = {ap:.3f}")
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel("Recall")
		plt.ylabel("Precision")
		plt.title("Precision-Recall Curve")
		# Mark operating point at chosen threshold
		if threshold is not None:
			from sklearn.metrics import precision_score, recall_score
			y_pred_thr = (y_prob >= float(threshold)).astype(int)
			prec_thr = precision_score(y_true, y_pred_thr, zero_division=0)
			rec_thr = recall_score(y_true, y_pred_thr, zero_division=0)
			plt.scatter([rec_thr], [prec_thr], color="red", zorder=5, label=f"thr={threshold:.3f}")
			plt.axvline(rec_thr, color="red", linestyle=":", alpha=0.7)
			plt.axhline(prec_thr, color="red", linestyle=":", alpha=0.7)
		plt.legend(loc="lower left")
		save_png_svg(fig, out_dir, "pr_curve")
	except Exception:
		pass


def collect_predictions(model: tf.keras.Model, ds: tf.data.Dataset, one_hot: bool) -> Tuple[np.ndarray, np.ndarray]:
	"""Return y_true (N,), y_prob (N,).
	ds yields ((x or (x, feats)), y). We only need labels and model predictions.
	"""
	y_true_list = []
	# Model.predict already iterates the dataset; we also capture y_true in parallel.
	# To avoid two passes, we'll iterate manually to collect batch predictions.
	y_prob_list = []

	for batch in ds:
		inputs, y = batch
		# Keras sometimes returns y as dict; ensure tensor
		y = tf.convert_to_tensor(y)
		# model inference
		preds = model.predict(inputs, verbose=0)

		# normalize shapes
		y_np = y.numpy()
		if one_hot and y_np.shape[-1] == 2:
			y_np = y_np[..., 1]
		else:
			y_np = y_np.reshape(-1)

		p_np = np.asarray(preds)
		# if logits with 2 units, convert to prob of class 1 using softmax
		if p_np.ndim == 2 and p_np.shape[-1] == 2:
			# softmax-ized prob of positive class
			exp = np.exp(p_np - np.max(p_np, axis=1, keepdims=True))
			p_np = (exp[:, 1] / np.sum(exp, axis=1)).reshape(-1)
		else:
			p_np = p_np.reshape(-1)

		y_true_list.append(y_np)
		y_prob_list.append(p_np)

	y_true_all = np.concatenate(y_true_list, axis=0) if y_true_list else np.array([])
	y_prob_all = np.concatenate(y_prob_list, axis=0) if y_prob_list else np.array([])
	return y_true_all, y_prob_all


def main():
	# Resolve run dir and model path
	run_dir: Optional[Path] = None
	model_path: Optional[Path]

	if RUN_DIR:
		run_dir = Path(RUN_DIR)
		if not run_dir.exists():
			raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
	else:
		run_dir = find_latest_run(RUNS_DIR)
		if run_dir is None:
			raise FileNotFoundError("No runs found. Set RUN_DIR explicitly.")

	if MODEL_PATH:
		model_path = Path(MODEL_PATH)
	else:
		model_path = discover_model_path(run_dir)
		if model_path is None:
			raise FileNotFoundError(f"Could not find a saved model under {run_dir}")

	print(f"Using run_dir: {run_dir}")
	print(f"Using model:   {model_path}")

	cfg = load_run_config(run_dir)
	tfrecord_dir = Path(cfg.get("TFRECORD_DIR", Path("./bin_records_cut2").resolve()))
	batch_size = int(cfg.get("BATCH_SIZE", DEFAULTS["BATCH_SIZE"]))

	# Collect eval files & infer shape
	eval_dir = tfrecord_dir / "eval"
	if not eval_dir.exists():
		eval_dir = tfrecord_dir / "test"
	eval_files = sorted(glob.glob(str(eval_dir / "*.tfrecord")))
	if not eval_files:
		raise FileNotFoundError(f"No TFRecords in {eval_dir}")

	n_channels, n_timepoints = infer_shape_from_tfrecord(eval_files, cfg)
	print(f"Inferred shape: n_channels={n_channels}, n_timepoints={n_timepoints}")

	# Build dataset
	ds = build_eval_dataset(cfg, tfrecord_dir, n_channels, n_timepoints, batch_size)

	# Load model
	print("Loading model…")
	# Try multiple loading strategies for models with custom classes
	model = None
	loading_strategies = [
		# Strategy 1: Standard loading (works with properly decorated models)
		lambda: tf.keras.models.load_model(str(model_path), compile=False),
		
		# Strategy 2: Load with custom objects (for older models without decorators)
		lambda: load_model_with_custom_objects(str(model_path)),
		
		# Strategy 3: Load weights only (fallback method)
		lambda: load_weights_only_fallback(str(model_path), cfg, n_channels, n_timepoints)
	]
	
	for i, strategy in enumerate(loading_strategies, 1):
		try:
			print(f"Trying loading strategy {i}...")
			model = strategy()
			print(f"✓ Successfully loaded model using strategy {i}")
			break
		except Exception as e:
			print(f"Strategy {i} failed: {e}")
			if i == len(loading_strategies):
				raise RuntimeError("All loading strategies failed") from e
	
	model.summary(print_fn=lambda s: None)

	# Iterate and collect predictions
	print("Running predictions on eval set…")
	y_true, y_prob = collect_predictions(model, ds, one_hot=bool(cfg.get("ONEHOT", DEFAULTS["ONEHOT"])) )
	if y_true.size == 0:
		raise RuntimeError("No data returned from dataset; cannot evaluate.")

	# Threshold selection: fixed or sweep
	if PREDICTION_THRESHOLD is None:
		best_thr, sweep_df = sweep_thresholds(y_true, y_prob, frame_hop_sec=float(cfg.get("FRAME_HOP_SEC", DEFAULTS["FRAME_HOP_SEC"])),
												 metric=THRESHOLD_SEARCH_METRIC,
												 steps=THRESHOLD_SWEEP_STEPS)
		chosen_thr = best_thr
		# save sweep
		out_dir = ensure_dir(run_dir / EVAL_SUBDIR_NAME)
		sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)
	else:
		chosen_thr = float(PREDICTION_THRESHOLD)

	# Predictions with chosen threshold
	y_pred = (y_prob >= chosen_thr).astype(int)

	# Metrics
	frame_hop_sec = float(cfg.get("FRAME_HOP_SEC", DEFAULTS["FRAME_HOP_SEC"]))
	metrics = compute_metrics(y_true, y_prob, frame_hop_sec, threshold=0.2)
	print("Consolidated metrics:")
	print(f"  - chosen_threshold: {chosen_thr:.4f}" + (" (auto)" if PREDICTION_THRESHOLD is None else " (fixed)"))
	if PREDICTION_THRESHOLD is None:
		print(f"  - selection_metric: {THRESHOLD_SEARCH_METRIC}")
	for k, v in metrics.items():
		if isinstance(v, float):
			print(f"  - {k}: {v:.4f}")
		else:
			print(f"  - {k}: {v}")

	# Save outputs
	out_dir = ensure_dir(run_dir / EVAL_SUBDIR_NAME)

	# predictions.csv
	pred_df = pd.DataFrame({
		"y_true": y_true.astype(int),
		"y_prob": y_prob.astype(float),
		"y_pred": y_pred.astype(int),
	})
	pred_df.to_csv(out_dir / "predictions.csv", index=False)

	# eval_metrics.csv (single-row)
	metrics_df = pd.DataFrame([metrics])
	metrics_df.to_csv(out_dir / "eval_metrics.csv", index=False)

	# Plots
	plot_confusion(y_true, y_pred, out_dir)
	plot_roc_pr(y_true, y_prob, out_dir, threshold=chosen_thr)

	print(f"Saved predictions and plots to: {out_dir}")


if __name__ == "__main__":
	main()
