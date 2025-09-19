import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

try:
    import keras_tuner as kt
except Exception as e:
    raise RuntimeError("keras-tuner is required. Install with `pip install keras-tuner`. Error: %s" % e)

from dataset import create_dataset_final_v2, DEFAULT_EEG_FEATURES
from models.TCN import build_tcn


# ======================== Configuration (edit as needed) ========================
TFRECORD_DIR = './bin_records_cut2'  # expects train/ and val/
RUNS_DIR = Path('./runs')

FS = 256
WINDOW_SEC = 5.0
N_CHANNELS = 22
N_TIMEPOINTS = int(WINDOW_SEC * FS)
BATCH_SIZE = 64
EPOCHS = 12
EARLYSTOP_PATIENCE = 4
MAX_STEPS_PER_EPOCH: Optional[int] = 1000  # cap steps for faster trials; None = all

ONEHOT = False
TIME_STEP = False
WINDOW_MODE = 'features'  # 'default' | 'soft' | 'features'
FEATURES_AS_VECTOR = True
FEATURE_NAMES = [
    'rms_eeg', 'line_length', 'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
    'bp_rel_theta', 'bp_rel_alpha', 'bp_rel_beta', 'spectral_entropy', 'sef95',
]

# Mixed precision/Device
MIXED_PRECISION = True
DEVICE = None  # auto-select in main


# ======================== Data helpers ========================
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def discover_tfrecords(root: str) -> Dict[str, List[str]]:
    root_p = Path(root)
    d = {}
    for split in ('train', 'val'):
        files = sorted((root_p / split).glob('*.tfrecord'))
        if not files and (root_p / f'{split}.tfrecord').exists():
            files = [root_p / f'{split}.tfrecord']
        if not files:
            raise FileNotFoundError(f'No TFRecords for {split} under {root}')
        d[split] = list(map(str, files))
    return d


def make_dataset(paths: List[str], *, shuffle: bool, cfg: Dict[str, Any]):
    return create_dataset_final_v2(
        paths,
        n_channels=cfg['n_channels'],
        n_timepoints=cfg['n_timepoints'],
        batch_size=cfg['batch_size'],
        one_hot=cfg['onehot'],
        time_step=cfg['time_step'],
        shuffle=shuffle,
        sample_rate=cfg['fs'],
        window_mode=cfg['window_mode'],
        feature_names=cfg['feature_names'],
        return_feature_vector=(cfg['features_as_vector'] and cfg['window_mode']=='features' and not cfg['time_step'])
    )


def steps_for(ds, max_steps: Optional[int]) -> int:
    card = tf.data.experimental.cardinality(ds)
    steps = int(card.numpy()) if card != tf.data.experimental.UNKNOWN_CARDINALITY else sum(1 for _ in ds)
    return min(steps, max_steps) if max_steps else steps


# ======================== HyperModel ========================
class EEGHyperModel(kt.HyperModel):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        # Choose model family (only TCN available in this repo)
        model_type = hp.Choice('MODEL', values=['TCN'], default='TCN')

        # Core HPs
        lr = hp.Float('LEARNING_RATE', min_value=1e-5, max_value=5e-3, sampling='log', default=2e-4)
        weight_decay = hp.Choice('WEIGHT_DECAY', values=[0.0, 1e-4, 1e-3, 1e-2], default=1e-3)
        use_ema = hp.Boolean('USE_EMA', default=True)
        ema_m = hp.Choice('EMA_MOMENTUM', values=[0.999, 0.995, 0.98], default=0.999)
        focal = hp.Boolean('FOCAL', default=False)
        dropout = hp.Float('DROPOUT_RATE', min_value=0.0, max_value=0.4, step=0.1, default=0.2)
        use_se = hp.Boolean('USE_SE', default=True)

        # TCN specifics
        tcn_k = hp.Choice('TCN_KERNEL_SIZE', values=[5, 7, 9, 11], default=7)
        tcn_b = hp.Int('TCN_BLOCKS', min_value=5, max_value=11, step=2, default=7)

        # Effective inputs
        eff_channels = self.cfg['n_channels'] + (
            len(self.cfg['feature_names'])
            if (self.cfg['window_mode'] == 'features' and not self.cfg['time_step'] and not self.cfg['features_as_vector'])
            else 0
        )
        feat_input_dim = (
            len(self.cfg['feature_names'])
            if (self.cfg['window_mode'] == 'features' and not self.cfg['time_step'] and self.cfg['features_as_vector'])
            else None
        )

        # Build model
        if model_type == 'TCN':
            model = build_tcn(
                input_shape=(self.cfg['n_timepoints'], eff_channels),
                num_classes=(2 if self.cfg['onehot'] else 1),
                kernel_size=tcn_k,
                num_blocks=tcn_b,
                time_step_classification=self.cfg['time_step'],
                one_hot=self.cfg['onehot'],
                hpc=False,
                separable=True,
                feat_input_dim=feat_input_dim,
                dropout_rate=dropout,
                # use_squeeze_excitation=use_se,
            )
        else:
            raise ValueError('Unsupported MODEL choice')

        # Loss & metrics
        if self.cfg['onehot']:
            loss = (
                tf.keras.losses.CategoricalFocalCrossentropy(alpha=[0.25, 0.55], gamma=2)
                if focal
                else tf.keras.losses.CategoricalCrossentropy()
            )
            metrics = [
                tf.keras.metrics.CategoricalAccuracy('accuracy'),
                tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                tf.keras.metrics.AUC(curve='ROC', name='roc_auc'),
            ]
        else:
            loss = (
                tf.keras.losses.BinaryFocalCrossentropy(alpha=0.75, gamma=2)
                if focal
                else tf.keras.losses.BinaryCrossentropy()
            )
            metrics = [
                tf.keras.metrics.BinaryAccuracy('accuracy'),
                tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                tf.keras.metrics.AUC(curve='ROC', name='roc_auc'),
            ]

        # Optimizer (constant LR for comparability across trials)
        opt_kwargs = dict(learning_rate=lr, weight_decay=weight_decay)
        if use_ema:
            opt_kwargs['use_ema'] = True
            opt_kwargs['ema_momentum'] = float(ema_m)
        optimizer = tf.keras.optimizers.AdamW(**opt_kwargs)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=True)
        return model


# ======================== Tuning entrypoint ========================
def main(project_name: Optional[str] = None, max_trials: int = 20, seed: int = 42):
    set_seeds(seed)

    # Precision & strategy
    mixed_precision.set_global_policy('mixed_float16' if MIXED_PRECISION else 'float32')
    # Auto-select device
    global DEVICE
    if DEVICE is None:
        gpus = tf.config.list_physical_devices('GPU')
        DEVICE = '/GPU:0' if gpus else '/CPU:0'
    strategy = tf.distribute.OneDeviceStrategy(DEVICE)

    tf_paths = discover_tfrecords(TFRECORD_DIR)

    # Base cfg
    cfg = {
        'fs': FS,
        'n_channels': N_CHANNELS,
        'n_timepoints': N_TIMEPOINTS,
        'batch_size': BATCH_SIZE,
        'onehot': ONEHOT,
        'time_step': TIME_STEP,
        'window_mode': WINDOW_MODE,
        'feature_names': FEATURE_NAMES,
        'features_as_vector': FEATURES_AS_VECTOR,
    }

    # Build count ds to compute steps per epoch (before strategy scope)
    count_ds = make_dataset(tf_paths['train'], shuffle=False, cfg=cfg)
    steps_per_epoch = steps_for(count_ds, MAX_STEPS_PER_EPOCH)

    # Datasets for training
    train_ds = make_dataset(tf_paths['train'], shuffle=True, cfg=cfg).repeat()
    val_ds = make_dataset(tf_paths['val'], shuffle=False, cfg=cfg)

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    proj = project_name or f'ktune_{stamp}'
    out_dir = RUNS_DIR
    (out_dir).mkdir(parents=True, exist_ok=True)

    with strategy.scope():
        hypermodel = EEGHyperModel(cfg)
        tuner = kt.RandomSearch(
            hypermodel,
            objective=kt.Objective('val_pr_auc', direction='max'),
            max_trials=max_trials,
            seed=seed,
            overwrite=False,
            directory=str(out_dir),
            project_name=proj,
        )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=EARLYSTOP_PATIENCE, restore_best_weights=True)

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[es],
        verbose=1,
    )

    # Save best
    best_models = tuner.get_best_models(num_models=1)
    if best_models:
        best = best_models[0]
        save_dir = Path(out_dir) / proj
        save_dir.mkdir(parents=True, exist_ok=True)
        best.save(save_dir / 'best_model.keras')
        # Save best hp
        best_hp = tuner.get_best_hyperparameters(1)[0]
        with open(save_dir / 'best_hparams.json', 'w') as f:
            json.dump(best_hp.values, f, indent=2)
        print('Best hyperparameters:', best_hp.values)


if __name__ == '__main__':
    main(project_name=None, max_trials=20)
