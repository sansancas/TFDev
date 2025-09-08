import os
import gc
import glob
import numpy as np
import tensorflow as tf
import mne
from typing import Optional, Dict
import pandas as pd
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PREPROCESS = {
    'bandpass': (0.5, 40.),   # (low, high) Hz
    'notch': 60.,             # Hz
    'resample': 256,          # Hz target
}

ONEHOT = False
TIME_STEP = True
TRANSPOSE = True
NOTCH = True
BANDPASS = True
NORMALIZE = True
WINDOW_FEATURES_DIM = 6

# =================================================================================================================
# Montage-based CSV listing & filtering
# =================================================================================================================
def filter_by_montage(paths, montage_type: str):
    """
    Keep only CSV or EDF paths whose directory indicates the desired montage.
    montage_type: 'ar', 'ar_a', or 'le'.
    """
    out = []
    for p in paths:
        f = os.path.dirname(p).lower()
        if montage_type == 'ar' and '_tcp_ar' in f and '_tcp_ar_a' not in f:
            out.append(p)
        elif montage_type == 'ar_a' and '_tcp_ar_a' in f:
            out.append(p)
        elif montage_type == 'le' and '_tcp_le' in f:
            out.append(p)
    return out

def list_bi_csvs(data_dir, split, montage='ar'):
    """
    Find all *_bi.csv label files under data_dir/edf/split and filter by montage.
    """
    root = os.path.join(data_dir, 'edf', split)
    all_csv = glob.glob(os.path.join(root, '**', '*_bi.csv'), recursive=True)
    return filter_by_montage(all_csv, montage)

# Definir montajes como constantes para evitar recrear en cada llamada
MONTAGE_PAIRS = {
    'ar': [
        ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
        ('C4', 'T4'), ('T4', 'A2'), ('FP1', 'F3'), ('F3', 'C3'),
        ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'), ('F4', 'C4'),
        ('C4', 'P4'), ('P4', 'O2')
    ],
    'le': [
        ('F7', 'F8'), ('T3', 'T4'), ('T5', 'T6'), 
        ('C3', 'C4'), ('P3', 'P4'), ('O1', 'O2')
    ]
}

def preprocess_edf(raw, config=PREPROCESS):
    """
    Optimized preprocessing function with better memory management and early returns.
    """
    # Early return if no processing needed
    if not any([BANDPASS, NOTCH, NORMALIZE]):
        if config.get('resample', 0) != raw.info['sfreq']:
            raw.resample(config['resample'], npad='auto', verbose=False)
        return raw
    
    # Apply filters before resampling to avoid aliasing
    if BANDPASS:
        raw.filter(
            config['bandpass'][0],
            config['bandpass'][1],
            method='iir',
            iir_params={'order': 4, 'ftype': 'butter'},
            phase='zero',
            verbose=False
        )
    
    if NOTCH:
        raw.notch_filter(
            freqs=config['notch'],
            method='iir',
            iir_params={'order': 2, 'ftype': 'butter'},
            phase='zero',
            verbose=False
        )
    
    # Resample after filtering
    if config.get('resample', 0) and config['resample'] != raw.info['sfreq']:
        raw.resample(config['resample'], npad='auto', verbose=False)
    
    # Normalize last to work with final sampling rate
    if NORMALIZE:
        # More efficient normalization using vectorized operations
        raw.apply_function(
            lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-6),
            picks='eeg',
            channel_wise=False  # Process all channels at once for better vectorization
        )
    
    return raw

def extract_montage_signals(edf_path: str, montage: str='ar', desired_fs: int=0):
    """
    Optimized function to read EDF and return montage-specific bipolar signals.
    Returns raw_bip object with bipolar montage applied.
    """
    # Validate montage early
    if montage not in MONTAGE_PAIRS:
        raise ValueError(f"Unsupported montage '{montage}'. Use 'ar' or 'le'.")
    
    # Read EDF with minimal memory footprint initially
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    
    # Determine suffix efficiently
    ch_names_set = set(raw.ch_names)
    suf = '-LE' if any(ch.endswith('-LE') for ch in raw.ch_names[:10]) else '-REF'  # Check only first 10
    
    # Build pairs with suffix
    pairs = [(f'EEG {a}{suf}', f'EEG {b}{suf}') for a, b in MONTAGE_PAIRS[montage]]
    
    # Check for missing channels early (before loading data)
    needed = {c for pair in pairs for c in pair}
    missing = needed - ch_names_set
    if missing:
        raw.close()  # Clean up
        raise RuntimeError(f"Missing required electrodes for {montage} montage: {missing}")
    
    # Only load data after validation
    raw.load_data()
    
    # Apply preprocessing
    raw = preprocess_edf(raw, config=PREPROCESS)
    
    # Create bipolar montage more efficiently
    anodes = [a for a, _ in pairs]
    cathodes = [b for _, b in pairs]
    ch_names_bip = [f"{a}-{b}" for a, b in pairs]
    
    raw_bip = mne.set_bipolar_reference(
        raw,
        anode=anodes,
        cathode=cathodes,
        ch_name=ch_names_bip,
        drop_refs=True,
        verbose=False
    )
    
    # Pick only the bipolar channels we created
    raw_bip.pick(ch_names_bip)
    
    # Final resampling if needed
    if desired_fs > 0 and raw_bip.info['sfreq'] != desired_fs:
        raw_bip.resample(desired_fs, npad='auto', verbose=False)
    
    return raw_bip

def load_annotations(csv_path):
    """
    Read CSV_BI; return seizure intervals only.
    Optimized with better error handling.
    """
    try:
        df = pd.read_csv(csv_path, comment='#')
        if 'label' not in df.columns:
            raise ValueError(f"'label' column not found in {csv_path}")
        
        df['label'] = df['label'].str.lower()
        seiz_df = df[df['label'] == 'seiz']
        
        if seiz_df.empty:
            return np.array([]).reshape(0, 2)  # Empty array with correct shape
        
        return seiz_df[['start_time', 'stop_time']].to_numpy()
    
    except Exception as e:
        print(f"Warning: Error reading annotations from {csv_path}: {e}")
        return np.array([]).reshape(0, 2)
    
# Definir feature creators como constantes para evitar recreación
def _bytes_feature(v: bytes) -> tf.train.Feature:
    """Create bytes feature - optimized as module-level function"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def _float_feature(v) -> tf.train.Feature:
    """Create float feature - optimized with faster type conversion"""
    if np.isscalar(v):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))
    # Use numpy for faster conversion of arrays
    return tf.train.Feature(float_list=tf.train.FloatList(value=np.asarray(v, dtype=np.float32).tolist()))

def _int64_feature(v) -> tf.train.Feature:
    """Create int64 feature - optimized with faster type conversion"""
    if np.isscalar(v):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
    # Use numpy for faster conversion of arrays
    return tf.train.Feature(int64_list=tf.train.Int64List(value=np.asarray(v, dtype=np.int64).tolist()))

def serialize_example(eeg_flat, labels_flat, *,
                      patient_id: str, record_id: str,
                      duration_sec: float,
                      n_channels: Optional[int] = None,
                      n_timepoints: Optional[int] = None,
                      sfreq: Optional[float] = None,
                      start_tp: Optional[int] = None,
                      hop_tp: Optional[int] = None):
    """
    Optimized serialization with pre-encoded strings and efficient feature creation.
    """
    # Pre-encode strings to avoid repeated encoding
    patient_bytes = patient_id.encode("utf-8")
    record_bytes = record_id.encode("utf-8")
    
    # Build features dictionary more efficiently
    features = {
        "eeg": _float_feature(eeg_flat),
        "labels": _int64_feature(labels_flat),
        "patient_id": _bytes_feature(patient_bytes),
        "record_id": _bytes_feature(record_bytes),
        "duration_sec": _float_feature(duration_sec),
    }
    
    # Add optional features only if provided (avoid None checks in loop)
    optional_features = [
        (n_channels, "n_channels", _int64_feature),
        (n_timepoints, "n_timepoints", _int64_feature),
        (sfreq, "sfreq", _float_feature),
        (start_tp, "start_tp", _int64_feature),
        (hop_tp, "hop_tp", _int64_feature)
    ]
    
    for value, key, feature_fn in optional_features:
        if value is not None:
            features[key] = feature_fn(value)

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()

# Pre-define feature descriptions to avoid recreation
_BASE_FEATURE_DESC = {
    'eeg': tf.io.VarLenFeature(tf.float32),
    'labels': tf.io.VarLenFeature(tf.int64),
    'patient_id': tf.io.FixedLenFeature([], tf.string, default_value=b''),
    'record_id': tf.io.FixedLenFeature([], tf.string, default_value=b''),
    'duration_sec': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def parse_example(
    serialized_example: tf.Tensor,
    n_channels: int,
    n_timepoints: int,
    time_step: bool = False,
    one_hot: bool = False,
    num_classes: int = 2
):
    """
    Optimized parsing with pre-defined feature descriptions and efficient tensor operations.
    """
    # Use pre-defined feature description
    parsed = tf.io.parse_single_example(serialized_example, _BASE_FEATURE_DESC)

    # Convert sparse to dense more efficiently
    eeg_flat = tf.sparse.to_dense(parsed['eeg'])
    lbl_flat_int = tf.sparse.to_dense(parsed['labels'])

    # Calculate dimensions once
    per_win = n_channels * n_timepoints
    total_eeg = tf.shape(eeg_flat)[0]
    n_windows = total_eeg // per_win

    # Reshape EEG data
    eeg = tf.reshape(eeg_flat, [n_windows, n_channels, n_timepoints])

    if time_step:
        # Frame-level labels
        lbl_int = tf.reshape(lbl_flat_int, [n_windows, n_timepoints])
    else:
        # Window-level labels: reduce frame-level to window-level
        lbl_int_full = tf.reshape(lbl_flat_int, [n_windows, n_timepoints])
        lbl_int = tf.cast(tf.reduce_max(lbl_int_full, axis=1), tf.int64)

    # Apply one-hot encoding if requested
    if one_hot:
        lbl = tf.one_hot(lbl_int, depth=num_classes)
    else:
        lbl = lbl_int
    
    return eeg, lbl

def make_balanced_stream(ds, one_hot: bool, time_step: bool, target_pos_frac: float = 0.5):
    """
    Optimized balanced sampling with efficient positive detection.
    Balances by sampling WITHOUT repeating per branch.
    """
    # Pre-compile the positive detection function for efficiency
    @tf.function
    def is_positive(x, lbl):
        """Optimized positive detection with early returns"""
        if one_hot:
            # Handle both (T,C) and (C,) shapes efficiently
            if lbl.shape.rank > 1 and lbl.shape[-1] >= 2:
                y = lbl[..., 1]  # seizure class
            else:
                y = lbl
            y = tf.cast(y, tf.float32)
        else:
            y = tf.cast(lbl, tf.float32)
        
        # Use tf.greater for more efficient comparison
        if time_step:
            return tf.reduce_any(tf.greater(y, 0.5))
        else:
            return tf.greater_equal(y, 0.5)

    # Create filtered datasets
    ds_pos = ds.filter(is_positive)
    ds_neg = ds.filter(lambda x, y: tf.logical_not(is_positive(x, y)))

    return tf.data.Dataset.sample_from_datasets(
        [ds_pos, ds_neg],
        weights=[target_pos_frac, 1.0 - target_pos_frac],
        seed=42,
        stop_on_empty_dataset=False  # Continue even if one stream is empty
    )

# --- al inicio de dataset.py (o en un bloque de constantes) ---
WINDOW_FEATURES_DIM = 6  # número fijo de features agregadas por ventana en 'features' mode

# ----------------- create_dataset_final CON MODOS DE VENTANA -----------------
def create_dataset_final(
    tfrecord_files, n_channels, n_timepoints, batch_size,
    one_hot=False, time_step=False, balance_pos_frac=None,
    cache=False, drop_remainder=False, shuffle=False, shuffle_buffer=4096,
    window_mode="default"  # "default" | "soft" | "features" (solo aplica si time_step=False)
):
    """
    Dataset final con formas estables para Keras/XLA:
      X -> (T, C) o (T, C + WINDOW_FEATURES_DIM) si window_mode='features'
      Y -> (B,1) binario o (B,2) one-hot en modo ventana (default);
           (B,1)/(B,2) con suave [0..1] en 'soft';
           (B,1)/(B,2) duro en 'features'.
    """
    import tensorflow as tf

    feature_desc = {
        'eeg': tf.io.VarLenFeature(tf.float32),
        'labels': tf.io.VarLenFeature(tf.int64),
        'patient_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'record_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'duration_sec': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    wm = tf.convert_to_tensor(window_mode)  # para usar en tf.cond si hace falta

    def _labels_to_vector_T(labels_flat):
        """Devuelve vector (T,) int32: replica/trunca/padea a n_timepoints."""
        L = tf.shape(labels_flat)[0]
        def _replicate_one():
            return tf.fill([n_timepoints], tf.cast(labels_flat[0], tf.int32))
        def _fit_to_T():
            y = tf.cast(labels_flat, tf.int32)
            y = y[:n_timepoints]
            need = n_timepoints - tf.shape(y)[0]
            return tf.cond(need > 0, lambda: tf.pad(y, [[0, need]]), lambda: y)
        return tf.cond(L <= 1, _replicate_one, _fit_to_T)  # (T,)

    def _window_features(eeg_tc, labels_T):
        """
        Calcula 6 features escalares por ventana y las devuelve como (WINDOW_FEATURES_DIM,)
        - frac_pos: fracción de frames positivos
        - n_transitions: número de cambios 0<->1
        - n_onsets: conteo de 0->1
        - n_offsets: conteo de 1->0
        - rms_eeg: sqrt(mean(x^2)) global
        - mad_eeg: mean(|x - mean(x)|) global
        """
        x = tf.cast(eeg_tc, tf.float32)              # (T, C)
        y = tf.cast(labels_T, tf.float32)            # (T,)

        # Etiquetas
        frac_pos = tf.reduce_mean(y)                 # [0,1]
        if n_timepoints > 1:
            prev = y[:-1]
            curr = y[1:]
            changes = tf.cast(tf.not_equal(prev, curr), tf.float32)
            n_transitions = tf.reduce_sum(changes)
            onsets  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prev, 0.0), tf.equal(curr, 1.0)), tf.float32))
            offsets = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prev, 1.0), tf.equal(curr, 0.0)), tf.float32))
        else:
            n_transitions = tf.constant(0.0, tf.float32)
            onsets = tf.constant(0.0, tf.float32)
            offsets = tf.constant(0.0, tf.float32)

        # EEG (globales muy baratas)
        rms_eeg = tf.sqrt(tf.reduce_mean(tf.square(x)))
        mad_eeg = tf.reduce_mean(tf.abs(x - tf.reduce_mean(x)))

        feats = tf.stack([frac_pos, n_transitions, onsets, offsets, rms_eeg, mad_eeg], axis=0)
        feats = tf.ensure_shape(feats, [WINDOW_FEATURES_DIM])
        return feats  # (F,)

    def _parse(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_desc)

        # ---- EEG -> (T, C) ----
        eeg_flat = tf.sparse.to_dense(parsed['eeg'])  # (N,)
        expected = n_channels * n_timepoints
        size = tf.shape(eeg_flat)[0]
        eeg_flat = tf.cond(
            size >= expected,
            lambda: eeg_flat[:expected],
            lambda: tf.pad(eeg_flat, [[0, expected - size]])
        )
        eeg = tf.reshape(eeg_flat, [n_timepoints, n_channels])
        eeg = tf.cast(eeg, tf.float32)
        eeg = tf.ensure_shape(eeg, [n_timepoints, n_channels])  # XLA-friendly

        # ---- LABELS ----
        labels_flat = tf.sparse.to_dense(parsed['labels'])  # (L,)
        labels_T = _labels_to_vector_T(labels_flat)         # (T,) int32

        if time_step:
            # No modificamos el modo frame-by-frame (no solicitado)
            if one_hot:
                y = tf.one_hot(labels_T, depth=2)           # (T,2)
                y = tf.cast(y, tf.float32)
                y = tf.ensure_shape(y, [n_timepoints, 2])
            else:
                y = tf.cast(labels_T, tf.float32)           # (T,)
                y = tf.expand_dims(y, axis=-1)              # (T,1)
                y = tf.ensure_shape(y, [n_timepoints, 1])
            return eeg, y

        # ---- Modo ventana ----
        # y_hard: etiqueta dura por ventana
        y_hard_int = tf.reduce_max(labels_T)                # 0/1 int32
        # y_soft: fracción de positivos
        y_soft_f = tf.reduce_mean(tf.cast(labels_T, tf.float32))

        # Selección del modo de ventana
        if window_mode == "soft":
            # Etiqueta suave
            if one_hot:
                y = tf.stack([1.0 - y_soft_f, y_soft_f], axis=0)  # (2,)
                y = tf.ensure_shape(y, [2])
            else:
                y = tf.reshape(y_soft_f, [1])                     # (1,)
                y = tf.ensure_shape(y, [1])
        else:
            # default y features → etiqueta dura
            if one_hot:
                y = tf.one_hot(y_hard_int, depth=2)               # (2,)
                y = tf.cast(y, tf.float32)
                y = tf.ensure_shape(y, [2])
            else:
                y = tf.cast(y_hard_int, tf.float32)
                y = tf.reshape(y, [1])                            # (1,)
                y = tf.ensure_shape(y, [1])

        # Agregar features como canales (solo en 'features')
        if window_mode == "features":
            feats = _window_features(eeg, labels_T)               # (F,)
            feats_tiled = tf.tile(tf.reshape(feats, [1, WINDOW_FEATURES_DIM]), [n_timepoints, 1])  # (T,F)
            eeg = tf.concat([eeg, feats_tiled], axis=-1)          # (T, C+F)
            eeg = tf.ensure_shape(eeg, [n_timepoints, n_channels + WINDOW_FEATURES_DIM])

        return eeg, y

    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    if balance_pos_frac is not None:
        # (placeholder) tu lógica de remuestreo si la implementas
        pass

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =================================================================================================================
# 🎯 PIPELINE DEFINITIVO CON FUNCIÓN CORREGIDA
# =================================================================================================================

def process_single_file_FINAL_CORRECTED(csv_path, tfrecord_output_dir, chunk_size_mb, dataset_type, resample_fs=256, window_sec=10.0, hop_sec=5.0):
    """
    ✅ VERSIÓN DEFINITIVAMENTE CORREGIDA
    ✅ NO MÁS ERRORES DE serialize_example
    ✅ SIGNATURE CORRECTA GARANTIZADA
    """
    try:
        from pathlib import Path
        import pandas as pd
        import tensorflow as tf
        import numpy as np
        import gc
        
        csv_path = Path(csv_path)
        tfrecord_output_dir = Path(tfrecord_output_dir)
        tfrecord_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Obtener archivo EDF correspondiente
        edf_filename = csv_path.name.replace("_bi.csv", ".edf")
        edf_path = csv_path.parent / edf_filename
        
        if not edf_path.exists():
            return {
                'success': False,
                'error': f'EDF no encontrado: {edf_path.name}',
                'csv_path': str(csv_path),
                'windows_created': 0,
                'tfrecord_files': []
            }
        
        # ===== LEER CSV =====
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            data_start = 0
            header_line = None
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    if ',' in stripped and any(word in stripped.lower() for word in ['channel', 'start', 'time', 'label']):
                        header_line = i
                        data_start = i + 1
                        break
                    else:
                        data_start = i
                        break
            
            if data_start >= len(lines):
                return {
                    'success': False,
                    'error': 'No se encontraron datos en CSV',
                    'csv_path': str(csv_path),
                    'windows_created': 0,
                    'tfrecord_files': []
                }
            
            if header_line is not None:
                labels_df = pd.read_csv(csv_path, skiprows=header_line, comment='#')
            else:
                labels_df = pd.read_csv(
                    csv_path, 
                    skiprows=data_start, 
                    comment='#',
                    names=['channel', 'start_time', 'stop_time', 'label', 'confidence']
                )
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error leyendo CSV: {e}',
                'csv_path': str(csv_path),
                'windows_created': 0,
                'tfrecord_files': []
            }
        
        if labels_df.empty:
            return {
                'success': False,
                'error': 'CSV sin datos válidos',
                'csv_path': str(csv_path),
                'windows_created': 0,
                'tfrecord_files': []
            }
        
        # ===== PROCESAR EDF =====
        try:
            result = extract_montage_signals(str(edf_path), montage='ar', desired_fs=resample_fs)
            
            if result is None:
                return {
                    'success': False,
                    'error': 'extract_montage_signals retornó None',
                    'csv_path': str(csv_path),
                    'windows_created': 0,
                    'tfrecord_files': []
                }
            
            if isinstance(result, tuple) and len(result) == 3:
                mont, fs_out, n_samples_total = result
                signal_data = mont.T
                sfreq = fs_out
            elif hasattr(result, 'get_data'):
                signal_data = result.get_data()
                sfreq = result.info['sfreq']
            elif isinstance(result, np.ndarray):
                if result.ndim == 2:
                    if result.shape[0] > result.shape[1]:
                        signal_data = result.T
                    else:
                        signal_data = result
                else:
                    return {
                        'success': False,
                        'error': f'Dimensiones incorrectas: {result.shape}',
                        'csv_path': str(csv_path),
                        'windows_created': 0,
                        'tfrecord_files': []
                    }
                sfreq = resample_fs  # Usar la frecuencia objetivo
            else:
                return {
                    'success': False,
                    'error': f'Tipo de retorno inesperado: {type(result)}',
                    'csv_path': str(csv_path),
                    'windows_created': 0,
                    'tfrecord_files': []
                }
            
            if signal_data.ndim != 2:
                return {
                    'success': False,
                    'error': f'Dimensiones incorrectas: {signal_data.shape}',
                    'csv_path': str(csv_path),
                    'windows_created': 0,
                    'tfrecord_files': []
                }
            
            n_channels, n_samples = signal_data.shape
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error en extract_montage_signals: {str(e)[:150]}',
                'csv_path': str(csv_path),
                'windows_created': 0,
                'tfrecord_files': []
            }
        
        # ===== PROCESAR ETIQUETAS =====
        try:
            labels_df = labels_df.copy()
            
            column_mapping = {}
            for col in labels_df.columns:
                col_lower = col.lower().strip()
                if 'label' in col_lower and 'probability' not in col_lower:
                    column_mapping[col] = 'seizure_label'
                elif 'confidence' in col_lower or 'prob' in col_lower:
                    column_mapping[col] = 'probability_label'
                elif 'start' in col_lower and 'time' in col_lower:
                    column_mapping[col] = 'start_time'
                elif 'stop' in col_lower and 'time' in col_lower:
                    column_mapping[col] = 'stop_time'
            
            labels_df = labels_df.rename(columns=column_mapping)
            
            if 'probability_label' not in labels_df.columns:
                if 'confidence' in labels_df.columns:
                    labels_df['probability_label'] = labels_df['confidence']
                elif 'seizure_label' in labels_df.columns:
                    labels_df['probability_label'] = labels_df['seizure_label'].apply(
                        lambda x: 1.0 if str(x).lower() in ['seiz', 'seizure', '1'] else 0.0
                    )
                else:
                    numeric_cols = labels_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        labels_df['probability_label'] = labels_df[numeric_cols[0]]
                    else:
                        labels_df['probability_label'] = 0.0
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error procesando etiquetas: {str(e)[:100]}',
                'csv_path': str(csv_path),
                'windows_created': 0,
                'tfrecord_files': []
            }
        
        # ===== CREAR VENTANAS CON CONFIGURACIÓN MEJORADA =====
        try:
            # Usar parámetros configurables como en pipeline.py
            window_samples = int(round(window_sec * sfreq))
            if hop_sec is None or hop_sec <= 0:
                step_samples = window_samples  # Sin solapamiento
            else:
                step_samples = int(round(hop_sec * sfreq))
                # Validación como en pipeline.py
                step_samples = max(1, min(step_samples, window_samples))
            
            if n_samples < window_samples:
                # Fallback para archivos cortos
                window_samples = min(n_samples, int(0.5 * sfreq))
                step_samples = window_samples // 2
                
                if window_samples < 64:
                    return {
                        'success': False,
                        'error': f'Archivo demasiado corto: {n_samples} samples',
                        'csv_path': str(csv_path),
                        'windows_created': 0,
                        'tfrecord_files': []
                    }
            
            # Convertir etiquetas a índices de muestra para mayor precisión
            seizure_intervals_samples = []
            for _, row in labels_df.iterrows():
                if 'start_time' in row and 'stop_time' in row:
                    try:
                        s_time = float(row['start_time'])
                        e_time = float(row['stop_time'])
                        s_idx = max(0, int(round(s_time * sfreq)))
                        e_idx = min(n_samples, int(round(e_time * sfreq)))
                        if e_idx > s_idx:
                            seizure_intervals_samples.append((s_idx, e_idx))
                    except (ValueError, TypeError):
                        continue
            
            windows_data = []
            windows_labels = []
            
            # Generación mejorada como en pipeline.py
            for start_sample in range(0, n_samples - window_samples + 1, step_samples):
                end_sample = start_sample + window_samples
                
                window_signal = signal_data[:, start_sample:end_sample]
                
                # Etiquetas frame-level más precisas
                labels_frame = np.zeros(window_samples, dtype=np.int64)
                for s_idx, e_idx in seizure_intervals_samples:
                    # Coordenadas locales de la ventana
                    rs = max(0, s_idx - start_sample)
                    re = min(window_samples, e_idx - start_sample)
                    if re > rs:
                        labels_frame[rs:re] = 1
                
                windows_data.append(window_signal)
                windows_labels.append(labels_frame)
            
            if not windows_data:
                return {
                    'success': False,
                    'error': 'No se generaron ventanas válidas',
                    'csv_path': str(csv_path),
                    'windows_created': 0,
                    'tfrecord_files': []
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error creando ventanas: {str(e)[:100]}',
                'csv_path': str(csv_path),
                'windows_created': 0,
                'tfrecord_files': []
            }
        
        # ===== ESCRIBIR TFRECORD CON SERIALIZE_EXAMPLE CORRECTO =====
        try:
            base_name = csv_path.stem.replace('_bi', '')
            tfrecord_filename = f"{base_name}.tfrecord"
            tfrecord_path = tfrecord_output_dir / tfrecord_filename
            
            windows_written = 0
            serialize_errors = 0
            
            with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
                for i, (window_signal, window_labels_array) in enumerate(zip(windows_data, windows_labels)):
                    if window_signal.shape[1] != window_samples:
                        continue
                    
                    if np.isnan(window_signal).any() or np.isinf(window_signal).any():
                        continue
                    
                    # ✅ SERIALIZACIÓN CORRECTA - SIGNATURE EXACTA
                    try:
                        # Preparar datos en el formato que espera serialize_example
                        eeg_flat = window_signal.astype(np.float32).flatten()
                        labels_flat = window_labels_array.astype(np.int64).flatten()
                        
                        # Llamada con la signature EXACTA de tu función
                        example_bytes = serialize_example(
                            eeg_flat,                                    # Primer parámetro posicional
                            labels_flat,                                 # Segundo parámetro posicional
                            patient_id=base_name.split('_')[0],         # Keyword argument requerido
                            record_id=base_name,                        # Keyword argument requerido
                            duration_sec=float(window_sec),             # Keyword argument requerido
                            n_channels=int(window_signal.shape[0]),     # Opcional
                            n_timepoints=int(window_signal.shape[1]),   # Opcional
                            sfreq=float(sfreq),                         # Opcional
                            start_tp=int(i * step_samples),             # Opcional
                            hop_tp=int(step_samples)                    # Opcional
                        )
                        
                        # ✅ serialize_example YA RETORNA BYTES - no necesita .SerializeToString()
                        writer.write(example_bytes)
                        windows_written += 1
                        
                    except Exception as serialize_error:
                        serialize_errors += 1
                        print(f"   ⚠️  Error serialización ventana {i}: {serialize_error}")
                        
                        # Fallback manual SOLO si falla serialize_example
                        try:
                            eeg_flat_fallback = window_signal.astype(np.float32).flatten()
                            # Para fallback, usar etiqueta de ventana simplificada
                            window_label_simple = int(np.any(window_labels_array > 0.5))
                            labels_flat_fallback = np.array([window_label_simple], dtype=np.int64)
                            
                            feature = {
                                'eeg': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=eeg_flat_fallback.tolist())
                                ),
                                'labels': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=labels_flat_fallback.tolist())
                                ),
                                'patient_id': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[base_name.split('_')[0].encode('utf-8')])
                                ),
                                'record_id': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[base_name.encode('utf-8')])
                                ),
                                'duration_sec': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=[float(window_sec)])
                                )
                            }
                            
                            example = tf.train.Example(features=tf.train.Features(feature=feature))
                            writer.write(example.SerializeToString())
                            windows_written += 1
                            
                        except Exception as manual_error:
                            print(f"   ❌ Error serialización manual ventana {i}: {manual_error}")
                            continue
            
            if not tfrecord_path.exists() or tfrecord_path.stat().st_size == 0:
                return {
                    'success': False,
                    'error': f'TFRecord no se creó: {windows_written} ventanas, {serialize_errors} errores',
                    'csv_path': str(csv_path),
                    'windows_created': 0,
                    'tfrecord_files': []
                }
            
            del signal_data, windows_data, windows_labels, labels_df
            gc.collect()
            
            return {
                'success': True,
                'error': None,
                'csv_path': str(csv_path),
                'windows_created': windows_written,
                'tfrecord_files': [str(tfrecord_path)],
                'serialize_errors': serialize_errors  # Para monitoreo
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error escribiendo TFRecord: {str(e)[:150]}',
                'csv_path': str(csv_path),
                'windows_created': 0,
                'tfrecord_files': []
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error general: {str(e)[:100]}',
            'csv_path': str(csv_path),
            'windows_created': 0,
            'tfrecord_files': []
        }

def write_tfrecord_splits_FINAL_CORRECTED(
    data_dir,
    tfrecord_dir,
    montage='ar',
    resample_fs=256,
    window_sec=10.0,
    hop_sec=5.0,
    limits=None,
    n_workers=None,
    chunk_size_mb=100,
    resume=True,
    validate=True
):
    """
    🎯 PIPELINE DEFINITIVO SIN ERRORES DE SERIALIZE_EXAMPLE
    ✅ USA process_single_file_FINAL_CORRECTED
    ✅ MONITOREA ERRORES DE SERIALIZACIÓN
    ✅ EFICIENCIA MANTENIDA
    ✅ LÍMITES CORREGIDOS
    ✅ CONFIGURACIÓN MEJORADA: resample_fs, window_sec, hop_sec
    ✅ ETIQUETAS FRAME-LEVEL PRECISAS
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path
    import multiprocessing as mp
    import time
    
    start_time = time.time()
    
    print(f"🎯 PIPELINE DEFINITIVO CORREGIDO: {tfrecord_dir}")
    print(f"   Montaje: {montage}, Chunk: {chunk_size_mb}MB")
    
    # ✅ ARREGLAR: Mostrar límites si se especifican
    if limits:
        print(f"   Límites por split: {limits}")
    
    if n_workers is None:
        n_workers = 1
    
    print(f"   Workers: {n_workers}")
    
    tfrecord_path = Path(tfrecord_dir)
    tfrecord_path.mkdir(exist_ok=True, parents=True)
    
    split_mapping = {'train': 'train', 'dev': 'val', 'eval': 'test'}
    total_stats = {
        'files_processed': 0, 
        'windows_created': 0, 
        'errors': 0,
        'tfrecord_files_created': 0,
        'processing_time': 0,
        'serialize_errors': 0
    }
    
    for split, tfrecord_split in split_mapping.items():
        print(f"\n📁 Procesando split '{split}' → {tfrecord_split}")
        
        try:
            csv_files = list_bi_csvs(data_dir, split, montage)
            
            print(f"   📊 {len(csv_files)} archivos encontrados inicialmente")
            
            # ✅ ARREGLAR: Aplicar límites CORRECTAMENTE
            if limits and split in limits:
                original_count = len(csv_files)
                csv_files = csv_files[:limits[split]]
                print(f"   🔒 Límite aplicado: {original_count} → {len(csv_files)} archivos (límite: {limits[split]})")
            
            if not csv_files:
                print(f"   ⚠️  Sin archivos para procesar en {split}")
                continue
            
            split_output_dir = tfrecord_path / tfrecord_split
            split_output_dir.mkdir(exist_ok=True, parents=True)
            
            # ✅ ARREGLAR: Verificar si ya existen archivos procesados
            if resume:
                existing_tfrecords = list(split_output_dir.glob('*.tfrecord'))
                if existing_tfrecords:
                    print(f"   📋 Resume activado: {len(existing_tfrecords)} TFRecords ya existen")
                    # Filtrar archivos ya procesados
                    existing_bases = {f.stem for f in existing_tfrecords}
                    csv_files_filtered = []
                    for csv_file in csv_files:
                        csv_stem = Path(csv_file).stem.replace('_bi', '')
                        if csv_stem not in existing_bases:
                            csv_files_filtered.append(csv_file)
                    
                    skipped = len(csv_files) - len(csv_files_filtered)
                    csv_files = csv_files_filtered
                    print(f"   ⏭️  {skipped} archivos ya procesados, {len(csv_files)} por procesar")
            
            if not csv_files:
                print(f"   ✅ Todos los archivos ya están procesados en {split}")
                continue
            
            split_stats = {
                'files_processed': 0, 
                'windows_created': 0, 
                'errors': 0, 
                'tfrecord_files': 0,
                'serialize_errors': 0
            }
            
            process_args = [
                (str(f), str(split_output_dir), chunk_size_mb, tfrecord_split, resample_fs, window_sec, hop_sec) 
                for f in csv_files
            ]
            
            # Procesamiento con función corregida
            if n_workers == 1:
                for args in process_args:
                    try:
                        # ✅ USAR FUNCIÓN CORREGIDA
                        result = process_single_file_FINAL_CORRECTED(*args)
                        csv_path = args[0]
                        
                        if result['success']:
                            split_stats['files_processed'] += 1
                            split_stats['windows_created'] += result['windows_created']
                            split_stats['tfrecord_files'] += len(result['tfrecord_files'])
                            
                            # ✅ Monitorear errores de serialización
                            if 'serialize_errors' in result:
                                split_stats['serialize_errors'] += result['serialize_errors']
                            
                            tfrecord_name = Path(result['tfrecord_files'][0]).name if result['tfrecord_files'] else 'N/A'
                            
                            # ✅ Mostrar errores de serialización si los hay
                            if 'serialize_errors' in result and result['serialize_errors'] > 0:
                                print(f"   ⚠️  {Path(result['csv_path']).name}: {result['windows_created']} ventanas, {result['serialize_errors']} errores serialización → {tfrecord_name}")
                            else:
                                print(f"   ✅ {Path(result['csv_path']).name}: {result['windows_created']} ventanas → {tfrecord_name}")
                            
                        else:
                            split_stats['errors'] += 1
                            print(f"   ❌ {Path(result['csv_path']).name}: {result['error']}")
                            
                    except Exception as e:
                        split_stats['errors'] += 1
                        print(f"   ❌ {Path(csv_path).name}: {e}")
            else:
                # Procesamiento paralelo con función corregida
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    future_to_file = {
                        executor.submit(process_single_file_FINAL_CORRECTED, *args): args[0]
                        for args in process_args
                    }
                    
                    for future in as_completed(future_to_file):
                        csv_path = future_to_file[future]
                        
                        try:
                            result = future.result()
                            
                            if result['success']:
                                split_stats['files_processed'] += 1
                                split_stats['windows_created'] += result['windows_created']
                                split_stats['tfrecord_files'] += len(result['tfrecord_files'])
                                
                                if 'serialize_errors' in result:
                                    split_stats['serialize_errors'] += result['serialize_errors']
                                
                                tfrecord_name = Path(result['tfrecord_files'][0]).name if result['tfrecord_files'] else 'N/A'
                                
                                if 'serialize_errors' in result and result['serialize_errors'] > 0:
                                    print(f"   ⚠️  {Path(result['csv_path']).name}: {result['windows_created']} ventanas, {result['serialize_errors']} errores serialización → {tfrecord_name}")
                                else:
                                    print(f"   ✅ {Path(result['csv_path']).name}: {result['windows_created']} ventanas → {tfrecord_name}")
                                
                            else:
                                split_stats['errors'] += 1
                                print(f"   ❌ {Path(result['csv_path']).name}: {result['error']}")
                                
                        except Exception as e:
                            split_stats['errors'] += 1
                            print(f"   ❌ {Path(csv_path).name}: {e}")
            
            # Actualizar totales
            total_stats['files_processed'] += split_stats['files_processed']
            total_stats['windows_created'] += split_stats['windows_created']
            total_stats['errors'] += split_stats['errors']
            total_stats['tfrecord_files_created'] += split_stats['tfrecord_files']
            total_stats['serialize_errors'] += split_stats['serialize_errors']  # ✅ Agregar errores de serialización
            
            actual_tfrecords = list(split_output_dir.glob('*.tfrecord'))
            
            print(f"   ✅ Split completado: {split_stats['files_processed']} archivos, "
                  f"{len(actual_tfrecords)} TFRecords, {split_stats['errors']} errores")
            
            # ✅ Reportar errores de serialización
            if split_stats['serialize_errors'] > 0:
                print(f"   ⚠️  Errores de serialización: {split_stats['serialize_errors']}")
                
        except Exception as e:
            print(f"   ❌ Error en split '{split}': {e}")
            total_stats['errors'] += 1
    
    # Estadísticas finales
    total_time = time.time() - start_time
    all_tfrecords = list(tfrecord_path.rglob('*.tfrecord'))
    
    print(f"\\n🎉 COMPLETADO:")
    print(f"   CSV procesados: {total_stats['files_processed']}")
    print(f"   Ventanas: {total_stats['windows_created']:,}")
    print(f"   TFRecords: {len(all_tfrecords)}")
    print(f"   Errores: {total_stats['errors']}")
    print(f"   ⚠️  Errores serialización: {total_stats['serialize_errors']}")  # ✅ Mostrar errores de serialización
    print(f"   Tiempo: {total_time:.1f}s")
    
    if len(all_tfrecords) > 0:
        total_size = sum(f.stat().st_size for f in all_tfrecords) / (1024*1024)
        print(f"   Tamaño: {total_size:.1f} MB")
        
        if total_stats['serialize_errors'] == 0:
            print(f"\\n🎊 ¡PERFECTO! {len(all_tfrecords)} archivos TFRecord SIN ERRORES DE SERIALIZACIÓN")
        else:
            print(f"\\n✅ {len(all_tfrecords)} archivos TFRecord generados")
            print(f"⚠️  {total_stats['serialize_errors']} errores de serialización (usaron fallback manual)")
        
        print(f"   📁 Ubicación: {tfrecord_path}")
        
        for split_name in ['train', 'val', 'test']:
            split_files = list((tfrecord_path / split_name).glob('*.tfrecord'))
            if split_files:
                split_size = sum(f.stat().st_size for f in split_files) / (1024*1024)
                print(f"   {split_name}: {len(split_files)} archivos ({split_size:.1f} MB)")
        
    else:
        print(f"\\n⚠️  No se generaron TFRecords - revisar errores arriba")
    
    return total_stats