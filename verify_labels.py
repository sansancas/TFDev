import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

# Local helpers from dataset utilities if available
try:
    from dataset import list_bi_csvs  # type: ignore
except Exception:
    list_bi_csvs = None  # Fallback to simple matching if not available


def _read_csv_intervals(csv_path: Path) -> List[Tuple[float, float, float]]:
    """Read TUH *_bi.csv and return list of (start_time, stop_time, prob) for seizure rows.
    Falls back gracefully if headers vary. Prob defaults to 1.0 when missing.
    """
    try:
        # Robust header skip similar to dataset.py
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        data_start = 0
        header_line = None
        for i, line in enumerate(lines):
            s = line.strip()
            if s and not s.startswith('#'):
                if ',' in s and any(w in s.lower() for w in ['channel', 'start', 'time', 'label']):
                    header_line = i
                    data_start = i + 1
                    break
                else:
                    data_start = i
                    break

        if data_start >= len(lines):
            return []

        if header_line is not None:
            df = pd.read_csv(csv_path, skiprows=header_line, comment='#')
        else:
            df = pd.read_csv(
                csv_path,
                skiprows=data_start,
                comment='#',
                names=['channel', 'start_time', 'stop_time', 'label', 'confidence'],
            )

        if df is None or df.empty:
            return []

        # Normalize columns
        colmap: Dict[str, str] = {}
        for col in df.columns:
            low = col.lower().strip()
            if 'start' in low and 'time' in low:
                colmap[col] = 'start_time'
            elif 'stop' in low and 'time' in low:
                colmap[col] = 'stop_time'
            elif 'conf' in low or 'prob' in low:
                colmap[col] = 'probability_label'
            elif 'label' in low and 'prob' not in low:
                colmap[col] = 'seizure_label'
        if colmap:
            df = df.rename(columns=colmap)

        if 'probability_label' not in df.columns:
            if 'seizure_label' in df.columns:
                df['probability_label'] = df['seizure_label'].apply(
                    lambda x: 1.0 if str(x).lower() in ['seiz', 'seizure', '1'] else 0.0
                )
            else:
                df['probability_label'] = 1.0

        # Keep seizure rows (prob > 0)
        rows = []
        for _, r in df.iterrows():
            try:
                s = float(r['start_time'])
                e = float(r['stop_time'])
                p = float(r.get('probability_label', 1.0))
                if e > s and p > 0:
                    rows.append((s, e, p))
            except Exception:
                continue
        return rows
    except Exception:
        return []


def _match_csv_by_base(data_dir: Path, split: str, base: str) -> Optional[Path]:
    """Find CSV path for a given recording base name (without _bi) under data_dir/edf/split.
    Uses list_bi_csvs when available; else falls back to glob.
    """
    if list_bi_csvs is not None:
        for m in ('ar', 'le'):
            try:
                candidates = list_bi_csvs(str(data_dir), split, montage=m)
                for c in candidates:
                    if Path(c).stem.replace('_bi', '') == base:
                        return Path(c)
            except Exception:
                continue
    # Fallback search
    root = data_dir / 'edf' / split
    for c in root.rglob('*_bi.csv'):
        if c.stem.replace('_bi', '') == base:
            return c
    return None


def _parse_tfrecord_labels(example_proto: tf.Tensor) -> Dict[str, tf.Tensor]:
    desc = {
        'labels': tf.io.VarLenFeature(tf.int64),
        'record_id': tf.io.FixedLenFeature([], tf.string, default_value=b''),
        'start_tp': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'n_timepoints': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'sfreq': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    parsed = tf.io.parse_single_example(example_proto, desc)
    labels = tf.sparse.to_dense(parsed['labels'])
    return {
        'labels': labels,  # (T,) or (1,) in fallback
        'record_id': parsed['record_id'],
        'start_tp': parsed['start_tp'],
        'n_timepoints': parsed['n_timepoints'],
        'sfreq': parsed['sfreq'],
    }


def verify_split(
    tfrecord_dir: Path,
    data_dir: Path,
    split: str,
    resample_fs: int,
    n_timepoints: int,
    raw_split: Optional[str] = None,
    max_files: int = 10,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compare TFRecord window labels vs raw CSV intervals for a split.
    Returns aggregate stats: total windows, tf_pos, raw_pos, agree, disagree.
    """
    split_dir = tfrecord_dir / split
    tfrecords = sorted(split_dir.glob('*.tfrecord'))
    if not tfrecords:
        raise FileNotFoundError(f"No TFRecords found in {split_dir}")
    if max_files > 0:
        tfrecords = tfrecords[:max_files]

    agg = {
        'total': 0,
        'tf_pos': 0,
        'raw_pos': 0,
        'agree': 0,
        'disagree': 0,
        'skipped': 0,
    }

    raw_split_eff = raw_split or split
    for tfp in tfrecords:
        base = tfp.stem  # matches record_id
        csv_path = _match_csv_by_base(data_dir, raw_split_eff, base)
        if csv_path is None:
            if verbose:
                print(f"[skip] CSV not found for {base}")
            continue

        intervals_s = _read_csv_intervals(csv_path)
        intervals = [
            (int(round(s * resample_fs)), int(round(e * resample_fs))) for (s, e, p) in intervals_s
        ]

        ds = tf.data.TFRecordDataset([str(tfp)], num_parallel_reads=tf.data.AUTOTUNE)
        ds = ds.map(_parse_tfrecord_labels, num_parallel_calls=tf.data.AUTOTUNE)

        file_stats = {'total': 0, 'tf_pos': 0, 'raw_pos': 0, 'agree': 0, 'disagree': 0, 'skipped': 0}

        for ex in ds:
            labels = ex['labels'].numpy().astype(np.int64)
            start_tp = int(ex['start_tp'].numpy())
            # Prefer per-example n_timepoints if present; else fall back to provided arg
            ntp = int(ex['n_timepoints'].numpy()) if int(ex['n_timepoints'].numpy()) > 0 else int(n_timepoints)

            # TFRecord window label (reduce frame labels)
            tf_win_pos = int(labels.max() > 0)

            # Raw label by overlap if metadata present, else skip compare
            if start_tp >= 0 and ntp > 0:
                end_tp = start_tp + ntp
                raw_pos = 0
                for s_idx, e_idx in intervals:
                    rs = max(0, s_idx - start_tp)
                    re = min(ntp, e_idx - start_tp)
                    if re > rs:
                        raw_pos = 1
                        break
                agree = int(raw_pos == tf_win_pos)
                disagree = 1 - agree
                file_stats['raw_pos'] += raw_pos
                file_stats['agree'] += agree
                file_stats['disagree'] += disagree
            else:
                # Missing metadata (likely fallback example); can't align precisely
                file_stats['skipped'] += 1

            file_stats['total'] += 1
            file_stats['tf_pos'] += tf_win_pos

        if verbose:
            tot = file_stats['total']
            if tot > 0:
                print(
                    f"[{split}] {base}: tf_pos={file_stats['tf_pos']}/{tot} ({file_stats['tf_pos']/tot:.1%}), "
                    f"raw_pos={file_stats['raw_pos']}/{tot - file_stats['skipped']} "
                    f"({(file_stats['raw_pos']/max(1, (tot - file_stats['skipped']))):.1%}), "
                    f"agree={file_stats['agree']}, disagree={file_stats['disagree']}, skipped={file_stats['skipped']}"
                )

        for k in agg:
            agg[k] += file_stats[k]

    if verbose:
        tot = agg['total']
        comp = tot - agg['skipped']
        print("\n=== Aggregate ===")
        print(
            f"TF windows: {tot}, comparable: {comp}\n"
            f"TF pos: {agg['tf_pos']}/{tot} ({(agg['tf_pos']/max(1, tot)):.2%})\n"
            f"RAW pos: {agg['raw_pos']}/{comp} ({(agg['raw_pos']/max(1, comp)):.2%})\n"
            f"Agree: {agg['agree']}, Disagree: {agg['disagree']}, Skipped: {agg['skipped']}"
        )

    return agg


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Verify TFRecord window labels vs TUH CSV annotations")
    ap.add_argument('--tfrecord-dir', required=True, help='Directory with TFRecords (expects train/val/test subdirs)')
    ap.add_argument('--data-dir', required=True, help='TUH root that contains edf/{train,dev,eval}')
    ap.add_argument('--split', default='train', choices=['train', 'val', 'test', 'dev', 'eval'])
    ap.add_argument('--sfreq', type=int, default=256, help='Resampled Hz used when writing TFRecords')
    ap.add_argument('--n-timepoints', type=int, default=1280, help='Samples per window (e.g., 5s * 256Hz)')
    ap.add_argument('--max-files', type=int, default=10, help='Max TFRecord files to verify per split')
    args = ap.parse_args()

    # Map split aliases
    split = args.split
    # TFRecords use train/val/test, TUH raw uses train/dev/eval
    split_map_for_raw = {'val': 'dev', 'test': 'eval', 'train': 'train', 'dev': 'dev', 'eval': 'eval'}
    raw_split = split_map_for_raw.get(split, split)

    verify_split(
        tfrecord_dir=Path(args.tfrecord_dir),
        data_dir=Path(args.data_dir),
        split=split,
        raw_split=raw_split,
        resample_fs=int(args.sfreq),
        n_timepoints=int(args.n_timepoints),
        max_files=int(args.max_files),
        verbose=True,
    )


if __name__ == '__main__':
    main()
