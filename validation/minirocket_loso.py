#!/usr/bin/env python3
"""
loso_baseline_v2.py — Phase 2.1: Comprehensive LOSO Baseline
=============================================================

Establishes the cross-subject LOSO baseline using MiniROCKET + RidgeClassifierCV
with 2000ms windows (confirmed optimal from Phase 2 diagnostics).

Features:
  - Strict Leave-One-Subject-Out cross-validation
  - Per-dataset, per-exercise configuration testing
  - Per-subject accuracy breakdown
  - Ready for normalization/domain adaptation extension (Phase 2.2+)

Usage:
    # Quick test on DB7 with 4 subjects:
    python loso_baseline_v2.py --config config.yaml --datasets ninapro_db7 --subjects 1 2 3 4

    # Full baseline on DB7:
    python loso_baseline_v2.py --config config.yaml --datasets ninapro_db7

    # All datasets (will take many hours):
    python loso_baseline_v2.py --config config.yaml --datasets ninapro_db2 ninapro_db3 ninapro_db7

    # Custom window and kernels:
    python loso_baseline_v2.py --config config.yaml --datasets ninapro_db7 --window_ms 1000 --num_kernels 8464

Phase 2.1 Output:
    - JSON results per dataset
    - Summary table in console
    - Per-subject accuracy breakdown
"""

import sys
import os
import gc
import time
import json
import argparse
import warnings
import numpy as np
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Add project root to path ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import yaml

# ── Import local MiniROCKET ──────────────────────────────────────────
from minirocket import MiniRocketPipeline

# ── Import data loaders ──────────────────────────────────────────────
try:
    from validation.data_loaders import load_ninapro_db
except ImportError:
    from data_loaders import load_ninapro_db


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2.1: LOSO Baseline with MiniROCKET + RidgeClassifierCV"
    )
    p.add_argument('--config', type=str,
                   default=os.path.join(PROJECT_ROOT, 'config.yaml'))
    p.add_argument('--datasets', nargs='+', required=True,
                   choices=['ninapro_db2', 'ninapro_db3', 'ninapro_db7'])
    p.add_argument('--subjects', nargs='+', type=int, default=None,
                   help='Subject IDs to include (default: ALL)')
    p.add_argument('--output', type=str, default=None,
                   help='Output JSON path (default: ./loso_baseline_results.json)')
    p.add_argument('--window_ms', type=int, default=2000,
                   help='Window size in ms (default: 2000 — confirmed optimal)')
    p.add_argument('--overlap', type=float, default=0.75,
                   help='Window overlap fraction (default: 0.75)')
    p.add_argument('--num_kernels', type=int, default=10000,
                   help='MiniROCKET kernels (default: 10000)')
    p.add_argument('--max_train_samples', type=int, default=80000,
                   help='Max training samples per fold (default: 80000)')
    p.add_argument('--exercise_filter', type=str, default=None,
                   choices=['E1', 'E2', 'E1+E2', 'all', 'E3'],
                   help='Filter exercises (default: all available)')
    p.add_argument('--skip_report', action='store_true',
                   help='Skip detailed per-subject report')
    return p.parse_args()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =====================================================================
# Windowing
# =====================================================================
def create_windows(emg, window_size, overlap):
    """
    Create sliding windows from raw EMG signal.
    Uses stride-trick for efficiency.

    Parameters
    ----------
    emg : np.ndarray, shape (N_samples, n_channels)
    window_size : int — number of samples per window
    overlap : float — fraction of overlap (0.0 to 1.0)

    Returns
    -------
    windows : np.ndarray, shape (N_windows, n_channels, window_size)
    """
    N, C = emg.shape
    step = max(1, int(window_size * (1.0 - overlap)))

    n_windows = (N - window_size) // step + 1
    if n_windows <= 0:
        return np.empty((0, C, window_size), dtype=np.float32)

    # Stride-trick for zero-copy windowing
    strides = (emg.strides[0] * step, emg.strides[1], emg.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        emg, shape=(n_windows, C, window_size), strides=strides
    )
    return np.ascontiguousarray(windows, dtype=np.float32)


def assign_window_labels(windows, labels, overlap):
    """
    Assign labels to windows using midpoint method.
    Each window is assigned the label of the sample at its center.
    """
    n_windows = windows.shape[0]
    N_samples = len(labels)
    window_size = windows.shape[2]
    step = max(1, int(window_size * (1 - overlap)))

    # Midpoint of each window
    mids = np.arange(n_windows) * step + window_size // 2
    mids = np.clip(mids, 0, N_samples - 1)
    return labels[mids]


# =====================================================================
# Exercise Filtering
# =====================================================================
def filter_by_exercise(labels, exercise_filter, dataset_name):
    """
    Filter labels and create valid mask based on exercise selection.

    Parameters
    ----------
    labels : np.ndarray — movement labels
    exercise_filter : str or None
    dataset_name : str

    Returns
    -------
    mask : np.ndarray (bool) — True for samples to KEEP
    n_classes : int — number of unique non-zero classes after filtering
    class_list : list — sorted list of unique non-zero labels
    """
    if exercise_filter is None:
        exercise_filter = 'all'

    if dataset_name == 'ninapro_db2':
        if exercise_filter == 'E1':
            mask = (labels >= 1) & (labels <= 17)
        elif exercise_filter == 'E2':
            mask = (labels >= 1) & (labels <= 40) & ~((labels >= 1) & (labels <= 17))
            # Actually E2 = labels 18-40
            mask = (labels >= 18) & (labels <= 40)
        elif exercise_filter == 'E3':
            mask = (labels >= 41) & (labels <= 49)
        elif exercise_filter == 'E1+E2':
            mask = (labels >= 1) & (labels <= 40)
        else:  # 'all'
            mask = np.ones(len(labels), dtype=bool)

    elif dataset_name == 'ninapro_db3':
        # DB3 with movement_map: labels 1-40 (E1=1-17, E2=18-40, E3 excluded)
        if exercise_filter == 'E1':
            mask = (labels >= 1) & (labels <= 17)
        elif exercise_filter == 'E2':
            mask = (labels >= 18) & (labels <= 40)
        elif exercise_filter == 'E1+E2':
            mask = (labels >= 1) & (labels <= 40)
        elif exercise_filter == 'all':
            mask = (labels >= 1) & (labels <= 40)
        else:
            mask = np.ones(len(labels), dtype=bool)

    elif dataset_name == 'ninapro_db7':
        # DB7: labels 1-41 (E1=1-17, E2=18-40, plus some extras)
        if exercise_filter == 'E1':
            mask = (labels >= 1) & (labels <= 17)
        elif exercise_filter == 'E2':
            mask = (labels >= 18) & (labels <= 40)
        elif exercise_filter == 'E1+E2':
            mask = (labels >= 1) & (labels <= 40)
        elif exercise_filter == 'all':
            mask = (labels >= 1)  # exclude label 0 (rest)
        else:
            mask = np.ones(len(labels), dtype=bool)

    else:
        mask = np.ones(len(labels), dtype=bool)

    # Count unique non-zero classes
    valid_labels = labels[mask]
    non_zero = valid_labels[valid_labels != 0]
    unique_classes = np.unique(non_zero) if len(non_zero) > 0 else np.array([])

    return mask, len(unique_classes), sorted(unique_classes.tolist())


# =====================================================================
# Stratified Subsample
# =====================================================================
def stratified_subsample(X, y, max_samples, random_state=42):
    """Subsample while preserving class distribution."""
    rng = np.random.RandomState(random_state)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total = len(y)
    indices = []

    for cls, count in zip(unique_classes, class_counts):
        cls_idx = np.where(y == cls)[0]
        n_sample = min(count, max(2, int(count * max_samples / total)))
        chosen = rng.choice(cls_idx, size=n_sample, replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


# =====================================================================
# Data Loading
# =====================================================================
def load_all_subjects(db_version, data_path, subjects, movement_map=None):
    """
    Load raw EMG data for all subjects.

    Returns:
        raw_data_per_subject : list of (subject_id, emg_array)
        labels_per_subject : list of (subject_id, labels_array)
    """
    raw_data = []
    label_data = []

    loader = load_ninapro_db(
        db_version=db_version,
        data_path=data_path,
        subjects=subjects,
        movement_map=movement_map,
        remove_class_zero=False
    )

    for emg, labels, meta in loader:
        subj_id = meta['subject_id']
        print(f"    Subject {subj_id}: EMG={emg.shape}, Labels={labels.shape}, "
              f"unique_labels={sorted(np.unique(labels).tolist())[:10]}...", flush=True)
        raw_data.append((subj_id, emg))
        label_data.append((subj_id, labels))

    return raw_data, label_data


# =====================================================================
# MiniROCKET LOSO (Phase 2.1 Baseline)
# =====================================================================
def minirocket_loso_baseline(dataset_name, raw_data_per_subject, labels_per_subject,
                             fs, window_size_ms=2000, overlap=0.75,
                             max_train_samples=80000, num_kernels=10000,
                             exercise_filter=None):
    """
    Run MiniROCKET + RidgeClassifierCV under strict LOSO.

    This is the Phase 2.1 baseline — NO normalization, NO domain adaptation,
    just MiniROCKET features + StandardScaler + RidgeClassifierCV.

    Parameters
    ----------
    dataset_name : str
    raw_data_per_subject : list of (subject_id, emg_array)
    labels_per_subject : list of (subject_id, labels_array)
    fs : int — sampling rate in Hz
    window_size_ms : int — window size in milliseconds
    overlap : float — window overlap fraction
    max_train_samples : int — cap training samples per fold
    num_kernels : int — number of MiniROCKET kernels
    exercise_filter : str or None — exercise filter

    Returns
    -------
    results : dict
    """
    window_size = int(window_size_ms * fs / 1000)
    n_subjects = len(raw_data_per_subject)

    # ── Exercise filtering info ──────────────────────────────────────
    # Check exercise filter on first subject's labels
    sample_labels = labels_per_subject[0][1]
    _, n_classes, class_list = filter_by_exercise(sample_labels, exercise_filter, dataset_name)

    print(f"\n    [LOSO Baseline] {n_subjects} subjects, {n_classes} classes, "
          f"window={window_size} ({window_size_ms}ms @ {fs}Hz), "
          f"overlap={overlap}, kernels={num_kernels}", flush=True)
    if exercise_filter:
        print(f"    [Exercise Filter] {exercise_filter}, classes: {class_list[:10]}{'...' if len(class_list)>10 else ''}", flush=True)

    # ── Window all subjects with exercise filtering ───────────────────
    all_windows = []
    all_win_labels = []

    for subj_idx, (subj_id, emg) in enumerate(raw_data_per_subject):
        labels = labels_per_subject[subj_idx][1]

        # Apply exercise filter
        mask, subj_n_classes, subj_classes = filter_by_exercise(labels, exercise_filter, dataset_name)

        if mask.sum() == 0:
            print(f"      Subject {subj_id}: [SKIP] No data after exercise filter", flush=True)
            all_windows.append(np.empty((0, emg.shape[1], window_size), dtype=np.float32))
            all_win_labels.append(np.array([], dtype=int))
            continue

        filtered_emg = emg[mask]
        filtered_labels = labels[mask]

        print(f"      Subject {subj_id}: windowing {filtered_emg.shape}...", end=' ', flush=True)
        t0 = time.time()
        windows = create_windows(filtered_emg, window_size, overlap)
        win_labels = assign_window_labels(windows, filtered_labels, overlap)

        # Remove class 0 windows (rest)
        non_rest_mask = win_labels != 0
        if non_rest_mask.sum() > 0 and not np.all(non_rest_mask):
            windows = windows[non_rest_mask]
            win_labels = win_labels[non_rest_mask]

        # Remove windows with classes not in the filtered class list
        if exercise_filter is not None:
            class_set = set(class_list)
            valid_mask = np.array([l in class_set for l in win_labels])
            if valid_mask.sum() > 0 and not np.all(valid_mask):
                windows = windows[valid_mask]
                win_labels = win_labels[valid_mask]

        all_windows.append(windows)
        all_win_labels.append(win_labels)

        actual_classes = len(np.unique(win_labels))
        print(f"{windows.shape[0]:,} windows, {actual_classes} classes ({time.time()-t0:.1f}s)", flush=True)

        del emg, filtered_emg
        gc.collect()

    # ── Verify all subjects have data ────────────────────────────────
    valid_subjects = []
    for i in range(n_subjects):
        if len(all_win_labels[i]) > 0:
            valid_subjects.append(i)

    if len(valid_subjects) < 2:
        print(f"    [ERROR] Only {len(valid_subjects)} valid subjects, need at least 2", flush=True)
        return {'error': 'insufficient_subjects', 'valid_subjects': len(valid_subjects)}

    # ── LOSO loop ────────────────────────────────────────────────────
    per_subject_acc = []
    per_subject_n_train = []
    per_subject_n_test = []
    per_subject_n_classes = []

    for test_idx in valid_subjects:
        test_subj_id = raw_data_per_subject[test_idx][0]
        t0 = time.time()

        # Build train/test sets
        train_windows = []
        train_labels = []
        for i in valid_subjects:
            if i == test_idx:
                continue
            if len(all_win_labels[i]) > 0:
                train_windows.append(all_windows[i])
                train_labels.append(all_win_labels[i])

        if not train_windows:
            per_subject_acc.append({
                'subject_id': int(test_subj_id),
                'accuracy': 0.0,
                'error': 'no_training_data',
                'n_train': 0,
                'n_test': 0,
            })
            continue

        X_train_raw = np.concatenate(train_windows, axis=0)
        y_train = np.concatenate(train_labels)
        X_test_raw = all_windows[test_idx]
        y_test = all_win_labels[test_idx]

        del train_windows, train_labels
        gc.collect()

        # Subsample train if needed (stratified)
        if len(y_train) > max_train_samples:
            X_train_raw, y_train = stratified_subsample(
                X_train_raw, y_train, max_train_samples
            )

        n_train_classes = len(np.unique(y_train))
        n_test_classes = len(np.unique(y_test))

        print(f"      Fold {len(per_subject_acc)+1}/{len(valid_subjects)} "
              f"(Subject {test_subj_id}): "
              f"train={X_train_raw.shape[0]:,} ({n_train_classes} cls), "
              f"test={X_test_raw.shape[0]:,} ({n_test_classes} cls)...",
              end=' ', flush=True)

        try:
            # MiniRocket + RidgeClassifierCV (StandardScaler inside pipeline)
            pipe = MiniRocketPipeline(num_kernels=num_kernels)
            pipe.fit(X_train_raw, y_train)
            acc = pipe.score(X_test_raw, y_test)

            per_subject_acc.append({
                'subject_id': int(test_subj_id),
                'accuracy': float(acc),
                'n_train': len(y_train),
                'n_test': len(y_test),
                'n_train_classes': n_train_classes,
                'n_test_classes': n_test_classes,
            })
            per_subject_n_train.append(len(y_train))
            per_subject_n_test.append(len(y_test))
            per_subject_n_classes.append(n_test_classes)

            dt = time.time() - t0
            print(f"acc={acc*100:.2f}% ({dt:.1f}s)", flush=True)

        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            per_subject_acc.append({
                'subject_id': int(test_subj_id),
                'accuracy': 0.0,
                'error': str(e),
                'n_train': len(y_train),
                'n_test': len(y_test),
            })

        del X_train_raw, X_test_raw, y_train, y_test
        gc.collect()

    # ── Aggregate results ────────────────────────────────────────────
    valid_accs = [r['accuracy'] for r in per_subject_acc if 'error' not in r]
    mean_acc = float(np.mean(valid_accs)) if valid_accs else 0.0
    std_acc = float(np.std(valid_accs, ddof=1)) if len(valid_accs) > 1 else 0.0
    median_acc = float(np.median(valid_accs)) if valid_accs else 0.0
    min_acc = float(np.min(valid_accs)) if valid_accs else 0.0
    max_acc = float(np.max(valid_accs)) if valid_accs else 0.0

    # Random chance baseline
    avg_n_classes = float(np.mean(per_subject_n_classes)) if per_subject_n_classes else 1.0
    random_baseline = 100.0 / avg_n_classes if avg_n_classes > 0 else 0.0

    return {
        'dataset': dataset_name,
        'exercise_filter': exercise_filter or 'all',
        'window_size_ms': window_size_ms,
        'overlap': overlap,
        'num_kernels': num_kernels,
        'n_subjects': len(per_subject_acc),
        'n_classes': int(avg_n_classes),
        'random_baseline_pct': round(random_baseline, 2),
        'mean_accuracy': round(mean_acc, 4),
        'std_accuracy': round(std_acc, 4),
        'median_accuracy': round(median_acc, 4),
        'min_accuracy': round(min_acc, 4),
        'max_accuracy': round(max_acc, 4),
        'ratio_vs_random': round(mean_acc / (random_baseline / 100.0), 2) if random_baseline > 0 else 0.0,
        'per_subject_accuracy': per_subject_acc,
        'avg_train_samples': int(np.mean(per_subject_n_train)) if per_subject_n_train else 0,
        'avg_test_samples': int(np.mean(per_subject_n_test)) if per_subject_n_test else 0,
    }


# =====================================================================
# Print Summary Table
# =====================================================================
def print_summary_table(all_results):
    """Print a formatted summary table of all results."""
    print("\n" + "=" * 90, flush=True)
    print("  PHASE 2.1: LOSO BASELINE RESULTS SUMMARY", flush=True)
    print("  MiniROCKET + RidgeClassifierCV + StandardScaler (no DA, no augmentation)", flush=True)
    print("=" * 90, flush=True)
    print(f"  {'Dataset':<16} {'Exercise':<10} {'Classes':>7} {'Mean%':>8} {'Std%':>7} "
          f"{'Median%':>8} {'Min%':>7} {'Max%':>7} {'vs Rand':>7}", flush=True)
    print("  " + "-" * 88, flush=True)

    for key, res in sorted(all_results.items()):
        if 'error' in res:
            print(f"  {res.get('dataset',''):<16} {res.get('exercise_filter',''):<10} "
                  f"{'ERROR: '+res['error']}", flush=True)
            continue

        print(f"  {res['dataset']:<16} {res['exercise_filter']:<10} "
              f"{res['n_classes']:>7} "
              f"{res['mean_accuracy']*100:>7.2f}% "
              f"{res['std_accuracy']*100:>6.2f}% "
              f"{res['median_accuracy']*100:>7.2f}% "
              f"{res['min_accuracy']*100:>6.2f}% "
              f"{res['max_accuracy']*100:>6.2f}% "
              f"{res['ratio_vs_random']:>6.1f}x", flush=True)

    print("=" * 90, flush=True)


def print_per_subject_report(result):
    """Print detailed per-subject accuracy breakdown."""
    print(f"\n    {'Subject':>8} {'Accuracy':>10} {'Train':>10} {'Test':>10} {'Classes':>8}", flush=True)
    print(f"    {'-'*50}", flush=True)

    for subj in result['per_subject_accuracy']:
        sid = subj['subject_id']
        acc = subj.get('accuracy', 0.0) * 100
        n_train = subj.get('n_train', 0)
        n_test = subj.get('n_test', 0)
        n_cls = subj.get('n_test_classes', '?')

        marker = ''
        if 'error' in subj:
            marker = f" [ERROR: {subj['error'][:30]}]"
            print(f"    {sid:>8} {'ERROR':>10} {n_train:>10,} {n_test:>10,} {n_cls:>8}{marker}", flush=True)
        else:
            if acc >= 30:
                marker = ' ***'
            elif acc >= 20:
                marker = ' **'
            elif acc >= 10:
                marker = ' *'
            print(f"    {sid:>8} {acc:>9.2f}% {n_train:>10,} {n_test:>10,} {n_cls:>8}{marker}", flush=True)


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    config = load_config(args.config)
    output_path = args.output or os.path.join(SCRIPT_DIR, 'loso_baseline_results.json')

    # ── Exercise configurations to test per dataset ───────────────────
    exercise_configs = {
        'ninapro_db2': ['E1', 'E2', 'E1+E2', 'all'],
        'ninapro_db3': ['E1', 'E2', 'E1+E2'],
        'ninapro_db7': ['E1', 'E2', 'E1+E2', 'all'],
    }

    all_results = {}

    print("=" * 90, flush=True)
    print("  PHASE 2.1: CROSS-SUBJECT LOSO BASELINE", flush=True)
    print("  MiniROCKET (v7.0) + RidgeClassifierCV + StandardScaler", flush=True)
    print(f"  Datasets:    {args.datasets}", flush=True)
    print(f"  Subjects:    {args.subjects or 'ALL'}", flush=True)
    print(f"  Window:      {args.window_ms}ms", flush=True)
    print(f"  Overlap:     {args.overlap}", flush=True)
    print(f"  Kernels:     {args.num_kernels}", flush=True)
    print(f"  Max Train:   {args.max_train_samples:,}", flush=True)
    print(f"  Timestamp:   {datetime.now().isoformat()}", flush=True)
    print("=" * 90, flush=True)

    for ds_key in args.datasets:
        ds_cfg = config.get('datasets', {}).get(ds_key, {})
        data_path = ds_cfg.get('path', '')
        fs = ds_cfg.get('sampling_rate', 2000)

        # Movement map for DB3
        movement_map = None
        if ds_key == 'ninapro_db3':
            movement_map = config.get('db3_to_db7_movement_map')

        # Determine exercise configs to test
        if args.exercise_filter:
            configs_to_test = [args.exercise_filter]
        else:
            configs_to_test = exercise_configs.get(ds_key, ['all'])

        for ex_filter in configs_to_test:
            config_key = f"{ds_key}_{ex_filter or 'all'}"

            print(f"\n{'#'*90}", flush=True)
            print(f"  DATASET: {ds_key.upper()} | Exercise: {ex_filter or 'all'}", flush=True)
            print(f"  Path: {data_path}", flush=True)
            print(f"  FS: {fs}Hz, Window: {args.window_ms}ms, Overlap: {args.overlap}", flush=True)
            print(f"  Subjects: {args.subjects or 'ALL'}", flush=True)
            print(f"{'#'*90}", flush=True)

            # Load raw data
            t0 = time.time()
            raw_data, label_data = load_all_subjects(
                db_version=ds_key.replace('ninapro_', '').upper(),
                data_path=data_path,
                subjects=args.subjects,
                movement_map=movement_map,
            )
            load_time = time.time() - t0
            print(f"  Loaded {len(raw_data)} subjects in {load_time:.1f}s", flush=True)

            if not raw_data:
                print(f"  [SKIP] No data for {ds_key}", flush=True)
                all_results[config_key] = {'error': 'no_data_loaded'}
                continue

            # Run LOSO baseline
            t0 = time.time()
            result = minirocket_loso_baseline(
                dataset_name=ds_key,
                raw_data_per_subject=raw_data,
                labels_per_subject=label_data,
                fs=fs,
                window_size_ms=args.window_ms,
                overlap=args.overlap,
                max_train_samples=args.max_train_samples,
                num_kernels=args.num_kernels,
                exercise_filter=ex_filter,
            )
            result['load_time_s'] = round(load_time, 1)
            result['total_time_s'] = round(time.time() - t0, 1)
            all_results[config_key] = result

            # Print result
            if 'error' not in result:
                print(f"\n  RESULT: {result['mean_accuracy']*100:.2f} +/- "
                      f"{result['std_accuracy']*100:.2f}% "
                      f"(median={result['median_accuracy']*100:.2f}%, "
                      f"random={result['random_baseline_pct']:.1f}%, "
                      f"ratio={result['ratio_vs_random']:.1f}x)", flush=True)
            else:
                print(f"\n  RESULT: ERROR - {result.get('error', 'unknown')}", flush=True)

            # Per-subject report
            if not args.skip_report and 'per_subject_accuracy' in result:
                print_per_subject_report(result)

            del raw_data, label_data
            gc.collect()

    # ── Summary table ─────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Save JSON ─────────────────────────────────────────────────────
    output = {
        'phase': '2.1',
        'title': 'LOSO Baseline - MiniROCKET + RidgeClassifierCV',
        'description': 'Cross-subject baseline with NO normalization, NO domain adaptation, NO augmentation',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'window_ms': args.window_ms,
            'overlap': args.overlap,
            'num_kernels': args.num_kernels,
            'max_train_samples': args.max_train_samples,
        },
        'datasets': args.datasets,
        'subjects_filter': args.subjects,
        'results': {},
    }

    for key, res in all_results.items():
        output['results'][key] = res

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_path}", flush=True)

    # ── Quick analysis ────────────────────────────────────────────────
    print("\n" + "=" * 90, flush=True)
    print("  QUICK ANALYSIS", flush=True)
    print("=" * 90, flush=True)

    for key, res in sorted(all_results.items()):
        if 'error' in res:
            continue
        mean_acc = res['mean_accuracy'] * 100
        rand = res['random_baseline_pct']
        if mean_acc > rand * 3:
            status = "PROMISING - well above random"
        elif mean_acc > rand * 2:
            status = "MODERATE - 2x+ random"
        elif mean_acc > rand * 1.5:
            status = "WEAK - slight improvement over random"
        else:
            status = "NEEDS WORK - near random chance"

        print(f"  {key:<30} {mean_acc:>6.2f}% (random={rand:.1f}%) -> {status}", flush=True)

    print("=" * 90, flush=True)
    print("\n  Phase 2.1 COMPLETE. Next: Phase 2.2 (Normalization) if baseline established.", flush=True)


if __name__ == '__main__':
    main()
