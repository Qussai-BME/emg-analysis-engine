#!/usr/bin/env python3
"""
main.py  — v6.1

Change vs v6:
  ✓ disable_cache mode (default: true)
    All features are kept in RAM. No disk writes during parallel processing.
    Eliminates the Windows multiprocessing race condition that caused
    "0 bytes written" / "No space left on device" errors.

    RAM budget: 36 subjects × ~1200 windows × 364 features × 8 bytes ≈ 126 MB
    Well within any modern machine's capacity.

  ✓ Cache is still used when disable_cache: false (for huge datasets like CEMHSEY)
  ✓ Checkpoint writes removed when cache disabled (no partial state to resume)
"""

import sys
import os
import time
import argparse
import yaml
import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from validation.checkpoint       import Checkpoint
from validation.data_loaders     import load_uci_gesture, load_ninapro_db7, load_cemhsey
from validation.process_engine   import extract_features_per_channel
from validation.metrics          import classification_accuracy, feature_statistics
from validation.report_generator import generate_report


def parse_args():
    parser = argparse.ArgumentParser(description="EMG Validation Suite v6.1")
    parser.add_argument('--config',   type=str, default='config.yaml')
    parser.add_argument('--datasets', nargs='+',
                        choices=['uci', 'ninapro7', 'cemhsey'], required=True)
    parser.add_argument('--quick',  action='store_true',
                        help='Process only first subject (smoke test)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-cached subjects (only when disable_cache: false)')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def dataset_name_mapping(name):
    return {'UCI_Gesture': 'uci',
            'Ninapro_DB7': 'ninapro7',
            'CEMHSEY':     'cemhsey'}.get(name, name)


def get_loader_for_subject(dataset_name, data_path, subject_id, day=None):
    if dataset_name == 'UCI_Gesture':
        return load_uci_gesture(data_path, subjects=[subject_id])
    elif dataset_name == 'Ninapro_DB7':
        return load_ninapro_db7(data_path, subjects=[subject_id])
    elif dataset_name == 'CEMHSEY':
        return load_cemhsey(data_path, subjects=[subject_id],
                            days=[day] if day else None)
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ─────────────────────────────────────────────────────────────────
#  Worker function
# ─────────────────────────────────────────────────────────────────
def process_one_subject(args):
    """
    Load subject → extract features → return in memory.

    When disable_cache=True (default):
      - No disk writes at all during parallel phase
      - Eliminates Windows multiprocessing race conditions
      - Features returned directly in the result dict

    When disable_cache=False:
      - Cache files written to cache_dir (useful for CEMHSEY)
      - Stale cache auto-detected via total_features mismatch
    """
    subject_key, dataset_name, data_path, day, proc_config, cache_dir, disable_cache = args

    try:
        # ── Cache hit (only when caching enabled) ─────────────────
        if not disable_cache:
            cache_file = os.path.join(cache_dir, f"{subject_key}_features.npy")
            label_file = os.path.join(cache_dir, f"{subject_key}_labels.npy")
            meta_file  = os.path.join(cache_dir, f"{subject_key}_meta.pkl")

            if all(os.path.exists(x) for x in [cache_file, label_file, meta_file]):
                features      = np.load(cache_file)
                window_labels = np.load(label_file)
                with open(meta_file, 'rb') as f:
                    mc = pickle.load(f)

                # Validate cache feature count
                expected = mc.get('total_features')
                if expected is not None and features.shape[1] != expected:
                    print(f"[cache STALE] {subject_key}: "
                          f"{features.shape[1]} != {expected}. Recomputing.", flush=True)
                    for stale in [cache_file, label_file, meta_file]:
                        if os.path.exists(stale):
                            os.remove(stale)
                else:
                    print(f"[cache] {subject_key}  {features.shape}", flush=True)
                    return {
                        'features':      features,
                        'labels':        window_labels,
                        'subject_key':   subject_key,
                        'feature_names': mc['feature_names'],
                        'n_channels':    mc['n_channels'],
                        'sampling_rate': mc.get('sampling_rate'),
                        'success':       True
                    }

        # ── Load raw EMG data ──────────────────────────────────────
        subj_part = subject_key.split('_')[0]
        loader    = get_loader_for_subject(
            dataset_name, data_path, subj_part,
            day=subject_key.split('_')[1] if '_' in subject_key else None
        )
        emg, labels, meta = next(loader)

        # ── Extract features ───────────────────────────────────────
        features_flat, windows, snr, feature_names = \
            extract_features_per_channel(emg, proc_config)

        n_windows  = features_flat.shape[0]
        n_channels = emg.shape[1]

        # ── Align labels to window midpoints ──────────────────────
        win_labels = np.array([
            labels[min((s + e) // 2, len(labels) - 1)]
            for s, e in windows
        ])
        assert len(win_labels) == n_windows, \
            f"Label/window mismatch: {len(win_labels)} vs {n_windows}"

        # ── Save to cache (only when caching enabled) ─────────────
        if not disable_cache:
            os.makedirs(cache_dir, exist_ok=True)
            try:
                np.save(cache_file, features_flat)
                np.save(label_file, win_labels)
                mc = {
                    'feature_names':  feature_names,
                    'n_channels':     n_channels,
                    'total_features': features_flat.shape[1],
                    'sampling_rate':  meta.get('sampling_rate'),
                }
                with open(meta_file, 'wb') as f:
                    pickle.dump(mc, f)
                print(f"[cached] {subject_key}  {features_flat.shape}", flush=True)
            except OSError as e:
                # Cache write failed (disk full, permissions, etc.)
                # This is non-fatal — we already have the features in memory
                print(f"[cache WRITE FAILED] {subject_key}: {e}. "
                      f"Continuing in-memory.", flush=True)

        print(f"[done] {subject_key}  features={features_flat.shape}  "
              f"labels={win_labels.shape}", flush=True)
        return {
            'features':      features_flat,
            'labels':        win_labels,
            'subject_key':   subject_key,
            'feature_names': feature_names,
            'n_channels':    n_channels,
            'sampling_rate': meta.get('sampling_rate'),
            'success':       True
        }

    except Exception as e:
        import traceback
        print(f"[error] {subject_key}: {e}", flush=True)
        traceback.print_exc()
        return {'subject_key': subject_key, 'success': False, 'error': str(e)}


# ─────────────────────────────────────────────────────────────────
#  Dataset pipeline
# ─────────────────────────────────────────────────────────────────
def process_dataset(loader_func, dataset_name, config,
                    checkpoint, quick=False, resume=False):

    proc_config   = config['processing']
    output_dir    = config['output_dir']
    disable_cache = config.get('disable_cache', True)
    cache_dir     = os.path.join(output_dir, 'cache', dataset_name)

    if not disable_cache:
        os.makedirs(cache_dir, exist_ok=True)

    processed_keys = checkpoint.get(dataset_name, []) if not disable_cache else []
    processed_set  = set(processed_keys)

    all_features, all_labels, all_groups = [], [], []
    issues        = []
    n_subjects    = 0
    n_channels    = None
    sampling_rate = None
    feat_names    = None

    # ── Build task list ───────────────────────────────────────────
    tasks = []
    for emg, labels, meta in loader_func:
        subject_key = f"{meta['subject']}_{meta.get('day', '')}".rstrip('_')
        if resume and not disable_cache and subject_key in processed_set:
            tqdm.write(f"[skip] {subject_key}")
            continue
        data_path = config['datasets'][dataset_name_mapping(dataset_name)]['path']
        tasks.append((
            subject_key, dataset_name, data_path,
            meta.get('day'), proc_config, cache_dir, disable_cache
        ))
        if quick:
            break

    if not tasks:
        print("No subjects to process.", flush=True)
        return None

    # ── Process tasks ─────────────────────────────────────────────
    parallel = config.get('parallel_processing', False)
    n_jobs   = min(config.get('n_jobs', cpu_count()), len(tasks))

    if parallel and len(tasks) > 1:
        print(f"Parallel: {len(tasks)} subjects / {n_jobs} workers  "
              f"[cache={'OFF' if disable_cache else 'ON'}]", flush=True)
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(process_one_subject, tasks),
                total=len(tasks), desc=dataset_name
            ))
    else:
        print(f"Sequential: {len(tasks)} subjects  "
              f"[cache={'OFF' if disable_cache else 'ON'}]", flush=True)
        results = [process_one_subject(t)
                   for t in tqdm(tasks, desc=dataset_name)]

    # ── Collect results ───────────────────────────────────────────
    for res in results:
        if res['success']:
            all_features.append(res['features'])
            all_labels.append(res['labels'])
            all_groups.append(np.full(len(res['labels']), n_subjects))
            if feat_names is None:
                feat_names    = res['feature_names']
                n_channels    = res['n_channels']
            if sampling_rate is None:
                sampling_rate = res.get('sampling_rate')
            if not disable_cache:
                processed_set.add(res['subject_key'])
                try:
                    checkpoint.update(dataset_name, list(processed_set))
                except OSError as e:
                    print(f"[checkpoint WRITE FAILED]: {e}", flush=True)
            n_subjects += 1
        else:
            issues.append(f"{res['subject_key']}: {res.get('error', '?')}")

    if not all_features:
        print("No features collected — all subjects failed.", flush=True)
        return None

    # ── Aggregate ─────────────────────────────────────────────────
    print(f"\n{'='*55}", flush=True)
    print("Aggregating …", flush=True)
    X      = np.vstack(all_features)
    y      = np.hstack(all_labels)
    groups = np.hstack(all_groups)
    print(f"Raw: X={X.shape}  y={y.shape}  subjects={len(np.unique(groups))}",
          flush=True)

    # ── Remove class 0 ────────────────────────────────────────────
    if config.get('remove_class_zero', True):
        mask      = y != 0
        n_removed = int((~mask).sum())
        X, y, groups = X[mask], y[mask], groups[mask]
        print(f"Removed {n_removed} class-0 windows ({100.*n_removed/(n_removed+len(y)):.1f}%). "
              f"Remaining: {len(y)}", flush=True)

    print(f"Final: X={X.shape}  classes={sorted(np.unique(y).tolist())}",
          flush=True)

    # ── Feature statistics ────────────────────────────────────────
    print("Computing feature statistics …", flush=True)
    feat_stats = feature_statistics(X, y, feat_names, max_features=30)

    # ── Classification ────────────────────────────────────────────
    class_names = sorted(np.unique(y).tolist())
    classification_result = None

    if len(class_names) > 1 and len(np.unique(groups)) >= 2:
        print("Starting LOSO-CV …", flush=True)
        t0 = time.time()

        clf_cfg         = config['classification']
        acc, std, cm = classification_accuracy(
            X, y, groups,
            classifier      = clf_cfg.get('classifier',         'XGBoost'),
            use_grid_search = clf_cfg.get('use_grid_search',    False),
            svm_c           = clf_cfg.get('svm_c',              1.0),
            svm_gamma       = clf_cfg.get('svm_gamma',          'scale'),
            class_weight    = clf_cfg.get('class_weight',       None),
            pca_components  = clf_cfg.get('pca_components',     None),
            n_estimators    = clf_cfg.get('n_estimators',       300),
            n_top_features  = clf_cfg.get('n_top_features',     150),
            feature_selection = clf_cfg.get('feature_selection','f_classif')
        )
        print(f"Classification: {acc:.4f} ± {std:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        classification_result = (acc, std, cm)
    else:
        issues.append("Classification skipped: insufficient classes or subjects.")

    # ── Save report ───────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    return {
        'n_subjects':     n_subjects,
        'n_channels':     n_channels,
        'sampling_rate':  sampling_rate,
        'n_movements':    len(class_names),
        'class_names':    [str(c) for c in class_names],
        'feature_stats':  {str(k): v for k, v in feat_stats.items()},
        'classification': classification_result,
        'issues':         issues
    }


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    config = load_config(args.config)

    # Checkpoint only meaningful when caching is on
    disable_cache = config.get('disable_cache', True)
    chk_path = os.path.join(config['output_dir'], 'checkpoint.pkl')
    if not disable_cache:
        os.makedirs(config['output_dir'], exist_ok=True)
    chk = Checkpoint(chk_path) if not disable_cache else None

    # Dummy checkpoint when cache disabled
    class _NoCheckpoint:
        def get(self, *a, **kw): return []
        def update(self, *a, **kw): pass
    if chk is None:
        chk = _NoCheckpoint()

    for ds in args.datasets:
        ds_key = ds.lower()

        if ds_key == 'uci':
            path = config['datasets']['uci']['path']
            if not os.path.exists(path):
                print(f"UCI path not found: {path}", flush=True)
                continue
            loader = load_uci_gesture(
                path,
                sampling_rate=config['datasets']['uci'].get('sampling_rate', 1000),
                subjects=[1] if args.quick else None
            )
            res = process_dataset(loader, 'UCI_Gesture', config, chk,
                                   quick=args.quick, resume=args.resume)
            if res:
                os.makedirs(config['output_dir'], exist_ok=True)
                generate_report('UCI_Gesture', config['processing'],
                                res, config['output_dir'])
                print("UCI_Gesture report done.", flush=True)

        elif ds_key == 'ninapro7':
            path = config['datasets'].get('ninapro7', {}).get('path', '')
            if not os.path.exists(path):
                print(f"Ninapro7 path not found: {path}", flush=True)
                continue
            loader = load_ninapro_db7(path, subjects=[1] if args.quick else None)
            res = process_dataset(loader, 'Ninapro_DB7', config, chk,
                                   quick=args.quick, resume=args.resume)
            if res:
                os.makedirs(config['output_dir'], exist_ok=True)
                generate_report('Ninapro_DB7', config['processing'],
                                res, config['output_dir'])

        elif ds_key == 'cemhsey':
            path = config['datasets'].get('cemhsey', {}).get('path', '')
            if not os.path.exists(path):
                print(f"CEMHSEY path not found: {path}", flush=True)
                continue
            loader = load_cemhsey(path, subjects=[1] if args.quick else None)
            res = process_dataset(loader, 'CEMHSEY', config, chk,
                                   quick=args.quick, resume=args.resume)
            if res:
                os.makedirs(config['output_dir'], exist_ok=True)
                generate_report('CEMHSEY', config['processing'],
                                res, config['output_dir'])


if __name__ == '__main__':
    main()