#!/usr/bin/env python3
"""
validate_engine.py — v7.0

Changes:
  + Uses new evaluate_model() with configurable validation strategy.
  + Passes train_ratio and random_state from config.
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
from validation.data_loaders     import load_uci_gesture, load_ninapro_db7_custom, load_cemhsey
from validation.process_engine   import extract_features_per_channel
from validation.metrics          import evaluate_model, feature_statistics
from validation.report_generator import generate_report


def parse_args():
    parser = argparse.ArgumentParser(description="EMG Validation Suite v7.0")
    parser.add_argument('--config',   type=str, default='config.yaml')
    parser.add_argument('--datasets', nargs='+',
                        choices=['uci', 'ninapro_db7', 'cemhsey'], required=True)
    parser.add_argument('--quick',  action='store_true',
                        help='Process only first subject')
    parser.add_argument('--resume', action='store_true',
                        help='Skip cached subjects (only when caching enabled)')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def dataset_name_mapping(name):
    return {'UCI_Gesture': 'uci',
            'Ninapro_DB7': 'ninapro_db7',
            'CEMHSEY':     'cemhsey'}.get(name, name)


def get_loader_for_subject(dataset_name, data_path, subject_id, day=None):
    if dataset_name == 'UCI_Gesture':
        return load_uci_gesture(data_path, subjects=[subject_id])
    elif dataset_name == 'Ninapro_DB7':
        return load_ninapro_db7_custom(data_path, subjects=[subject_id])
    elif dataset_name == 'CEMHSEY':
        return load_cemhsey(data_path, subjects=[subject_id],
                            days=[day] if day else None)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def process_one_subject(args):
    subject_key, dataset_name, data_path, day, proc_config, cache_dir, disable_cache = args
    try:
        if not disable_cache:
            cache_file = os.path.join(cache_dir, f"{subject_key}_features.npy")
            label_file = os.path.join(cache_dir, f"{subject_key}_labels.npy")
            meta_file  = os.path.join(cache_dir, f"{subject_key}_meta.pkl")
            if all(os.path.exists(x) for x in [cache_file, label_file, meta_file]):
                features = np.load(cache_file)
                window_labels = np.load(label_file)
                with open(meta_file, 'rb') as f:
                    mc = pickle.load(f)
                expected = mc.get('total_features')
                if expected is not None and features.shape[1] != expected:
                    for stale in [cache_file, label_file, meta_file]:
                        if os.path.exists(stale):
                            os.remove(stale)
                else:
                    print(f"[cache] {subject_key}  {features.shape}", flush=True)
                    return {
                        'features': features, 'labels': window_labels,
                        'subject_key': subject_key, 'feature_names': mc['feature_names'],
                        'n_channels': mc['n_channels'], 'sampling_rate': mc.get('sampling_rate'),
                        'success': True
                    }

        loader = get_loader_for_subject(dataset_name, data_path, subject_key, day)
        emg, labels, meta = next(loader)
        features_flat, windows, snr, feature_names = extract_features_per_channel(emg, proc_config)

        win_labels = np.array([
            labels[min((s + e) // 2, len(labels) - 1)]
            for s, e in windows
        ])

        if not disable_cache:
            os.makedirs(cache_dir, exist_ok=True)
            try:
                np.save(cache_file, features_flat)
                np.save(label_file, win_labels)
                mc = {
                    'feature_names': feature_names, 'n_channels': emg.shape[1],
                    'total_features': features_flat.shape[1],
                    'sampling_rate': meta.get('sampling_rate')
                }
                with open(meta_file, 'wb') as f:
                    pickle.dump(mc, f)
                print(f"[cached] {subject_key}  {features_flat.shape}", flush=True)
            except OSError as e:
                print(f"[cache WRITE FAILED] {subject_key}: {e}", flush=True)

        print(f"[done] {subject_key}  features={features_flat.shape}", flush=True)
        return {
            'features': features_flat, 'labels': win_labels, 'subject_key': subject_key,
            'feature_names': feature_names, 'n_channels': emg.shape[1],
            'sampling_rate': meta.get('sampling_rate'), 'success': True
        }

    except Exception as e:
        import traceback
        print(f"[error] {subject_key}: {e}", flush=True)
        traceback.print_exc()
        return {'subject_key': subject_key, 'success': False, 'error': str(e)}


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
    n_subjects = 0
    feat_names = None

    tasks = []
    for emg, labels, meta in loader_func:
        subject_key = meta['subject']
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

    parallel = config.get('parallel_processing', False)
    n_jobs   = min(config.get('n_jobs', cpu_count()), len(tasks))

    if parallel and len(tasks) > 1:
        print(f"Parallel: {len(tasks)} subjects / {n_jobs} workers", flush=True)
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(pool.imap(process_one_subject, tasks),
                                total=len(tasks), desc=dataset_name))
    else:
        results = [process_one_subject(t) for t in tqdm(tasks, desc=dataset_name)]

    for res in results:
        if res['success']:
            all_features.append(res['features'])
            all_labels.append(res['labels'])
            all_groups.append(np.full(len(res['labels']), n_subjects))
            if feat_names is None:
                feat_names = res['feature_names']
            if not disable_cache:
                processed_set.add(res['subject_key'])
                try:
                    checkpoint.update(dataset_name, list(processed_set))
                except OSError:
                    pass
            n_subjects += 1

    if not all_features:
        print("No features collected.", flush=True)
        return None

    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    groups = np.hstack(all_groups)

    if config.get('remove_class_zero', True):
        mask = y != 0
        X, y, groups = X[mask], y[mask], groups[mask]

    class_names = sorted(np.unique(y).tolist())
    feat_stats = feature_statistics(X, y, feat_names, max_features=30)

    classification_result = None
    if len(class_names) > 1:
        val_cfg = config.get('validation', {})
        strategy = val_cfg.get('strategy', 'loso')
        train_ratio = val_cfg.get('train_ratio', 0.7)
        random_state = val_cfg.get('random_state', 42)

        clf_cfg = config['classification']
        t0 = time.time()
        acc, std, cm = evaluate_model(
            X, y, groups,
            strategy=strategy,
            train_ratio=train_ratio,
            random_state=random_state,
            classifier=clf_cfg.get('classifier', 'XGBoost'),
            svm_c=clf_cfg.get('svm_c', 1.0),
            svm_gamma=clf_cfg.get('svm_gamma', 'scale'),
            class_weight=clf_cfg.get('class_weight', None),
            pca_components=clf_cfg.get('pca_components', None),
            n_estimators=clf_cfg.get('n_estimators', 300),
            n_top_features=clf_cfg.get('n_top_features', 150),
            feature_selection=clf_cfg.get('feature_selection', 'f_classif')
        )
        print(f"Classification ({strategy}): {acc:.4f} ± {std:.4f}  ({time.time()-t0:.1f}s)")
        classification_result = (acc, std, cm)

    os.makedirs(output_dir, exist_ok=True)
    return {
        'n_subjects': n_subjects,
        'n_channels': res['n_channels'] if results else None,
        'sampling_rate': results[0].get('sampling_rate') if results else None,
        'n_movements': len(class_names),
        'class_names': [str(c) for c in class_names],
        'feature_stats': {str(k): v for k, v in feat_stats.items()},
        'classification': classification_result,
        'issues': []
    }


def main():
    args   = parse_args()
    config = load_config(args.config)

    if 'ninapro_db7' in args.datasets and 'ninapro_db7' not in config.get('datasets', {}):
        config.setdefault('datasets', {})['ninapro_db7'] = {
            'path': r'E:\NinaProDB7',
            'sampling_rate': 2000
        }

    disable_cache = config.get('disable_cache', True)
    chk_path = os.path.join(config['output_dir'], 'checkpoint.pkl')
    if not disable_cache:
        os.makedirs(config['output_dir'], exist_ok=True)
    chk = Checkpoint(chk_path) if not disable_cache else type('', (), {'get': lambda *a: [], 'update': lambda *a: None})()

    for ds in args.datasets:
        ds_key = ds.lower()
        if ds_key == 'uci':
            path = config['datasets']['uci']['path']
            loader = load_uci_gesture(path, subjects=[1] if args.quick else None)
            res = process_dataset(loader, 'UCI_Gesture', config, chk,
                                   quick=args.quick, resume=args.resume)
            if res:
                generate_report('UCI_Gesture', config['processing'], res, config['output_dir'])

        elif ds_key == 'ninapro_db7':
            path = config['datasets']['ninapro_db7']['path']
            loader = load_ninapro_db7_custom(path, subjects=[1] if args.quick else None)
            res = process_dataset(loader, 'Ninapro_DB7', config, chk,
                                   quick=args.quick, resume=args.resume)
            if res:
                generate_report('Ninapro_DB7', config['processing'], res, config['output_dir'])

        elif ds_key == 'cemhsey':
            path = config['datasets'].get('cemhsey', {}).get('path', '')
            loader = load_cemhsey(path, subjects=[1] if args.quick else None)
            res = process_dataset(loader, 'CEMHSEY', config, chk,
                                   quick=args.quick, resume=args.resume)
            if res:
                generate_report('CEMHSEY', config['processing'], res, config['output_dir'])


if __name__ == '__main__':
    main()