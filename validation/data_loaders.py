"""
data_loaders.py  — v5.5

Changes vs v5.4:
  + Fixed subject filtering in load_uci_physical_action: now accepts both 'S1' and 1.
  + Added clearer warnings when no data found for a requested subject.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_uci_gesture(data_path, sampling_rate=None, subjects=None):
    """
    Universal UCI-style loader.

    Supports:
      - Original UCI  (6 channels, label from filename)
      - Pattern DB    (10 columns: time, 8 EMG, class)

    Auto-detects format.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"UCI data path not found: {data_path}")

    subject_folders = sorted(
        f for f in os.listdir(data_path) if f.lower().startswith('subject')
    )
    if subjects is not None:
        subject_folders = [
            sf for sf in subject_folders
            if any(str(s) in sf for s in subjects)
        ]

    for subj_folder in subject_folders:
        subj_path = os.path.join(data_path, subj_folder)
        all_files = sorted(
            f for f in os.listdir(subj_path)
            if f.endswith('.csv') or f.endswith('.txt')
        )

        emg_list   = []
        label_list = []
        subject_fs = None

        for file_name in all_files:
            file_path = os.path.join(subj_path, file_name)
            try:
                # Detect header
                df_probe = pd.read_csv(
                    file_path, sep=r'\s+', engine='python',
                    header=0, nrows=5, dtype=str
                )
                has_header = any(
                    any(c.isalpha() for c in str(col))
                    for col in df_probe.columns
                )

                read_kw = dict(sep=r'\s+', engine='python',
                               header=0 if has_header else None)

                df = pd.read_csv(file_path, **read_kw)
                if df.shape[1] < 6:
                    # Try comma-separated as fallback
                    df = pd.read_csv(file_path,
                                     header=0 if has_header else None)

                n_cols = df.shape[1]

                if n_cols == 10:
                    # Pattern Database: time | 8 EMG | class
                    emg    = df.iloc[:, 1:9].values.astype(np.float64)
                    labels = df.iloc[:, 9].values.astype(np.int32)
                    if subject_fs is None and sampling_rate is None:
                        time_col = df.iloc[:, 0].values.astype(float)
                        if len(time_col) > 1:
                            dt = np.median(np.diff(time_col))
                            subject_fs = int(1000 / dt) if dt > 0 else 1000
                        else:
                            subject_fs = 1000
                else:
                    # Original UCI: drop time column if non-numeric
                    first_col = df.iloc[:, 0]
                    if not np.issubdtype(first_col.dtype, np.number):
                        df = df.iloc[:, 1:]
                    emg    = df.values.astype(np.float64)
                    match  = re.search(r'(\d+)', file_name)
                    label  = int(match.group(1)) if match else -1
                    labels = np.full(emg.shape[0], label, dtype=np.int32)
                    if sampling_rate is None:
                        subject_fs = 1000

                emg_list.append(emg)
                label_list.append(labels)

            except Exception as e:
                warnings.warn(f"Error reading {file_path}: {e}")
                continue

        if emg_list:
            emg_all    = np.vstack(emg_list)
            labels_all = np.hstack(label_list)
            metadata   = {
                'dataset':      'UCI Gesture',
                'subject':      subj_folder,
                'sampling_rate': sampling_rate or subject_fs,
                'n_channels':    emg_all.shape[1],
                'n_samples':     emg_all.shape[0],
                'file_count':    len(all_files)
            }
            yield emg_all, labels_all, metadata


def load_ninapro_db7(data_path, subjects=None):
    """
    Ninapro DB7 loader.
    Supports HDF5 (v7.3) and legacy MAT files.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Ninapro DB7 path not found: {data_path}")

    mat_files = sorted(
        f for f in os.listdir(data_path)
        if f.endswith('.mat') and f.startswith('S')
    )
    if subjects is not None:
        mat_files = [
            f for f in mat_files
            if any(f'_{s}_' in f or f'S{s}.mat' in f for s in subjects)
        ]

    for mat_file in mat_files:
        mat_path = os.path.join(data_path, mat_file)
        try:
            data   = loadmat(mat_path)
            emg    = data['emg'].astype(np.float64)
            labels = data['restimulus'].squeeze().astype(np.int32)
            fs     = int(data['sampling_frequency']) if 'sampling_frequency' in data else 2000
        except Exception:
            try:
                import h5py
                with h5py.File(mat_path, 'r') as f:
                    emg    = np.array(f['emg']).T
                    if emg.ndim == 2 and emg.shape[0] < emg.shape[1]:
                        emg = emg.T
                    labels = np.array(f['restimulus']).squeeze()
                    fs     = int(np.array(f['sampling_frequency']).item()) \
                             if 'sampling_frequency' in f else 2000
                emg    = emg.astype(np.float64)
                labels = labels.astype(np.int32)
            except Exception as e2:
                warnings.warn(f"Failed to load {mat_file}: {e2}")
                continue

        yield emg, labels, {
            'dataset':      'Ninapro DB7',
            'subject':      mat_file.replace('.mat', ''),
            'sampling_rate': fs,
            'n_channels':    emg.shape[1],
            'n_samples':     emg.shape[0]
        }


def load_ninapro_db7_custom(root_path, subjects=None):
    """
    Custom loader for NinaPro DB7 based on actual file structure.
    
    Expected structure:
        root_path/
            Subject_1/
                S1_E1_A1.mat
                S1_E2_A1.mat
            Subject_2/
                ...
    
    Each .mat file contains:
        'emg'        : (n_samples, 12) EMG channels
        'restimulus' : (n_samples, 1) gesture labels (0 = rest, 1..41 = gestures)
    Sampling rate defaults to 2000 Hz (DB7 standard).
    """
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Path not found: {root_path}")

    # Find all subject folders (directories) inside root_path
    subject_folders = [
        d for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    ]
    if subjects is not None:
        # Filter subjects if a list of subject IDs is provided
        subject_folders = [
            sf for sf in subject_folders
            if any(str(s).lower() in sf.lower() for s in subjects)
        ]

    for subj_folder in sorted(subject_folders):
        subj_path = os.path.join(root_path, subj_folder)
        mat_files = [
            f for f in os.listdir(subj_path)
            if f.endswith('.mat')
        ]

        emg_list = []
        labels_list = []
        fs = 2000   # Default for DB7

        for mat_file in sorted(mat_files):
            mat_path = os.path.join(subj_path, mat_file)
            try:
                # Attempt loading with scipy.io (works for v7.2 and earlier)
                data = loadmat(mat_path)
                emg = data['emg'].astype(np.float64)
                labels = data['restimulus'].squeeze().astype(np.int32)

                # If sampling frequency exists, use it
                if 'sampling_frequency' in data:
                    fs = int(data['sampling_frequency'].item())
            except (NotImplementedError, TypeError):
                # Fallback for MATLAB v7.3 files (HDF5)
                import h5py
                with h5py.File(mat_path, 'r') as f:
                    emg = np.array(f['emg']).T
                    if emg.ndim == 2 and emg.shape[0] < emg.shape[1]:
                        emg = emg.T
                    labels = np.array(f['restimulus']).squeeze()
                    if 'sampling_frequency' in f:
                        fs = int(np.array(f['sampling_frequency']).item())
                emg = emg.astype(np.float64)
                labels = labels.astype(np.int32)
            except Exception as e:
                warnings.warn(f"Failed to load {mat_file} in {subj_folder}: {e}")
                continue

            emg_list.append(emg)
            labels_list.append(labels)

        if emg_list:
            # Concatenate all data from this subject
            emg_combined = np.vstack(emg_list)
            labels_combined = np.hstack(labels_list)
            metadata = {
                'dataset': 'Ninapro DB7 (custom)',
                'subject': subj_folder,
                'sampling_rate': fs,
                'n_channels': emg_combined.shape[1],
                'n_samples': emg_combined.shape[0],
                'files_processed': len(mat_files)
            }
            yield emg_combined, labels_combined, metadata


def load_cemhsey(data_path, subjects=None, days=None):
    """
    CEMHSEY loader.
    Structure: <data_path>/s1/day1.mat, day2.mat, ...
    Expected variables: 'emg', 'labels', optionally 'sampling_rate'.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CEMHSEY path not found: {data_path}")

    subj_dirs = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d.startswith('s')
    )
    if subjects is not None:
        subj_dirs = [d for d in subj_dirs if any(str(s) in d for s in subjects)]

    for subj_dir in subj_dirs:
        subj_path = os.path.join(data_path, subj_dir)
        day_files = sorted(f for f in os.listdir(subj_path) if f.endswith('.mat'))
        if days is not None:
            day_files = [f for f in day_files if any(str(d) in f for d in days)]

        for day_file in day_files:
            mat_path = os.path.join(subj_path, day_file)
            try:
                data   = loadmat(mat_path)
                emg    = data['emg'].astype(np.float64)
                labels = data['labels'].squeeze().astype(np.int32)
                fs     = int(data['sampling_rate'].item()) \
                         if 'sampling_rate' in data else 2000
                yield emg, labels, {
                    'dataset':      'CEMHSEY',
                    'subject':      subj_dir,
                    'day':          day_file.replace('.mat', ''),
                    'sampling_rate': fs,
                    'n_channels':    emg.shape[1],
                    'n_samples':     emg.shape[0]
                }
            except Exception as e:
                warnings.warn(f"Failed to load {mat_path}: {e}")
                continue


def load_uci_physical_action(data_path, subjects=None):
    """
    Loader for UCI EMG Physical Action Data Set — v5.5 (improved filtering).

    Structure expected:
        data_path/
            sub1/
                Aggressive/txt/*.txt
                Normal/txt/*.txt
            sub2/ ...
    
    Each .txt file contains 8 columns (EMG only). Label is inferred from the
    parent directory name ('Aggressive' → 1, 'Normal' → 0). Subject ID is taken
    from the 'subX' folder.

    Parameters
    ----------
    data_path : str
        Root directory containing the 'subX' folders.
    subjects : list, optional
        Filter specific subject IDs. Accepts both 'S1' and 1.

    Yields
    ------
    emg : np.ndarray
        EMG data of shape (n_samples, 8).
    labels : np.ndarray
        Binary labels: 0 = normal, 1 = aggressive.
    metadata : dict
        Dataset metadata.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"UCI Physical Action path not found: {data_path}")

    # 1. Collect all .txt files recursively
    all_txt_files = []
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith('.txt'):
                all_txt_files.append(os.path.join(root, f))

    if not all_txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_path}")

    # 2. Group files by subject ID extracted from the path
    subject_files = {}
    for fpath in all_txt_files:
        fname = os.path.basename(fpath)
        if 'readme' in fname.lower():
            continue

        # Normalize path for splitting
        norm_path = os.path.normpath(fpath)
        parts = norm_path.split(os.sep)

        # Extract subject from folder named 'subX'
        subj_id = None
        for part in parts:
            if part.lower().startswith('sub') and part[3:].isdigit():
                subj_id = f"S{part[3:]}"
                break
        if subj_id is None:
            # Fallback: use the folder immediately above the file
            parent = os.path.basename(os.path.dirname(fpath))
            if parent.lower() == 'txt':
                # go one level up
                parent = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            subj_id = parent if parent else 'unknown'

        subject_files.setdefault(subj_id, []).append(fpath)

    # 3. Filter subjects if requested (case-insensitive, accepts 'S1' or 1)
    if subjects is not None:
        filtered = {}
        for subj, files in subject_files.items():
            # Extract numeric part for flexible matching
            match = re.search(r'\d+', subj)
            subj_num = match.group(0) if match else subj
            # Check if any requested subject matches either the full ID or the number
            if any(str(s) == subj or str(s) == subj_num for s in subjects):
                filtered[subj] = files
        subject_files = filtered

        # Warn if a requested subject was not found
        if not subject_files:
            warnings.warn(f"No data found for requested subject(s): {subjects}")
            return

    # 4. Process each subject
    for subj_id, files in sorted(subject_files.items()):
        emg_segments = []
        labels_segments = []
        for fpath in sorted(files):
            # Determine label from path
            norm_path = os.path.normpath(fpath).lower()
            if 'aggressive' in norm_path:
                label_val = 1
            elif 'normal' in norm_path:
                label_val = 0
            else:
                # Fallback: check filename
                fname_lower = os.path.basename(fpath).lower()
                if 'aggressive' in fname_lower:
                    label_val = 1
                elif 'normal' in fname_lower:
                    label_val = 0
                else:
                    warnings.warn(f"Cannot determine label for {fpath}, skipping.")
                    continue

            # Read EMG data (8 columns expected)
            try:
                data = np.loadtxt(fpath, delimiter=None)
            except Exception:
                try:
                    df = pd.read_csv(fpath, header=None, sep=None, engine='python')
                    data = df.values.astype(np.float64)
                except Exception as e2:
                    warnings.warn(f"Failed to read {fpath}: {e2}")
                    continue

            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.shape[1] != 8:
                warnings.warn(f"File {fpath} has {data.shape[1]} columns (expected 8), skipping.")
                continue

            emg = data.astype(np.float64)
            label_col = np.full(emg.shape[0], label_val, dtype=int)

            emg_segments.append(emg)
            labels_segments.append(label_col)

        if not emg_segments:
            warnings.warn(f"No valid data for subject {subj_id}")
            continue

        emg_all = np.vstack(emg_segments)
        labels_all = np.hstack(labels_segments)

        metadata = {
            'dataset': 'UCI_Physical_Action',
            'subject': subj_id,
            'sampling_rate': 1000,
            'n_channels': emg_all.shape[1],
            'n_samples': emg_all.shape[0],
            'files_processed': len(files)
        }
        yield emg_all, labels_all, metadata


def explore_uci_physical(data_path):
    """
    Diagnostic: list subjects and label distribution for UCI Physical Action dataset.
    """
    import os
    import re

    all_files = []
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith('.txt'):
                all_files.append(os.path.join(root, f))

    subjects = {}
    for fpath in all_files:
        fname = os.path.basename(fpath)
        if 'readme' in fname.lower():
            continue

        norm_path = os.path.normpath(fpath)
        parts = norm_path.split(os.sep)

        # Subject
        subj_id = None
        for part in parts:
            if part.lower().startswith('sub') and part[3:].isdigit():
                subj_id = f"S{part[3:]}"
                break
        if subj_id is None:
            parent = os.path.basename(os.path.dirname(fpath))
            if parent.lower() == 'txt':
                parent = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            subj_id = parent if parent else 'unknown'

        # Label
        norm_lower = norm_path.lower()
        if 'aggressive' in norm_lower:
            label = 'Aggressive'
        elif 'normal' in norm_lower:
            label = 'Normal'
        else:
            label = 'Unknown'

        subjects.setdefault(subj_id, {'total': 0, 'aggressive': 0, 'normal': 0, 'unknown': 0})
        subjects[subj_id]['total'] += 1
        if label == 'Aggressive':
            subjects[subj_id]['aggressive'] += 1
        elif label == 'Normal':
            subjects[subj_id]['normal'] += 1
        else:
            subjects[subj_id]['unknown'] += 1

    print(f"Found {len(all_files)} .txt files.")
    for subj, counts in sorted(subjects.items()):
        print(f"{subj}: total={counts['total']} (Aggressive={counts['aggressive']}, Normal={counts['normal']}, Unknown={counts['unknown']})")       