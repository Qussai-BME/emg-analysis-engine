# Sample EMG Data

This directory contains example EMG data files for testing and demonstration purposes.

## Files

- `sample_emg.csv` – A synthetic EMG signal with 1000 samples at 2000 Hz (0.5 seconds).  
  The signal is generated using Gaussian noise to simulate muscle activity.  
  Use this file to test the pipeline without needing your own data.

## Format

All data files are expected to be in one of the following formats:
- CSV: single column of amplitude values (no header or with header 'amplitude').
- TXT: one value per line.
- NPY: 1D numpy array.

For multi‑channel files, the engine automatically selects the first numeric column.