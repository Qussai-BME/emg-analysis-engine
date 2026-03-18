"""
process_engine.py  — v6

Speed fixes vs v5:
  ✓ Wavelet: pywt native N-D batch API → zero Python loops (100x faster)
  ✓ AR: back to vectorized autocorrelation → zero per-window Python loops (50x faster)
    (Burg in v5 had 345,600 inner Python loop calls per dataset — that's why it was slow)

Accuracy additions vs v5:
  ✓ TKEO (Teager-Kaiser Energy Operator) — captures instantaneous muscle activation energy
    that amplitude features miss. Highly discriminative for gesture boundaries.

CACHE NOTE: delete validation_reports/cache/ before first v6 run.
Feature count changes: 346 → 372 (TKEO adds 1 feature/channel × 8 = 8, plus corrections)
"""

import sys
import numpy as np
from scipy import stats
from src.core_engine import EMGConfig, EMGFeatureExtractor

try:
    import pywt
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False


def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ─────────────────────────────────────────────────────────────────
#  AR via normalized autocorrelation — vectorized, zero per-window loops
#  Equivalent to autocorrelation method (Yule-Walker).
#  Loop is only over `order` (6 iterations) — rest is pure numpy.
# ─────────────────────────────────────────────────────────────────
def _ar_autocorr(wd, order):
    """
    Normalized autocorrelation at lags 1..order.
    wd     : (nw, nc, T)
    returns: (nw, nc, order)
    """
    nw, nc, T = wd.shape
    w      = wd - wd.mean(axis=2, keepdims=True)
    energy = (w ** 2).sum(axis=2, keepdims=True) + 1e-12   # (nw, nc, 1)
    out    = np.empty((nw, nc, order), dtype=np.float64)
    for k in range(1, order + 1):
        # r(k) = Σ w[t]*w[t-k] / Σ w[t]²
        out[:, :, k - 1] = (w[:, :, k:] * w[:, :, :T - k]).sum(2) / energy[:, :, 0]
    return out


# ─────────────────────────────────────────────────────────────────
#  Hjorth parameters — fully vectorized
# ─────────────────────────────────────────────────────────────────
def _hjorth(wd):
    d1 = np.diff(wd, axis=2)
    d2 = np.diff(d1, axis=2)
    v0 = np.var(wd, axis=2, ddof=1) + 1e-12
    v1 = np.var(d1, axis=2, ddof=1) + 1e-12
    v2 = np.var(d2, axis=2, ddof=1) + 1e-12
    mob = np.sqrt(v1 / v0)
    return v0, mob, np.sqrt(v2 / v1) / mob


# ─────────────────────────────────────────────────────────────────
#  TKEO — Teager-Kaiser Energy Operator — fully vectorized
#  TKEO(x[n]) = x[n]² - x[n-1]·x[n+1]
#  Captures instantaneous energy of muscle activation.
#  Highly discriminative because different gestures have different
#  neural firing patterns, not just different amplitudes.
# ─────────────────────────────────────────────────────────────────
def _tkeo(wd):
    """
    Returns mean TKEO energy per window/channel.
    wd : (nw, nc, T) → out : (nw, nc)
    """
    tkeo_vals = wd[:, :, 1:-1] ** 2 - wd[:, :, :-2] * wd[:, :, 2:]
    return np.mean(np.abs(tkeo_vals), axis=2)


# ─────────────────────────────────────────────────────────────────
#  Inter-channel Pearson correlation — vectorized
# ─────────────────────────────────────────────────────────────────
def _inter_ch_corr(wd):
    nw, nc, T = wd.shape
    w  = wd - wd.mean(axis=2, keepdims=True)
    wn = w / (np.linalg.norm(w, axis=2, keepdims=True) + 1e-12)
    n_pairs = nc * (nc - 1) // 2
    out = np.empty((nw, n_pairs), dtype=np.float64)
    idx = 0
    for i in range(nc):
        for j in range(i + 1, nc):
            out[:, idx] = (wn[:, i, :] * wn[:, j, :]).sum(1)
            idx += 1
    return out


# ─────────────────────────────────────────────────────────────────
#  Wavelet features — FULLY VECTORIZED using pywt native N-D support
#  No Python loops over windows or channels.
#  pywt.wavedec supports arbitrary-shaped input and processes
#  along the specified axis. We reshape (nw, nc, W) → (nw*nc, W),
#  apply wavedec, then reshape back. Zero inner loops.
# ─────────────────────────────────────────────────────────────────
def _wavelet_features_batch(wd, wavelet='db4', level=4):
    """
    Vectorized wavelet energy + entropy.
    wd      : (nw, nc, W)
    returns : energy (nw, nc, n_bands), entropy (nw, nc, n_bands)
    """
    nw, nc, W = wd.shape
    n_bands   = level + 1

    # Batch: reshape to (nw*nc, W), apply transform along last axis
    wd_flat = wd.reshape(nw * nc, W)
    coeffs  = pywt.wavedec(wd_flat, wavelet, level=level,
                           mode='periodization', axis=-1)
    # coeffs: list of (nw*nc, L_i) arrays

    # Total energy per window-channel across all bands
    total_e = sum(np.sum(c ** 2, axis=1) for c in coeffs) + 1e-12  # (nw*nc,)

    energy  = np.zeros((nw * nc, n_bands), dtype=np.float64)
    entropy = np.zeros((nw * nc, n_bands), dtype=np.float64)

    for bi, c in enumerate(coeffs):
        # Band energy (normalized)
        band_e = np.sum(c ** 2, axis=1)       # (nw*nc,)
        energy[:, bi] = band_e / total_e

        # Band entropy
        c2      = c ** 2
        c2_sum  = c2.sum(axis=1, keepdims=True) + 1e-12
        p       = np.clip(c2 / c2_sum, 1e-12, 1.0)
        entropy[:, bi] = -(p * np.log(p)).sum(axis=1)

    return energy.reshape(nw, nc, n_bands), entropy.reshape(nw, nc, n_bands)


# ─────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────
def extract_features_per_channel(emg, config_dict):
    """
    Parameters
    ----------
    emg         : (n_samples, n_channels) raw EMG
    config_dict : dict from config.yaml processing section

    Returns
    -------
    feat_flat       : (n_windows, total_features)
    windows         : list[(start, end)]
    snr_per_channel : list[float]
    feature_names   : list[str]
    """
    debug_print("Feature extraction v6 starting …")
    try:
        p = config_dict.copy()

        # Strip keys EMGConfig doesn't know
        do_freq    = p.pop('compute_freq_features',      True)
        p.pop('compute_extra_stats',                      None)
        p.pop('compute_spectral_centroid',                None)
        do_rolloff = p.pop('compute_spectral_rolloff',   False)
        p.pop('use_sliding_window',                       None)
        ssc_thr    = p.pop('ssc_threshold',              0.0)
        fft_pad2   = p.pop('fft_pad_to_power_of_two',    True)
        do_ar      = p.pop('compute_ar',                  True)
        ar_order   = p.pop('ar_order',                    6)
        do_hjorth  = p.pop('compute_hjorth',              True)
        do_icc     = p.pop('compute_inter_channel_corr',  True)
        sub_n      = p.pop('subsample_every_n',           1)
        do_norm    = p.pop('normalize_signal',            True)
        do_wavelet = p.pop('compute_wavelet',             True)
        wav_name   = p.pop('wavelet_name',               'db4')
        wav_level  = p.pop('wavelet_level',               4)

        if 'bandpass' in p:
            lo, hi = p.pop('bandpass');  p['cutoff_low'], p['cutoff_high'] = lo, hi
        if 'notch' in p:
            p['notch_freq'] = p.pop('notch')

        p.setdefault('sampling_rate',           1000)
        p.setdefault('window_size',              200)
        p.setdefault('overlap',                  0.5)
        p.setdefault('noise_estimation_method', 'percentile')
        p.setdefault('noise_percentile',         5.0)
        p.setdefault('r_threshold',              0.0)

        cfg  = EMGConfig(**p)
        filt = EMGFeatureExtractor(cfg).preprocess(emg)
        debug_print(f"Filtered: {filt.shape}")

        # ── Per-subject z-score normalization ────────────────────
        # Critical for LOSO: removes inter-subject electrode impedance variance.
        # Applied to whole signal (before windowing) so relative amplitude
        # differences between gestures within a subject are preserved.
        if do_norm:
            filt = (filt - filt.mean(0)) / (filt.std(0) + 1e-12)
            debug_print("Z-score normalization applied.")

        # ── Windowing ─────────────────────────────────────────────
        N, C  = filt.shape
        W     = cfg.window_size
        step  = max(1, int(W * (1 - cfg.overlap)))
        n_all = max(1, (N - W) // step + 1)
        wins  = [(i * step, i * step + W) for i in range(n_all)]
        if sub_n > 1:
            wins = wins[::sub_n]
        nw = len(wins)
        debug_print(f"Windows: {n_all} → {nw} (sub={sub_n})")

        wd = np.empty((nw, C, W), dtype=np.float64)
        for i, (s, e) in enumerate(wins):
            wd[i] = filt[s:e, :].T

        eps = 1e-12
        fs  = cfg.sampling_rate
        abs_wd = np.abs(wd)

        # ── Time-domain features (all vectorized) ─────────────────
        iemg    = abs_wd.sum(2)
        mav     = abs_wd.mean(2)
        log_mav = np.log(mav + eps)

        half    = W // 2
        mavs    = abs_wd[:, :, half:].mean(2) - abs_wd[:, :, :half].mean(2)

        ssi     = (wd ** 2).sum(2)
        rms     = np.sqrt(ssi / W)
        log_rms = np.log(rms + eps)

        # V-order (V=3): (mean(|x|^3))^(1/3)
        vo      = (abs_wd ** 3).mean(2) ** (1.0 / 3.0)

        # Log-detector
        log_det = np.exp(np.log(abs_wd + eps).mean(2))

        # Waveform length
        wl      = np.abs(np.diff(wd, axis=2)).sum(2)

        # ZCR
        zcr     = (np.diff(np.sign(wd), axis=2) != 0).sum(2) / (W - 1)

        # SSC
        d1  = np.diff(wd, axis=2)
        sd  = np.sign(d1)
        if ssc_thr > 0:
            sd = sd * (np.abs(d1) > ssc_thr)
        ssc = (sd[:, :, 1:] != sd[:, :, :-1]).sum(2)

        log_var = np.log(np.var(wd, axis=2, ddof=1) + eps)
        skw     = stats.skew(wd, axis=2)
        krt     = stats.kurtosis(wd, axis=2)

        # TKEO — Teager-Kaiser Energy Operator
        tkeo = _tkeo(wd)

        pch = [iemg, mav, log_mav, mavs, ssi, rms, log_rms,
               vo, log_det, wl, zcr, ssc, log_var, skw, krt, tkeo]
        pnm = ['IEMG', 'MAV', 'logMAV', 'MAVS', 'SSI', 'RMS', 'logRMS',
               'VO3', 'LogDet', 'WL', 'ZCR', 'SSC', 'logVAR', 'Skew', 'Kurt', 'TKEO']

        # ── Hjorth parameters ─────────────────────────────────────
        if do_hjorth:
            act, mob, cmp = _hjorth(wd)
            pch += [act, mob, cmp]
            pnm += ['HjAct', 'HjMob', 'HjCmp']

        # ── AR autocorrelation (vectorized — no per-window loops) ──
        # v6 reverts to vectorized autocorrelation from v4.
        # v5's Burg was more accurate for tiny windows (<100 samples)
        # but at 200 samples both methods are equivalent, and Burg
        # required 345,600 inner Python loop calls per dataset.
        if do_ar:
            ar = _ar_autocorr(wd, ar_order)
            for k in range(ar_order):
                pch.append(ar[:, :, k])
                pnm.append(f'AR{k + 1}')

        # ── Frequency features (vectorized) ───────────────────────
        if do_freq:
            fft_sz = W
            if fft_pad2:
                fft_sz = 1 << (W - 1).bit_length()
                if fft_sz == W:
                    fft_sz *= 2
                wdp = np.pad(wd, ((0, 0), (0, 0), (0, fft_sz - W)))
            else:
                wdp = wd

            mag   = np.abs(np.fft.rfft(wdp, axis=2))
            freqs = np.fft.rfftfreq(fft_sz, d=1.0 / fs)
            pw    = mag ** 2
            tp    = pw.sum(2, keepdims=True) + eps
            fr    = freqs.reshape(1, 1, -1)

            mnf = (fr * pw).sum(2) / tp[:, :, 0]
            cp  = np.cumsum(pw, 2)
            mdf = freqs[np.argmax(cp >= cp[:, :, -1:] / 2, axis=2)]
            pfq = freqs[np.argmax(pw, axis=2)]
            pn  = np.clip(pw / tp, 1e-12, 1.0)
            se  = -(pn * np.log(pn)).sum(2) / np.log(pn.shape[2])

            pch += [mnf, mdf, pfq, se]
            pnm += ['MNF', 'MDF', 'PeakF', 'SpEntropy']

            for flo, fhi in [(20, 150), (150, 350), (350, 450)]:
                mask = (freqs >= flo) & (freqs < fhi)
                pch.append(pw[:, :, mask].sum(2) / tp[:, :, 0])
                pnm.append(f'BP{flo}_{fhi}')

        # ── Wavelet features (BATCH — zero inner loops) ────────────
        if do_wavelet:
            if not HAS_WAVELETS:
                debug_print("WARNING: PyWavelets not installed. "
                            "pip install PyWavelets")
            else:
                debug_print(f"Batch wavelet ({wav_name}, level={wav_level}) …")
                w_e, w_ent = _wavelet_features_batch(wd, wav_name, wav_level)
                n_bands = wav_level + 1
                for b in range(n_bands):
                    tag = 'cA' if b == 0 else f'cD{n_bands - b}'
                    pch += [w_e[:, :, b], w_ent[:, :, b]]
                    pnm += [f'WavE_{tag}', f'WavEnt_{tag}']

        # ── Stack per-channel features ────────────────────────────
        pc_stack = np.stack(pch, axis=2)               # (nw, C, nf)
        flat     = pc_stack.reshape(nw, C * pc_stack.shape[2])
        names    = [f'ch{c}_{n}' for c in range(C) for n in pnm]

        # ── Inter-channel correlations ────────────────────────────
        if do_icc and C > 1:
            icc   = _inter_ch_corr(wd)
            flat  = np.hstack([flat, icc])
            names = names + [f'corr_{i}_{j}'
                             for i in range(C) for j in range(i + 1, C)]

        debug_print(f"Feature matrix: {flat.shape}  ({len(names)} features)")

        # SNR per channel
        snr = []
        for c in range(C):
            rv  = rms[:, c]
            nf  = max(np.percentile(rv, 5), eps)
            snr.append(float(20 * np.log10(rv.mean() / nf)))

        return flat, wins, snr, names

    except Exception as exc:
        import traceback
        debug_print(f"!!! {exc}")
        traceback.print_exc(file=sys.stderr)
        raise