# EMG Analysis Engine — Module A
### A modular, clinically-informed foundation for EMG signal processing

**GitHub:** [Qussai-BME/emg-analysis-engine](https://github.com/Qussai-BME/emg-analysis-engine) · **Live App:** [streamlit.app](https://emg-analysis-engine-qussai-adlbi.streamlit.app/) · **License:** MIT

**Author:** Qussai Adlbi · [LinkedIn](https://www.linkedin.com/in/qussai-adlbi-99aa05385) · adlbiqussai@gmail.com
**Institution:** Al-Andalus University · Pázmány Péter Catholic University · **Python:** 3.10

**📄 Cite this work:**
[DOI](https://doi.org/10.5281/zenodo.18965272)
*Full paper available on Zenodo*

---

## Why this exists

Electromyography (EMG) signals are the electrical language of muscle contraction — used daily to diagnose ALS, myopathy, and carpal tunnel syndrome, and to control advanced prosthetics.

The tools available today are broken in two opposite directions. Proprietary clinical systems cost upward of $15,000, ship as sealed units, and offer zero access to the algorithm underneath. Open-source alternatives are lab-specific scripts: undocumented preprocessing decisions, filter cutoffs chosen by someone who left three years ago, results that cannot be reproduced six months later.

Same field. Two failure modes. One consequence: the path from EMG signal to clinical insight is longer, more expensive, and less trustworthy than it needs to be.

**This project** is Module A of a larger biomedical AI ecosystem. It is not a black box. Every filter, every feature, every parameter is chosen with clinical intent and documented transparently. The architecture is designed with **ISO 13485 quality system concepts** in mind — traceability, risk-aware design, and configuration management as first-class concerns — so that the distance between this codebase and a certifiable product is as short as possible.

---

## What it does

**Input**
Upload CSV, TXT, NPY, or EDF files, or explore instantly with the built-in synthetic demo. Multi-channel support with automatic time-column removal, interactive channel selection with Select All / Clear All, and live signal preview.

**Preprocessing**
Zero-phase 4th-order Butterworth bandpass filter (20–450 Hz) — the IEEE/ISEK clinical standard — combined with an adaptive 50 Hz notch filter. The notch applies only if a powerline peak is detected via PSD analysis; clean recordings are not distorted. Multiple filter types available: Butterworth, Chebyshev, Bessel, Elliptic.

**Quality gating**
SNR estimation with adaptive noise floor (percentile, median, or manual). Signals below 20 dB are flagged before feature extraction runs. No silent failures.

**Feature extraction**
MAV, RMS, ZCR, Waveform Length, SSC (time domain) + MDF, MNF (frequency domain), extracted via configurable sliding windows. The five time-domain features have the strongest evidence base for prosthetic control and clinical EMG assessment.

**Output**
Interactive Streamlit dashboard + standardized JSON. Five analysis tabs, statistical tools (descriptive stats, correlation matrix, PCA, fatigue index), PDF reports (detailed or simplified), and SQLite session logging.

**Validation suite**
A dedicated `validation/` module for Leave-One-Subject-Out (LOSO) cross-validation on public datasets (UCI Gesture, Ninapro DB7, CEMHSEY). Includes configurable feature pipelines, parallel processing, checkpointing, and automatic Markdown/HTML/JSON report generation. **86.92% accuracy on UCI Gesture** (36 subjects, 7 gestures, strict LOSO protocol — full details below).

---

## Architecture

```
emg-analysis-engine/
│
├── src/
│   ├── core_engine.py            # IEEE/ISEK-compliant filtering + feature extraction
│   ├── app.py                    # Streamlit dashboard (entry point)
│   ├── emg_stats.py              # Statistical tools (PCA, fatigue index, correlation)
│   ├── database.py               # SQLite session storage
│   ├── pdf_report.py             # PDF report generator
│   └── api.py                    # Optional REST API (FastAPI)
│
├── validation/
│   ├── config.yaml               # Full pipeline configuration
│   ├── validate_engine.py        # CLI entry point
│   ├── process_engine.py         # Advanced features: wavelet, AR, Hjorth, TKEO
│   ├── data_loaders.py           # UCI / Ninapro DB7 / CEMHSEY loaders
│   ├── metrics.py                # LOSO-CV + classifiers (RF, XGBoost, LDA)
│   ├── validate_engine.py        # Orchestration + parallel processing
│   ├── report_generator.py       # Markdown / HTML / JSON reports
│   └── checkpoint.py             # Resume-from-checkpoint utility
│
├── data/
│   ├── sample_emg.csv            # 3-second synthetic EMG for quick testing
│   ├── README.md                 # Data format documentation
│   └── emg+data+for+gestures/    # UCI Gesture dataset (36 subjects)
│       └── subject1/ … subject36/
│
├── docs/
│   └── images/                   # Screenshots + confusion matrix
│
├── requirements.txt
├── README.md
├── .gitignore                    # Excludes venv/, validation_reports/, __pycache__/
└── experiment_notes.md           # Lab notebook: "What surprised me today?"

# Generated at runtime (gitignored):
├── validation_reports/           # HTML/JSON reports + per-subject feature cache
├── __pycache__/
└── venv/
```

`src/` keeps all source code in one place. `validation/` provides a complete, reproducible pipeline for every reported result. `core_engine.py` is fully decoupled from the Streamlit interface — a separation that prevented a class of silent bugs where interface state altered computation. Every directory serves a purpose.

---

## Screenshots

### Main Dashboard
[Main Dashboard](docs/images/screenshot1.png)
*Interactive dashboard: signal visualization, control panel, and real-time analysis.*

### Feature Extraction and Spectral Analysis
[Feature Extraction](docs/images/screenshot2.png)
*Feature extraction (MAV, RMS, ZCR, WL) and frequency-domain analysis with EMG bandwidth highlighted.*

### Statistics Tab
[Statistics Tab](docs/images/screenshot3.png)
*Descriptive statistics per channel, correlation matrix, PCA, and fatigue index.*

---

## Get started in 2 minutes

**Prerequisites:** Python 3.9–3.11, pip

```bash
git clone https://github.com/Qussai-BME/emg-analysis-engine.git
cd emg-analysis-engine
pip install -r requirements.txt
streamlit run src/app.py
```

Open `http://localhost:8501`.

**Demo mode** — click "Simulation" in the sidebar to explore with synthetic EMG instantly.
**Your own data** — switch to "Upload File", select a CSV/TXT/NPY/EDF, choose channels, and analyse.

**Example console output:**
```
[INFO] Demo mode active – synthetic EMG generated
[INFO] Filter applied: Butterworth 4th-order, 20–450 Hz
[INFO] Features extracted: MAV=0.142, RMS=0.198, ZCR=87, WL=14.3, SSC=134
[INFO] SNR: 24.7 dB – Signal quality: ACCEPTABLE
[INFO] Output saved: results/example_output.json
```

---

## Validation results

### UCI Gesture dataset — 86.92% LOSO accuracy

**Dataset:** 36 subjects · 8 channels · 7 hand gestures · 1000 Hz
**Protocol:** Leave-One-Subject-Out cross-validation · no data leakage (PCA and scaler fitted on training fold only)
**Classifier:** XGBoost / Random Forest (configurable)

**Feature set (364 features total):**
time-domain (IEMG, MAV, logMAV, MAVS, SSI, RMS, logRMS, V-order, Log-detector, WL, ZCR, SSC, logVAR, Skewness, Kurtosis, TKEO) · Hjorth parameters (Activity, Mobility, Complexity) · AR autocorrelation coefficients (order 6) · wavelet energy and entropy (db4, 4 levels) · frequency-domain (MNF, MDF, peak frequency, spectral entropy, band powers 20–150 / 150–350 / 350–450 Hz) · inter-channel Pearson correlations (28 pairs)

| Metric | Value |
|--------|-------|
| Mean LOSO accuracy | **86.92%** |
| Std across subjects | ±14.65%* |
| Subjects | 36 |
| Gestures classified | 7 |
| Feature count | 364 |
| Processing time (36 subjects) | < 10 min (4 cores) |

*High standard deviation reflects genuine inter-subject variability in UCI Gesture: electrode placement and skin impedance differ significantly across 36 subjects recorded in uncontrolled conditions. This is a property of the dataset, not a modelling instability — per-subject accuracy breakdown is available in the full validation report.*

[Confusion Matrix](docs/images/UCI_Gesture_cm.png)

**Reproduce these results:**
```bash
pip install xgboost PyWavelets
python validation/validate_engine.py --datasets uci --config validation/config.yaml
```

**Validation on Ninapro DB1 and CEMHSEY** is ongoing and will be reported in subsequent releases.

---

## What surprised me — four hard lessons

This section exists because reproducibility is not just about code. It is about documenting the moments when assumptions break.

**1. The 50 Hz notch filter can be wrong.**
I initially applied it to every signal. Battery-powered recordings have no powerline interference — applying a notch where no peak exists distorts the signal. The engine now runs a PSD check first. The notch fires only when warranted. This is a one-line policy change that took real data to discover.

**2. Window size is a clinical decision, not a parameter.**
A 100 ms window reacts fast but produces jittery features. A 200 ms window stabilises estimates but loses temporal resolution. There is no universally correct number — it depends on whether you are controlling a prosthetic (prioritise speed) or diagnosing fatigue (prioritise stability). The right response is to expose the trade-off, not hide it behind a default.

**3. Data leakage makes results meaningless.**
Early classification tests showed 95% accuracy. Switching to Leave-One-Subject-Out cross-validation brought that number down substantially. The reason: random splits place parts of the same subject in both train and test sets. The model was memorising inter-session patterns, not learning gesture signatures. Subject-wise separation is now a non-negotiable constraint in every future module.

**4. Cloud deployment forces honest engineering.**
Streamlit Cloud provides 1 GB RAM. A 500,000-sample, 8-channel EMG file — together with filtered copies, feature arrays, and Plotly figures — exceeds that limit. The app crashed silently. The fix required smart downsampling for visualisation, file size checks on upload, and graceful degradation when memory headroom is low. Engineering for deployment constraints is as important as engineering for accuracy.

These four lessons are now encoded in the architecture. They are why the engine handles real data, not just synthetic examples.

---

## Roadmap

### Module A — complete
- [x] IEEE/ISEK-compliant preprocessing pipeline
- [x] Adaptive SNR quality gating
- [x] Five validated time-domain features + frequency features
- [x] Full validation suite with LOSO-CV and public datasets
- [x] Streamlit dashboard + PDF reports + SQLite logging
- [x] ISO 13485 quality system concepts applied throughout
- [ ] Automated performance benchmarking (speed, memory, accuracy)

### Module A2 — MyoControl Lite *(next)*
Gesture classification (6 hand movements) using Ninapro DB1. PSD-based adaptive notch, gold-standard features, LOSO SVM validation. Released as a separate Streamlit tab.

### Module B — Gait analysis
EMG + IMU fusion via complementary filter. Joint angle estimation, stance/swing phase detection, spatiotemporal gait parameters.

### Module C — Surgical robot interface
Real-time EMG → velocity control of a UR5 arm in PyBullet. Exponential moving average smoothing for natural motion. Gesture-to-command mapping for 6 hand postures.

### Module D — AI-enhanced control
Adaptive gain based on real-time fatigue estimation. 8–12 Hz tremor bandstop filter on the control signal. Maintains consistent prosthetic performance as muscle output degrades.

### Long-term — Data-Fusion Hub
Cloud-scalable platform aggregating EMG, gait kinematics, and robotic telemetry under a single versioned schema. Designed for longitudinal neurorehabilitation studies and personalised ML models.

---

## Limitations

**This project is:**
- ✅ A research-grade signal processing and classification tool
- ✅ An open, reproducible platform for EMG research
- ✅ A foundation developed with ISO 13485 quality management concepts (traceability, risk-aware design, configuration management)

**This project is not:**
- ❌ An FDA-approved or CE-marked medical device
- ❌ Clinically validated on patient populations
- ❌ Suitable for diagnostic or treatment decisions
- ❌ A replacement for regulated clinical EMG systems

Current validation uses open-source research datasets. Electrode placement variability, skin impedance differences, and inter-subject anatomical variation are not fully controlled. Real-world clinical noise environments differ from the recording conditions of these datasets.

Any clinical application requires IRB-approved trials, regulatory review (FDA 510(k) or CE marking), and validation on large, diverse patient datasets.

Transparency about what a system cannot do is not a weakness. In medical engineering, it is the only ethical baseline.

---

## Built on solid science

- De Luca, C.J. (1997). The use of surface electromyography in biomechanics. *Journal of Applied Biomechanics.*
- Phinyomark, A. et al. (2012). Feature reduction and selection for EMG signal classification. *Expert Systems with Applications.*
- Oskoei, M.A. & Hu, H. (2007). Myoelectric control systems — a survey. *Biomedical Signal Processing and Control.*
- IEEE/ISEK standards for surface EMG processing.
- PhysioNet, UCI Machine Learning Repository — open datasets used for validation.
- ISO 13485:2016 — Medical devices quality management systems (concepts applied in architecture).

---

## Cite this work

> Qussai Adlbi. (2026). *EMG Analysis Engine — Module A: An open-source, IEEE/ISEK-compliant platform for reproducible EMG signal processing and feature extraction.* Zenodo. https://doi.org/10.5281/zenodo.18965272

```bibtex
@software{adlbi2026emg,
  author    = {Qussai Adlbi},
  title     = {{EMG Analysis Engine — Module A}},
  month     = mar,
  year      = 2026,
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.18965272},
  url       = {https://doi.org/10.5281/zenodo.18965272}
}
```

---

## Collaboration and funding

Actively seeking:
- Research collaborators in biomedical engineering, neurology, and rehabilitation medicine
- Academic partners for clinical dataset access and IRB-approved validation studies
- Grant opportunities: NIH NIBIB, Wellcome Trust, EU Horizon Europe
- Institutional pilots with rehabilitation centres or prosthetics labs

If your institution works with EMG data and needs a reproducible, auditable pipeline — let's talk.

📧 adlbiqussai@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/qussai-adlbi-99aa05385)
🐙 [GitHub](https://github.com/Qussai-BME)
🏫 Al-Andalus University / Pázmány Péter Catholic University

---

## License

MIT — open for research use.
Commercial deployment and clinical use require separate agreements and regulatory compliance.

---

*Built at the intersection of signal processing, clinical need, and the stubborn belief that good science should be accessible.*
*This is not the final version. It is the right foundation.*