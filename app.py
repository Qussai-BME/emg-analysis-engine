#!/usr/bin/env python3
"""
app.py - EMG Analysis Dashboard
Simplified file upload - accepts any numeric data automatically
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import tempfile
import os

from core_engine import EMGFeatureExtractor, EMGConfig, EMGSignalSimulator

# ---------------------------
# Helper function to read any numeric file
# ---------------------------
def read_numeric_file(uploaded_file):
    """
    Attempt to read any file and extract a 1D numeric array.
    Supports: CSV (comma, tab, space separated), TXT, NPY.
    Returns: (signal_array, column_name, message)
    """
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_content = uploaded_file.getvalue()

    # For .npy files
    if file_ext == '.npy':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        try:
            arr = np.load(tmp_path)
            if arr.ndim == 1:
                return arr, "numpy array", f"Loaded 1D array with {len(arr)} samples."
            elif arr.ndim == 2 and arr.shape[1] == 1:
                return arr.flatten(), "numpy array", f"Loaded 2D array with {arr.shape[0]} samples (single column)."
            elif arr.ndim == 2:
                # multi-column: take first column by default
                return arr[:, 0], f"column 0", f"Loaded multi-column array, using first column ({arr.shape[0]} samples)."
            else:
                raise ValueError("Unsupported array dimensions")
        finally:
            os.unlink(tmp_path)

    # For text files (CSV, TXT) - try multiple parsing methods
    # First, decode content to string
    try:
        text = file_content.decode('utf-8')
    except UnicodeDecodeError:
        # Try latin-1 if utf-8 fails
        text = file_content.decode('latin-1')

    # Split into lines and strip
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("File is empty")

    # Try different strategies:

    # Strategy 1: treat as one column (every line is a number)
    try:
        arr = np.loadtxt(lines)  # loadtxt can handle list of strings
        if arr.ndim == 0:  # single number
            arr = np.array([arr])
        return arr, "single column", f"Loaded {len(arr)} samples as single column."
    except:
        pass

    # Strategy 2: try pandas with auto separator
    try:
        # Use pandas to read with separator detection
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None)
        # Drop columns that are completely non-numeric
        numeric_cols = []
        for col in df.columns:
            # Try converting to numeric, coerce errors to NaN
            s = pd.to_numeric(df[col], errors='coerce')
            if s.notna().any():
                numeric_cols.append((col, s))
        if numeric_cols:
            # Take the first numeric column
            col_idx, s = numeric_cols[0]
            arr = s.dropna().values
            return arr, f"column {col_idx+1}", f"Loaded {len(arr)} samples from column {col_idx+1}."
    except:
        pass

    # Strategy 3: try numpy genfromtxt (very forgiving)
    try:
        arr = np.genfromtxt(uploaded_file, invalid_raise=False)
        if arr.ndim == 1:
            return arr, "auto-detected", f"Loaded {len(arr)} samples."
        elif arr.ndim == 2:
            # Use first column if multiple
            return arr[:, 0], "first column", f"Loaded multi-column, using first column ({arr.shape[0]} samples)."
    except:
        pass

    raise ValueError("Could not parse file as numeric data. Please ensure it contains numbers.")

# Page configuration
st.set_page_config(
    page_title="EMG Analysis Engine | Biomedical Module A",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 700; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem; }
    .info-box { background: #F3F4F6; padding: 1rem; border-radius: 10px; border-left: 4px solid #3B82F6; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'simulator' not in st.session_state:
    st.session_state.simulator = EMGSignalSimulator()
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'uploaded_signal' not in st.session_state:
    st.session_state.uploaded_signal = None
if 'uploaded_info' not in st.session_state:
    st.session_state.uploaded_info = None

# Header
st.markdown('<p class="main-header">🧬 Integrated EMG-Analysis Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Module A of the Biomedical AI Ecosystem | IEEE Standards | ISO 13485 Concepts</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/biotech.png", width=80)
    st.title("⚙️ Configuration")

    st.markdown("### Signal Source")
    source_option = st.radio("Choose input source:", ["Simulation", "Upload File"], index=0)

    uploaded_file = None
    if source_option == "Upload File":
        st.markdown("#### Upload EMG data")
        uploaded_file = st.file_uploader(
            "Supported formats: CSV, TXT, NPY (any numeric data)",
            type=['csv', 'txt', 'npy'],
            help="The system will automatically extract numeric columns."
        )

        if uploaded_file is not None:
            try:
                # Read file using our robust function
                signal, col_info, msg = read_numeric_file(uploaded_file)
                st.session_state.uploaded_signal = signal
                st.session_state.uploaded_info = f"✅ {msg} (using {col_info})"
                st.success(st.session_state.uploaded_info)
                # Show a small preview
                preview = signal[:min(10, len(signal))]
                st.markdown(f"**Preview:** {preview}")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.session_state.uploaded_signal = None

    st.markdown("### Signal Parameters")
    sampling_rate = st.slider("Sampling Rate (Hz)", 500, 4000, 2000, 100)
    duration = st.slider("Duration (s) [for simulation]", 1.0, 10.0, 3.0, 0.5)
    intensity = st.slider("Intensity", 0.1, 2.0, 1.0, 0.1)

    st.markdown("### Filter Settings")
    cutoff_low = st.slider("High-pass (Hz)", 5.0, 50.0, 20.0, 5.0)
    cutoff_high = st.slider("Low-pass (Hz)", 200.0, 500.0, 450.0, 10.0)
    filter_order = st.slider("Filter Order", 2, 8, 4, 1)

    st.markdown("### Feature Extraction")
    window_size_ms = st.slider("Window (ms)", 50, 300, 100, 10)
    overlap = st.slider("Overlap", 0.0, 0.9, 0.5, 0.1)

    if st.button("🚀 Initialize Engine", type="primary", use_container_width=True):
        config = EMGConfig(
            sampling_rate=sampling_rate,
            cutoff_low=cutoff_low,
            cutoff_high=cutoff_high,
            filter_order=filter_order,
            window_size=int(window_size_ms * sampling_rate / 1000),
            overlap=overlap
        )
        st.session_state.engine = EMGFeatureExtractor(config)
        st.success("✅ Engine initialized!")

    st.markdown("---")
    if st.button("📊 Export JSON", use_container_width=True) and st.session_state.processing_history:
        latest = st.session_state.processing_history[-1]
        st.download_button(
            label="Download JSON",
            data=json.dumps(latest, indent=2),
            file_name=f"emg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Main metrics
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Sampling Rate", f"{sampling_rate} Hz")
with col2: st.metric("Bandwidth", f"{cutoff_low:.0f}-{cutoff_high:.0f} Hz")
with col3: st.metric("Window", f"{window_size_ms} ms", f"{overlap*100:.0f}% overlap")
with col4: st.metric("Filter Order", f"{filter_order}")

# Process button
if st.button("🎯 Generate & Analyze", type="primary", use_container_width=True):
    if st.session_state.engine is None:
        st.warning("⚠️ Please initialize the engine first.")
    else:
        with st.spinner("Processing..."):
            # Get signal
            if source_option == "Upload File" and st.session_state.uploaded_signal is not None:
                raw_signal = st.session_state.uploaded_signal * intensity
                actual_duration = len(raw_signal) / sampling_rate
                st.info(f"Using uploaded signal: {len(raw_signal)} samples, {actual_duration:.2f}s")
            else:
                if source_option == "Upload File" and st.session_state.uploaded_signal is None:
                    st.warning("Please upload a file first.")
                    st.stop()
                raw_signal = st.session_state.simulator.generate_contraction(
                    duration=duration,
                    sampling_rate=sampling_rate,
                    intensity=intensity
                )
                actual_duration = duration

            results = st.session_state.engine.process_stream(raw_signal)
            st.session_state.processing_history.append(results)

            # Create tabs (same as before, but ensure no Arabic text)
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Signal Analysis",
                "📊 Feature Extraction",
                "📉 Spectral Analysis",
                "📋 Technical Report"
            ])

            with tab1:
                # Time domain plots
                fig = make_subplots(rows=3, cols=1,
                                    subplot_titles=('Raw EMG Signal', 'Filtered Signal', 'Activation Envelope'),
                                    vertical_spacing=0.1)
                t_raw = np.linspace(0, actual_duration, len(raw_signal))
                fig.add_trace(go.Scatter(x=t_raw, y=raw_signal, mode='lines', name='Raw', line=dict(color='lightgray')), row=1, col=1)
                filtered = st.session_state.engine.preprocess(raw_signal)
                t_filtered = np.linspace(0, actual_duration, len(filtered))
                fig.add_trace(go.Scatter(x=t_filtered, y=filtered, mode='lines', name='Filtered', line=dict(color='#3B82F6')), row=2, col=1)
                # Envelope (RMS)
                window_samples = st.session_state.engine.config.window_size
                rms_values, t_rms = [], []
                for i in range(0, len(filtered) - window_samples, window_samples//2):
                    segment = filtered[i:i+window_samples]
                    rms_values.append(np.sqrt(np.mean(segment**2)))
                    t_rms.append(i / sampling_rate)
                fig.add_trace(go.Scatter(x=t_rms, y=rms_values, mode='lines', name='RMS Envelope', line=dict(color='#10B981', width=3)), row=3, col=1)
                fig.update_layout(height=600, showlegend=True)
                fig.update_xaxes(title_text="Time (s)", row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Features over time
                features_df = pd.DataFrame(results['time_series']['features'])
                timestamps = results['time_series']['timestamps']
                fig2 = make_subplots(rows=2, cols=2,
                                     subplot_titles=('MAV (Activation)', 'RMS (Power)',
                                                     'Zero Crossing Rate', 'Waveform Length'))
                fig2.add_trace(go.Scatter(x=timestamps, y=features_df['MAV'], mode='lines+markers', name='MAV', line=dict(color='#EF4444')), row=1, col=1)
                fig2.add_trace(go.Scatter(x=timestamps, y=features_df['RMS'], mode='lines+markers', name='RMS', line=dict(color='#3B82F6')), row=1, col=2)
                fig2.add_trace(go.Scatter(x=timestamps, y=features_df['ZCR'], mode='lines+markers', name='ZCR', line=dict(color='#10B981')), row=2, col=1)
                fig2.add_trace(go.Scatter(x=timestamps, y=features_df['WL'], mode='lines+markers', name='WL', line=dict(color='#F59E0B')), row=2, col=2)
                fig2.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="info-box"><b>📊 Summary Statistics</b><br>'
                                f'Mean Activation: {results["summary_statistics"]["mean_activation"]:.3f}<br>'
                                f'Peak Activation: {results["summary_statistics"]["peak_activation"]:.3f}<br>'
                                f'Fatigue Index: {results["summary_statistics"]["fatigue_index"]:.3f}</div>',
                                unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="info-box"><b>🔧 Filter Performance</b><br>'
                                f'Bandpass: {cutoff_low:.0f}-{cutoff_high:.0f} Hz<br>'
                                f'Notch: 50 Hz<br>'
                                f'SNR: {results["signal_quality"]["mean_snr"]:.1f} dB</div>',
                                unsafe_allow_html=True)

            with tab3:
                # Spectral analysis
                from scipy.fft import fft, fftfreq
                filtered = st.session_state.engine.preprocess(raw_signal)
                N = len(filtered)
                yf = fft(filtered)
                xf = fftfreq(N, 1/sampling_rate)[:N//2]
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=xf, y=2.0/N * np.abs(yf[:N//2]), mode='lines', fill='tozeroy', name='Power Spectrum', line=dict(color='#8B5CF6')))
                fig3.add_vrect(x0=20, x1=150, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Low freq")
                fig3.add_vrect(x0=150, x1=450, fillcolor="blue", opacity=0.2, line_width=0, annotation_text="EMG band")
                fig3.update_layout(title="Frequency Domain Analysis", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", height=400)
                st.plotly_chart(fig3, use_container_width=True)

            with tab4:
                # Technical report
                st.markdown("### 🔬 Technical Analysis Report")
                st.markdown(f"""
**Signal Quality Assessment:**
- SNR: {results['signal_quality']['mean_snr']:.1f} dB {'✅ Good' if results['signal_quality']['mean_snr'] > 20 else '⚠️ Acceptable'}
- Artifacts: {'Detected' if results['signal_quality']['artifact_detected'] else 'None detected'}

**Feature Analysis:**
- Mean Activation: {results['summary_statistics']['mean_activation']:.3f}
- Fatigue Index: {results['summary_statistics']['fatigue_index']:.3f}
- Peak Performance: {results['summary_statistics']['peak_activation']:.3f} at {timestamps[np.argmax([f['RMS'] for f in features_df.to_dict('records')])]:.2f}s

**Next Steps:** Integrate with gait analysis (Module B).
""")
                with st.expander("Raw JSON Output"):
                    st.json(results)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    Module A - Integrated EMG Analysis Engine | © 2026 Qussai Adlbi
</div>
""", unsafe_allow_html=True)