#!/usr/bin/env python3
"""
app.py - EMG Analysis Dashboard
Professional Streamlit interface for Module A
Optimized for performance and memory usage
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
import gc  # Garbage collector

from core_engine import EMGFeatureExtractor, EMGConfig, EMGSignalSimulator

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
    .warning-box { background: #FEF3C7; padding: 1rem; border-radius: 10px; border-left: 4px solid #F59E0B; }
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

    # File upload with size limit
    uploaded_file = None
    MAX_FILE_SIZE_MB = 10
    if source_option == "Upload File":
        st.markdown("#### Upload EMG data")
        uploaded_file = st.file_uploader(
            "Supported formats: CSV, TXT, NPY",
            type=['csv', 'txt', 'npy'],
            help="Files larger than 10MB may cause performance issues."
        )

        if uploaded_file is not None:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large! Maximum size is {MAX_FILE_SIZE_MB} MB. Your file is {file_size_mb:.2f} MB.")
                st.stop()
            else:
                st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")

    st.markdown("### Performance Settings")
    
    # Downsampling option
    use_downsampling = st.checkbox("Use downsampling for large files (recommended)", True,
                                   help="Reduces memory usage for files >20,000 samples")
    if use_downsampling:
        max_samples = st.number_input("Max samples to process", 
                                      min_value=1000, 
                                      max_value=50000, 
                                      value=20000, 
                                      step=1000,
                                      help="Signals longer than this will be downsampled")
    
    # Memory warning
    st.markdown("---")
    st.markdown('<div class="warning-box">⚠️ <b>Memory Notice:</b> Streamlit Cloud has 1GB RAM limit. Large files (>50,000 samples) may crash.</div>', 
                unsafe_allow_html=True)

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
        with st.spinner("Processing... (this may take a few seconds for large files)"):
            
            # Get signal
            if source_option == "Upload File" and uploaded_file is not None:
                try:
                    # Read file based on extension
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        # Find first numeric column
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) == 0:
                            st.error("No numeric columns found in CSV.")
                            st.stop()
                        raw_signal = df[numeric_cols[0]].values.astype(float)
                        st.info(f"Using column: {numeric_cols[0]}")
                    
                    elif uploaded_file.name.endswith('.txt'):
                        content = uploaded_file.read().decode('utf-8')
                        numbers = []
                        for line in content.split():
                            line = line.strip()
                            if line:
                                try:
                                    numbers.append(float(line))
                                except ValueError:
                                    continue
                        raw_signal = np.array(numbers)
                    
                    elif uploaded_file.name.endswith('.npy'):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        raw_signal = np.load(tmp_path)
                        os.unlink(tmp_path)
                        if raw_signal.ndim > 1:
                            raw_signal = raw_signal.flatten()
                    
                    else:
                        st.error("Unsupported file format")
                        st.stop()
                    
                    # Check minimum samples
                    min_absolute = 10
                    min_recommended = 200
                    
                    if len(raw_signal) < min_absolute:
                        st.error(f"Signal too short: {len(raw_signal)} samples. Minimum required: {min_absolute}")
                        st.stop()
                    elif len(raw_signal) < min_recommended:
                        st.warning(f"⚠️ Signal length ({len(raw_signal)} samples) is below recommended minimum ({min_recommended}). Results may be unstable.")
                    
                    # Apply downsampling if enabled
                    original_len = len(raw_signal)
                    if use_downsampling and len(raw_signal) > max_samples:
                        # Simple downsampling by taking every nth sample
                        step = len(raw_signal) // max_samples
                        if step > 1:
                            raw_signal = raw_signal[::step]
                            st.info(f"📉 Signal downsampled from {original_len} to {len(raw_signal)} samples for performance.")
                    
                    # Apply intensity
                    if intensity != 1.0:
                        raw_signal = raw_signal * intensity
                    
                    actual_duration = len(raw_signal) / sampling_rate
                    
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.stop()
            else:
                # Simulation
                if source_option == "Upload File" and uploaded_file is None:
                    st.warning("Please upload a file first.")
                    st.stop()
                raw_signal = st.session_state.simulator.generate_contraction(
                    duration=duration,
                    sampling_rate=sampling_rate,
                    intensity=intensity
                )
                actual_duration = duration

            # Process signal
            try:
                results = st.session_state.engine.process_stream(raw_signal)
                st.session_state.processing_history.append(results)
                
                # Clear large variables to free memory
                filtered = None
                gc.collect()
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Signal Analysis", 
                "📊 Feature Extraction", 
                "📉 Spectral Analysis",
                "📋 Technical Report"
            ])

            with tab1:
                # Time domain plots - use fewer points for display if signal is very large
                display_step = max(1, len(raw_signal) // 5000)  # Show max 5000 points
                
                fig = make_subplots(rows=3, cols=1,
                                    subplot_titles=('Raw EMG Signal', 'Filtered Signal', 'Activation Envelope'),
                                    vertical_spacing=0.1)
                
                # Raw signal (downsampled for display)
                t_raw = np.linspace(0, actual_duration, len(raw_signal))
                fig.add_trace(
                    go.Scatter(x=t_raw[::display_step], y=raw_signal[::display_step], 
                              mode='lines', name='Raw', line=dict(color='lightgray')),
                    row=1, col=1
                )
                
                # Filtered signal
                filtered = st.session_state.engine.preprocess(raw_signal)
                t_filtered = np.linspace(0, actual_duration, len(filtered))
                fig.add_trace(
                    go.Scatter(x=t_filtered[::display_step], y=filtered[::display_step], 
                              mode='lines', name='Filtered', line=dict(color='#3B82F6')),
                    row=2, col=1
                )
                
                # Envelope (RMS) - compute on full signal but show fewer points
                window_samples = st.session_state.engine.config.window_size
                step_env = max(1, window_samples // 2)
                rms_values = []
                t_rms = []
                for i in range(0, len(filtered) - window_samples, step_env):
                    segment = filtered[i:i+window_samples]
                    rms_values.append(np.sqrt(np.mean(segment**2)))
                    t_rms.append(i / sampling_rate)
                
                # Downsample envelope for display if needed
                env_display_step = max(1, len(rms_values) // 1000)
                fig.add_trace(
                    go.Scatter(x=t_rms[::env_display_step], y=rms_values[::env_display_step], 
                              mode='lines', name='RMS Envelope', line=dict(color='#10B981', width=3)),
                    row=3, col=1
                )
                
                fig.update_layout(height=600, showlegend=True)
                fig.update_xaxes(title_text="Time (s)", row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Features over time
                features_df = pd.DataFrame(results['time_series']['features'])
                timestamps = results['time_series']['timestamps']
                
                # Downsample for display if needed
                display_step_feat = max(1, len(timestamps) // 500)
                
                fig2 = make_subplots(rows=2, cols=2,
                                     subplot_titles=('MAV (Activation)', 'RMS (Power)',
                                                     'Zero Crossing Rate', 'Waveform Length'))
                
                fig2.add_trace(
                    go.Scatter(x=timestamps[::display_step_feat], 
                              y=features_df['MAV'][::display_step_feat], 
                              mode='lines+markers', name='MAV', line=dict(color='#EF4444')),
                    row=1, col=1
                )
                
                fig2.add_trace(
                    go.Scatter(x=timestamps[::display_step_feat], 
                              y=features_df['RMS'][::display_step_feat], 
                              mode='lines+markers', name='RMS', line=dict(color='#3B82F6')),
                    row=1, col=2
                )
                
                fig2.add_trace(
                    go.Scatter(x=timestamps[::display_step_feat], 
                              y=features_df['ZCR'][::display_step_feat], 
                              mode='lines+markers', name='ZCR', line=dict(color='#10B981')),
                    row=2, col=1
                )
                
                fig2.add_trace(
                    go.Scatter(x=timestamps[::display_step_feat], 
                              y=features_df['WL'][::display_step_feat], 
                              mode='lines+markers', name='WL', line=dict(color='#F59E0B')),
                    row=2, col=2
                )
                
                fig2.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

                # Feature statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="info-box">
                    <b>📊 Summary Statistics</b><br>
                    Mean Activation: {results['summary_statistics']['mean_activation']:.3f}<br>
                    Peak Activation: {results['summary_statistics']['peak_activation']:.3f}<br>
                    Fatigue Index: {results['summary_statistics']['fatigue_index']:.3f}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="info-box">
                    <b>🔧 Filter Performance</b><br>
                    Bandpass: {cutoff_low:.0f}-{cutoff_high:.0f} Hz<br>
                    Notch: 50 Hz<br>
                    SNR: {results['signal_quality']['mean_snr']:.1f} dB
                    </div>
                    """, unsafe_allow_html=True)

            with tab3:
                # Spectral analysis - use FFT on downsampled signal for performance
                from scipy.fft import fft, fftfreq
                
                # Use a subset for FFT if signal is very large
                fft_max_points = 10000
                if len(raw_signal) > fft_max_points:
                    fft_signal = raw_signal[::len(raw_signal)//fft_max_points]
                    fft_fs = sampling_rate / (len(raw_signal)//fft_max_points)
                else:
                    fft_signal = raw_signal
                    fft_fs = sampling_rate
                
                N = len(fft_signal)
                yf = fft(fft_signal)
                xf = fftfreq(N, 1/fft_fs)[:N//2]
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=xf, y=2.0/N * np.abs(yf[:N//2]),
                    mode='lines', fill='tozeroy',
                    name='Power Spectrum',
                    line=dict(color='#8B5CF6')
                ))
                
                fig3.add_vrect(x0=20, x1=150, fillcolor="green", opacity=0.2, 
                              line_width=0, annotation_text="Low freq")
                fig3.add_vrect(x0=150, x1=450, fillcolor="blue", opacity=0.2,
                              line_width=0, annotation_text="EMG band")
                
                fig3.update_layout(
                    title="Frequency Domain Analysis",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Magnitude",
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)

            with tab4:
                # Technical report
                st.markdown("### 🔬 Technical Analysis Report")
                
                # Clinical interpretation
                mean_mav = results['summary_statistics']['mean_activation']
                if mean_mav < 0.1:
                    activation_desc = "Very low (resting muscle)"
                elif mean_mav < 0.3:
                    activation_desc = "Low (light contraction)"
                elif mean_mav < 0.6:
                    activation_desc = "Moderate (normal contraction)"
                else:
                    activation_desc = "High (strong contraction)"
                
                snr = results['signal_quality']['mean_snr']
                if snr > 25:
                    snr_desc = "Excellent"
                elif snr > 20:
                    snr_desc = "Good"
                elif snr > 15:
                    snr_desc = "Acceptable"
                else:
                    snr_desc = "Poor"
                
                st.markdown(f"""
                **Clinical Interpretation:**
                - **Muscle Activity:** {activation_desc}
                - **Signal Quality:** {snr_desc} (SNR: {snr:.1f} dB)
                
                **Signal Characteristics:**
                - Duration: {actual_duration:.2f} seconds
                - Samples: {len(raw_signal)}
                - Mean Activation: {mean_mav:.4f}
                - Peak Activation: {results['summary_statistics']['peak_activation']:.4f}
                
                **Filter Settings Applied:**
                - Bandpass: {cutoff_low:.0f}-{cutoff_high:.0f} Hz
                - Notch: 50 Hz
                - Window: {window_size_ms} ms with {overlap*100:.0f}% overlap
                """)
                
                with st.expander("View Raw JSON Output"):
                    st.json(results)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    Module A - Integrated EMG Analysis Engine | © 2026 Qussai Adlbi<br>
    <small>Optimized for Streamlit Cloud • Max file size: 10MB • Downsampling enabled for large files</small>
</div>
""", unsafe_allow_html=True)