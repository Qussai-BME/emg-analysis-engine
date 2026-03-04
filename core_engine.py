#!/usr/bin/env python3
"""
core_engine.py - EMG Signal Processing Core
Biomedical Engineering Module A: Integrated EMG-Analysis Engine
Author: Qussai Adlbi
Standards: IEEE Biomedical Signal Processing Guidelines | ISO 13485 Concepts
"""

import numpy as np
import logging
from scipy import signal
from scipy.fft import fft, fftfreq
import json
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging for production debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emg_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EMGConfig:
    """Configuration parameters following biomedical standards"""
    sampling_rate: int = 2000  # Hz (Nyquist > 400Hz for EMG)
    cutoff_low: float = 20.0    # Hz (remove DC drift)
    cutoff_high: float = 450.0  # Hz (EMG typical bandwidth)
    filter_order: int = 4        # 4th order Butterworth
    notch_freq: float = 50.0     # Hz (power line interference)
    window_size: int = 200        # samples (100ms at 2000Hz)
    overlap: float = 0.5          # 50% overlap for feature extraction
    
    def validate(self):
        """Input validation for configuration parameters"""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.cutoff_high >= self.sampling_rate / 2:
            raise ValueError("Cutoff high must be < Nyquist frequency")
        if self.filter_order < 1 or self.filter_order > 10:
            raise ValueError("Filter order should be between 1 and 10")
        return True

class EMGFeatureExtractor:
    """
    Core signal processing engine for EMG analysis.
    Implements gold-standard features for prosthetic control.
    """
    
    def __init__(self, config: EMGConfig):
        self.config = config
        self.config.validate()
        self.filters = self._design_filters()
        logger.info(f"EMG Engine initialized with config: {config}")
        
    def _design_filters(self) -> Dict:
        """Design Butterworth bandpass and notch filters"""
        nyquist = self.config.sampling_rate / 2
        
        # Bandpass filter coefficients (4th order Butterworth)
        b_band, a_band = signal.butter(
            self.config.filter_order,
            [self.config.cutoff_low / nyquist, 
             self.config.cutoff_high / nyquist],
            btype='band'
        )
        
        # Notch filter for power line interference (50Hz)
        b_notch, a_notch = signal.iirnotch(
            self.config.notch_freq / nyquist,
            30.0  # Quality factor
        )
        
        return {
            'bandpass': (b_band, a_band),
            'notch': (b_notch, a_notch)
        }
    
    def preprocess(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline with error handling.
        Steps: Bandpass filter -> Notch filter -> Artifact detection
        """
        try:
            # Input validation
            if len(raw_signal) < self.config.window_size:
                raise ValueError(f"Signal length {len(raw_signal)} < window size {self.config.window_size}")
            if np.isnan(raw_signal).any():
                raise ValueError("Signal contains NaN values")
            
            # Apply filters
            b_band, a_band = self.filters['bandpass']
            b_notch, a_notch = self.filters['notch']
            
            # filtfilt for zero-phase distortion
            filtered = signal.filtfilt(b_band, a_band, raw_signal)
            filtered = signal.filtfilt(b_notch, a_notch, filtered)
            
            # Detect motion artifacts (simple threshold)
            if np.max(np.abs(filtered)) > 10 * np.std(filtered):
                logger.warning("Possible motion artifact detected - signal may be saturated")
            
            return filtered
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def extract_features(self, signal_segment: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features for prosthetic control.
        Features: MAV, RMS, ZCR, WL, SSC
        """
        try:
            features = {}
            
            # Mean Absolute Value (MAV) - muscle activation level
            features['MAV'] = float(np.mean(np.abs(signal_segment)))
            
            # Root Mean Square (RMS) - signal power
            features['RMS'] = float(np.sqrt(np.mean(signal_segment**2)))
            
            # Zero Crossing Rate (ZCR) - frequency estimation
            zero_crossings = np.where(np.diff(np.signbit(signal_segment)))[0]
            features['ZCR'] = float(len(zero_crossings) / len(signal_segment))
            
            # Waveform Length (WL) - complexity measure
            features['WL'] = float(np.sum(np.abs(np.diff(signal_segment))))
            
            # Slope Sign Changes (SSC) - frequency content
            slopes = np.diff(signal_segment)
            ssc = np.sum((slopes[:-1] * slopes[1:]) < 0)
            features['SSC'] = float(ssc / len(signal_segment))
            
            # Additional metrics for quality assurance
            features['SNR_estimate'] = float(20 * np.log10(np.std(signal_segment) / 0.01))  # Assume 10uV noise floor
            features['signal_power'] = float(np.mean(signal_segment**2))
            
            logger.debug(f"Extracted features: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise
    
    def process_stream(self, raw_signal: np.ndarray) -> Dict:
        """
        Complete processing pipeline for streaming EMG data.
        Output formatted for future fusion with gait/surgical modules.
        """
        try:
            # Preprocess entire signal
            filtered = self.preprocess(raw_signal)
            
            # Sliding window feature extraction
            step = int(self.config.window_size * (1 - self.config.overlap))
            if step < 1:
                step = 1
            n_windows = (len(filtered) - self.config.window_size) // step + 1
            if n_windows < 1:
                n_windows = 1
                # take only one window at the beginning
                start = 0
                end = self.config.window_size
                windows_indices = [(start, end)]
            else:
                windows_indices = [(i * step, i * step + self.config.window_size) for i in range(n_windows)]
            
            features_over_time = []
            timestamps = []
            
            for start, end in windows_indices:
                window = filtered[start:end]
                features = self.extract_features(window)
                features_over_time.append(features)
                timestamps.append(start / self.config.sampling_rate)
            
            # Prepare JSON-compatible output for data fusion hub
            output = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'sampling_rate': self.config.sampling_rate,
                    'window_size': self.config.window_size,
                    'overlap': self.config.overlap,
                    'filter_config': asdict(self.config)
                },
                'signal_quality': {
                    'mean_snr': float(np.mean([f['SNR_estimate'] for f in features_over_time])),
                    'artifact_detected': False  # Add logic later
                },
                'time_series': {
                    'timestamps': timestamps,
                    'features': features_over_time
                },
                'summary_statistics': {
                    'mean_activation': float(np.mean([f['MAV'] for f in features_over_time])),
                    'peak_activation': float(np.max([f['RMS'] for f in features_over_time])),
                    'fatigue_index': self._compute_fatigue_index(features_over_time)
                }
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Stream processing failed: {str(e)}")
            raise
    
    def _compute_fatigue_index(self, features_over_time: list) -> float:
        """Compute muscle fatigue index using median frequency shift"""
        # Simplified version - will be enhanced in Module B
        if len(features_over_time) < 10:
            return 0.0
        
        # Fatigue typically shows as decrease in MAV and increase in ZCR
        mav_trend = np.polyfit(range(len(features_over_time[-10:])), 
                               [f['MAV'] for f in features_over_time[-10:]], 1)[0]
        return float(-mav_trend)  # Positive = fatigue

class EMGSignalSimulator:
    """Generate synthetic EMG signals for testing and demonstration"""
    
    @staticmethod
    def generate_contraction(duration: float, sampling_rate: int, 
                            intensity: float = 1.0) -> np.ndarray:
        """
        Generate realistic EMG signal during muscle contraction.
        Based on Gaussian noise modulated by contraction profile.
        """
        n_samples = int(duration * sampling_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Contraction envelope (rise, sustain, fall)
        envelope = np.zeros(n_samples)
        rise = int(0.2 * n_samples)  # 20% rise time
        sustain = int(0.6 * n_samples)  # 60% sustain
        fall = n_samples - rise - sustain
        
        if rise > 0:
            envelope[:rise] = np.linspace(0, intensity, rise)
        if sustain > 0:
            envelope[rise:rise+sustain] = intensity
        if fall > 0:
            envelope[rise+sustain:] = np.linspace(intensity, 0, fall)
        
        # EMG is Gaussian noise shaped by envelope
        noise = np.random.normal(0, 0.1 * intensity, n_samples)
        signal = envelope * noise
        
        # Add realistic noise floor
        signal += np.random.normal(0, 0.01, n_samples)
        
        return signal
    
    @staticmethod
    def generate_cyclic(duration: float, sampling_rate: int,
                        cycle_freq: float = 1.0) -> np.ndarray:
        """Generate cyclic EMG pattern (e.g., for gait)"""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        # Sinusoidal activation pattern
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * cycle_freq * t)
        noise = np.random.normal(0, 0.1, len(t))
        return envelope * noise

# Unit test for self-validation
if __name__ == "__main__":
    logger.info("Running self-validation...")
    
    # Test configuration
    config = EMGConfig()
    engine = EMGFeatureExtractor(config)
    simulator = EMGSignalSimulator()
    
    # Generate test signal
    test_signal = simulator.generate_contraction(duration=2.0, 
                                                 sampling_rate=config.sampling_rate)
    
    # Process
    result = engine.process_stream(test_signal)
    
    # Validate output format
    assert 'metadata' in result, "Missing metadata"
    assert 'time_series' in result, "Missing time series data"
    assert 'summary_statistics' in result, "Missing summary statistics"
    
    # Save example output for documentation
    with open('example_output.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Self-validation passed. Example output saved.")
    logger.info(f"Mean activation: {result['summary_statistics']['mean_activation']:.3f}")