"""
Advanced Feature Extraction for Raga Detection
Implements mel-spectrogram, CQT, tonal histograms, and swara histograms
Captures phrase patterns, gamakas, and tonal distributions
"""

import numpy as np
import librosa
from typing import Dict, Tuple, Optional, List


# Carnatic music swara mapping (12 swaras in an octave)
SWARA_NAMES = ['S', 'R1', 'R2/G1', 'R3/G2', 'G3', 'M1', 'M2', 'P', 'D1', 'D2/N1', 'D3/N2', 'N3']
SWARA_CENTS = [0, 90, 112, 182, 204, 294, 316, 498, 588, 610, 680, 702, 792, 884, 996, 1088, 1200]


class AdvancedFeatureExtractor:
    """Extract ML-ready features optimized for raga identification"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize feature extractor
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sr = sample_rate
        
    def extract_mel_spectrogram(self, audio: np.ndarray,
                               n_mels: int = 128,
                               n_fft: int = 2048,
                               hop_length: int = 512,
                               fmax: int = 8000) -> np.ndarray:
        """
        Extract Mel spectrogram (time-frequency representation)
        Perfect for CNN input - captures timbral and temporal patterns
        
        Args:
            audio: Audio signal
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length
            fmax: Maximum frequency
            
        Returns:
            Mel spectrogram in dB (shape: [n_mels, time_frames])
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmax=fmax
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_cqt(self, audio: np.ndarray,
                   hop_length: int = 512,
                   n_bins: int = 84,
                   bins_per_octave: int = 12) -> np.ndarray:
        """
        Extract Constant-Q Transform (CQT)
        Better than Mel for pitch-based music - logarithmic frequency scale
        Aligns with musical notes (each bin = semitone)
        
        Args:
            audio: Audio signal
            hop_length: Hop length
            n_bins: Number of frequency bins
            bins_per_octave: Bins per octave (12 = semitone resolution)
            
        Returns:
            CQT spectrogram in dB (shape: [n_bins, time_frames])
        """
        cqt = np.abs(librosa.cqt(
            audio,
            sr=self.sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave
        ))
        
        # Convert to dB
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        return cqt_db
    
    def extract_chroma(self, audio: np.ndarray,
                      hop_length: int = 512) -> np.ndarray:
        """
        Extract chromagram (pitch class profile)
        Groups all pitches into 12 pitch classes (C, C#, D, ...)
        
        Args:
            audio: Audio signal
            hop_length: Hop length
            
        Returns:
            Chromagram (shape: [12, time_frames])
        """
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sr,
            hop_length=hop_length
        )
        return chroma
    
    def pitch_to_cents(self, pitch_hz: float, tonic_hz: float) -> float:
        """
        Convert pitch to cents relative to tonic
        
        Args:
            pitch_hz: Pitch frequency in Hz
            tonic_hz: Tonic frequency in Hz
            
        Returns:
            Cents above tonic (1200 cents = 1 octave)
        """
        if pitch_hz <= 0 or tonic_hz <= 0:
            return np.nan
        return 1200 * np.log2(pitch_hz / tonic_hz)
    
    def cents_to_swara(self, cents: float) -> Tuple[str, float]:
        """
        Map cents to nearest swara
        
        Args:
            cents: Cents relative to tonic (modulo 1200)
            
        Returns:
            Tuple of (swara_name, deviation_in_cents)
        """
        # Normalize to one octave
        cents_normalized = cents % 1200
        
        # Find nearest swara
        distances = np.abs(np.array(SWARA_CENTS) - cents_normalized)
        nearest_idx = np.argmin(distances)
        
        swara_name = SWARA_NAMES[min(nearest_idx, len(SWARA_NAMES) - 1)]
        deviation = distances[nearest_idx]
        
        return swara_name, deviation
    
    def extract_tonal_histogram(self, pitch_contour: np.ndarray,
                                tonic_hz: float,
                                n_bins: int = 120,
                                normalize: bool = True) -> np.ndarray:
        """
        Create tonal histogram (distribution of pitch classes)
        Shows which notes are used and how often
        
        Args:
            pitch_contour: Array of pitch values in Hz
            tonic_hz: Tonic frequency in Hz
            n_bins: Number of bins (120 = 10 cents resolution)
            normalize: Whether to normalize histogram
            
        Returns:
            Tonal histogram (shape: [n_bins])
        """
        # Remove NaN/zero values
        valid_pitches = pitch_contour[~np.isnan(pitch_contour) & (pitch_contour > 0)]
        
        if len(valid_pitches) == 0:
            return np.zeros(n_bins)
        
        # Convert to cents
        cents = np.array([self.pitch_to_cents(p, tonic_hz) for p in valid_pitches])
        cents = cents[~np.isnan(cents)]
        
        # Normalize to one octave (0-1200 cents)
        cents_normalized = cents % 1200
        
        # Create histogram
        hist, _ = np.histogram(cents_normalized, bins=n_bins, range=(0, 1200))
        
        if normalize and hist.sum() > 0:
            hist = hist.astype(float) / hist.sum()
        
        return hist
    
    def extract_swara_histogram(self, pitch_contour: np.ndarray,
                                tonic_hz: float,
                                tolerance_cents: float = 50) -> Dict[str, float]:
        """
        Create swara histogram (probability distribution of swaras)
        Key feature for distinguishing ragas with different note usage
        
        Args:
            pitch_contour: Array of pitch values in Hz
            tonic_hz: Tonic frequency in Hz
            tolerance_cents: Tolerance for swara assignment
            
        Returns:
            Dictionary mapping swara names to probabilities
        """
        # Remove NaN/zero values
        valid_pitches = pitch_contour[~np.isnan(pitch_contour) & (pitch_contour > 0)]
        
        if len(valid_pitches) == 0:
            return {swara: 0.0 for swara in SWARA_NAMES}
        
        # Convert to cents
        cents = np.array([self.pitch_to_cents(p, tonic_hz) for p in valid_pitches])
        cents = cents[~np.isnan(cents)]
        
        # Map to swaras
        swara_counts = {swara: 0 for swara in SWARA_NAMES}
        
        for c in cents:
            swara, deviation = self.cents_to_swara(c)
            if deviation <= tolerance_cents:
                swara_counts[swara] += 1
        
        # Normalize to probabilities
        total = sum(swara_counts.values())
        if total > 0:
            swara_probs = {k: v / total for k, v in swara_counts.items()}
        else:
            swara_probs = swara_counts
        
        return swara_probs
    
    def extract_pitch_statistics(self, pitch_contour: np.ndarray,
                                 tonic_hz: float) -> Dict[str, float]:
        """
        Extract statistical features from pitch contour
        Captures gamaka characteristics and phrase patterns
        
        Args:
            pitch_contour: Array of pitch values in Hz
            tonic_hz: Tonic frequency in Hz
            
        Returns:
            Dictionary of statistical features
        """
        valid_pitches = pitch_contour[~np.isnan(pitch_contour) & (pitch_contour > 0)]
        
        if len(valid_pitches) == 0:
            return {
                'mean_cents': 0,
                'std_cents': 0,
                'pitch_range_cents': 0,
                'vibrato_rate': 0,
                'pitch_change_rate': 0
            }
        
        # Convert to cents
        cents = np.array([self.pitch_to_cents(p, tonic_hz) for p in valid_pitches])
        cents = cents[~np.isnan(cents)]
        
        # Pitch derivatives (measure gamakas/oscillations)
        pitch_diff = np.diff(valid_pitches)
        pitch_change_rate = np.abs(pitch_diff).mean() if len(pitch_diff) > 0 else 0
        
        # Estimate vibrato rate (number of zero crossings in derivative)
        zero_crossings = np.sum(np.diff(np.sign(pitch_diff)) != 0) if len(pitch_diff) > 1 else 0
        vibrato_rate = zero_crossings / len(valid_pitches) if len(valid_pitches) > 0 else 0
        
        return {
            'mean_cents': float(np.mean(cents)),
            'std_cents': float(np.std(cents)),
            'pitch_range_cents': float(np.max(cents) - np.min(cents)),
            'vibrato_rate': float(vibrato_rate),
            'pitch_change_rate': float(pitch_change_rate)
        }
    
    def extract_all_features(self, audio: np.ndarray,
                            pitch_contour: Optional[np.ndarray] = None,
                            tonic_hz: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Extract all features for ML model
        
        Args:
            audio: Audio signal (preprocessed with HPS + bandpass)
            pitch_contour: Pre-computed pitch contour (optional)
            tonic_hz: Tonic frequency in Hz (optional)
            
        Returns:
            Dictionary containing all features
        """
        features = {}
        
        # Time-frequency features (for CNN)
        features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
        features['cqt'] = self.extract_cqt(audio)
        features['chroma'] = self.extract_chroma(audio)
        
        # Pitch-based features (if available)
        if pitch_contour is not None and tonic_hz is not None:
            features['tonal_histogram'] = self.extract_tonal_histogram(
                pitch_contour, tonic_hz
            )
            features['swara_histogram'] = self.extract_swara_histogram(
                pitch_contour, tonic_hz
            )
            features['pitch_statistics'] = self.extract_pitch_statistics(
                pitch_contour, tonic_hz
            )
        
        return features
    
    def prepare_for_cnn(self, mel_spec: np.ndarray,
                       target_shape: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        Prepare mel spectrogram for CNN input
        Resize and normalize to fixed shape
        
        Args:
            mel_spec: Mel spectrogram
            target_shape: Target shape (height, width)
            
        Returns:
            Resized and normalized spectrogram
        """
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (
            target_shape[0] / mel_spec.shape[0],
            target_shape[1] / mel_spec.shape[1]
        )
        
        # Resize
        resized = zoom(mel_spec, zoom_factors, order=1)
        
        # Normalize to [0, 1]
        normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        
        return normalized
