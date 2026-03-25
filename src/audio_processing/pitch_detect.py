"""
Pitch Detection and Tonic Identification
Detects fundamental frequency (F0) and identifies the tonic note
Uses CREPE and pYIN for accurate pitch contour including gamakas
"""

import numpy as np
import librosa
from scipy.signal import medfilt
from collections import Counter
from typing import Tuple, Optional, Dict

# Optional CREPE import - will use pYIN as fallback
try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False


class PitchDetector:
    """Detects pitch and tonic from audio with gamaka-aware algorithms"""
    
    def __init__(self, sample_rate: int = 22050, 
                 fmin: float = 80, 
                 fmax: float = 600,
                 method: str = 'pyin'):
        """
        Initialize pitch detector
        
        Args:
            sample_rate: Audio sample rate
            fmin: Minimum frequency to detect (Hz)
            fmax: Maximum frequency to detect (Hz)
            method: 'crepe', 'pyin', or 'piptrack'
        """
        self.sr = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.method = method
        
        # Validate method
        if method == 'crepe' and not CREPE_AVAILABLE:
            print("⚠️ CREPE not available, falling back to pYIN")
            self.method = 'pyin'
        
    def detect_pitch(self, audio: np.ndarray, 
                     hop_length: int = 512) -> np.ndarray:
        """
        Detect pitch (F0) using librosa's piptrack
        
        Args:
            audio: Audio signal
            hop_length: Number of samples between frames
            
        Returns:
            Array of pitch values (Hz) over time
        """
        # Use piptrack for pitch detection
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=hop_length
        )
        
        # Extract pitch with highest magnitude at each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_values.append(pitch if pitch > 0 else np.nan)
        
        return np.array(pitch_values)
    
    def detect_pitch_pyin(self, audio: np.ndarray,
                          hop_length: int = 512,
                          fill_na: Optional[float] = None) -> np.ndarray:
        """
        Detect pitch using pYIN algorithm (probabilistic YIN - robust for gamakas)
        pYIN is better than YIN for continuous pitch tracking with ornamentation
        
        Args:
            audio: Audio signal
            hop_length: Number of samples between frames
            fill_na: Value to fill for unvoiced frames (None keeps NaN)
            
        Returns:
            Array of pitch values (Hz) over time
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=hop_length,
            fill_na=fill_na
        )
        return f0
    
    def detect_pitch_crepe(self, audio: np.ndarray,
                          hop_length: int = 512,
                          model_capacity: str = 'full') -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect pitch using CREPE (CNN-based, state-of-the-art for gamakas)
        CREPE excels at tracking fast oscillations and continuous pitch
        
        Args:
            audio: Audio signal
            hop_length: Number of samples between frames
            model_capacity: 'tiny', 'small', 'medium', 'large', or 'full'
            
        Returns:
            Tuple of (pitch_values, confidence_scores)
        """
        if not CREPE_AVAILABLE:
            raise ImportError("CREPE not installed. Install with: pip install crepe")
        
        # CREPE expects time in seconds
        step_size = hop_length / self.sr * 1000  # Convert to milliseconds
        
        time, frequency, confidence, activation = crepe.predict(
            audio,
            self.sr,
            step_size=step_size,
            model_capacity=model_capacity,
            viterbi=True  # Smooths pitch contour
        )
        
        # Filter by frequency range and confidence
        frequency = np.where(
            (frequency >= self.fmin) & (frequency <= self.fmax) & (confidence > 0.5),
            frequency,
            np.nan
        )
        
        return frequency, confidence
    
    def detect_pitch_auto(self, audio: np.ndarray,
                         hop_length: int = 512) -> Dict[str, np.ndarray]:
        """
        Automatically detect pitch using the configured method
        
        Args:
            audio: Audio signal
            hop_length: Number of samples between frames
            
        Returns:
            Dictionary with 'pitch', 'confidence' (if available), and 'method'
        """
        if self.method == 'crepe' and CREPE_AVAILABLE:
            pitch, confidence = self.detect_pitch_crepe(audio, hop_length)
            return {
                'pitch': pitch,
                'confidence': confidence,
                'method': 'crepe'
            }
        elif self.method == 'pyin':
            pitch = self.detect_pitch_pyin(audio, hop_length)
            return {
                'pitch': pitch,
                'confidence': None,
                'method': 'pyin'
            }
        else:
            pitch = self.detect_pitch(audio, hop_length)
            return {
                'pitch': pitch,
                'confidence': None,
                'method': 'piptrack'
            }
    
    def detect_pitch_yin(self, audio: np.ndarray,
                         hop_length: int = 512) -> np.ndarray:
        """
        Detect pitch using YIN algorithm (legacy support)
        Note: pYIN is generally better - use detect_pitch_pyin() instead
        
        Args:
            audio: Audio signal
            hop_length: Number of samples between frames
            
        Returns:
            Array of pitch values (Hz) over time
        """
        f0 = librosa.yin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=hop_length
        )
        return f0
    
    def smooth_pitch(self, pitch_values: np.ndarray, 
                     kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filtering to smooth pitch contour
        
        Args:
            pitch_values: Raw pitch values
            kernel_size: Size of median filter kernel (must be odd)
            
        Returns:
            Smoothed pitch values
        """
        # Remove NaN values for filtering
        valid_mask = ~np.isnan(pitch_values)
        smoothed = pitch_values.copy()
        
        if valid_mask.sum() > kernel_size:
            valid_pitches = pitch_values[valid_mask]
            smoothed_valid = medfilt(valid_pitches, kernel_size=kernel_size)
            smoothed[valid_mask] = smoothed_valid
        
        return smoothed
    
    def estimate_tonic(self, pitch_values: np.ndarray,
                      method: str = 'histogram') -> float:
        """
        Estimate the tonic (Sa) frequency from pitch values
        
        Args:
            pitch_values: Array of detected pitches
            method: 'histogram' or 'median'
            
        Returns:
            Estimated tonic frequency (Hz)
        """
        # Remove invalid values
        valid_pitches = pitch_values[~np.isnan(pitch_values)]
        valid_pitches = valid_pitches[valid_pitches > 0]
        
        if len(valid_pitches) == 0:
            raise ValueError("No valid pitch values found")
        
        if method == 'median':
            # Simple median approach
            return np.median(valid_pitches)
        
        elif method == 'histogram':
            # Convert to cents for better binning
            cents = 1200 * np.log2(valid_pitches / self.fmin)
            
            # Create histogram
            hist, bin_edges = np.histogram(cents, bins=100)
            
            # Find peaks
            peak_idx = hist.argmax()
            peak_cents = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
            
            # Convert back to Hz
            tonic_hz = self.fmin * (2 ** (peak_cents / 1200))
            return tonic_hz
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def pitch_to_cents(self, pitch: float, tonic: float) -> float:
        """
        Convert pitch to cents relative to tonic
        
        Args:
            pitch: Frequency in Hz
            tonic: Tonic frequency in Hz
            
        Returns:
            Pitch in cents (1200 cents = 1 octave)
        """
        if pitch <= 0 or tonic <= 0:
            return np.nan
        return 1200 * np.log2(pitch / tonic)
    
    def cents_to_note(self, cents: float) -> Tuple[str, int]:
        """
        Convert cents to Carnatic note name
        
        Args:
            cents: Pitch in cents relative to tonic
            
        Returns:
            Tuple of (note_name, cents_deviation)
        """
        # Carnatic scale in cents (12 swaras)
        carnatic_scale = {
            'S': 0,      # Shadja
            'R1': 90,    # Shuddha Rishabha
            'R2': 204,   # Chatushruti Rishabha
            'G2': 294,   # Sadharana Gandhara
            'G3': 408,   # Antara Gandhara
            'M1': 498,   # Shuddha Madhyama
            'M2': 612,   # Prati Madhyama
            'P': 702,    # Panchama
            'D1': 792,   # Shuddha Dhaivata
            'D2': 906,   # Chatushruti Dhaivata
            'N2': 996,   # Kaisiki Nishada
            'N3': 1110   # Kakali Nishada
        }
        
        # Normalize to single octave
        cents_normalized = cents % 1200
        
        # Find closest note
        min_diff = float('inf')
        closest_note = 'S'
        
        for note, note_cents in carnatic_scale.items():
            diff = abs(cents_normalized - note_cents)
            if diff < min_diff:
                min_diff = diff
                closest_note = note
        
        deviation = cents_normalized - carnatic_scale[closest_note]
        return closest_note, int(deviation)
    
    def analyze_pitch(self, audio_path: str) -> dict:
        """
        Complete pitch analysis pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with pitch analysis results
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        # Detect pitch
        print("Detecting pitch...")
        pitch_values = self.detect_pitch_yin(audio)
        
        # Smooth pitch contour
        smoothed_pitch = self.smooth_pitch(pitch_values)
        
        # Estimate tonic
        print("Estimating tonic...")
        tonic = self.estimate_tonic(smoothed_pitch)
        
        # Convert to cents
        cents_values = np.array([
            self.pitch_to_cents(p, tonic) 
            for p in smoothed_pitch
        ])
        
        # Get note sequence
        notes = []
        for cents in cents_values:
            if not np.isnan(cents):
                note, deviation = self.cents_to_note(cents)
                notes.append(note)
        
        return {
            'tonic_hz': tonic,
            'pitch_contour': smoothed_pitch,
            'cents_contour': cents_values,
            'note_sequence': notes,
            'unique_notes': list(set([n for n in notes if n]))
        }


if __name__ == "__main__":
    # Example usage
    detector = PitchDetector()
    
    audio_path = "path/to/audio.mp3"
    try:
        results = detector.analyze_pitch(audio_path)
        print(f"Tonic: {results['tonic_hz']:.2f} Hz")
        print(f"Unique notes: {results['unique_notes']}")
    except Exception as e:
        print(f"Error: {e}")
