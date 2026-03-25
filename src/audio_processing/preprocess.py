"""
Audio Preprocessing Functions
Handles audio loading, cleaning, alignment, and normalization
Implements Harmonic-Percussive Separation (HPS) and bandpass filtering
for Carnatic music analysis
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional


class AudioPreprocessor:
    """Prepares audio data for analysis with Carnatic music optimizations"""
    
    def __init__(self, sample_rate: int = 22050, duration: Optional[float] = None):
        """
        Initialize preprocessor with target sample rate and duration
        
        Args:
            sample_rate: Target sample rate for audio (Hz)
            duration: Target duration in seconds (None = use full audio)
        """
        self.sr = sample_rate
        self.duration = duration
        
        # Carnatic music frequency range: 80-1800 Hz
        # Captures tanpura (100-150 Hz), voice/violin (200-2000 Hz)
        # Suppresses mridangam low/high frequencies
        self.lowcut = 80.0   # Hz
        self.highcut = 1800.0  # Hz
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with specified sample rate and duration
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            return y, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {e}")
    
    def pad_or_trim(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad with zeros or trim audio to target length
        
        Args:
            audio: Audio signal
            target_length: Target length in samples
            
        Returns:
            Audio with exact target length
        """
        if len(audio) < target_length:
            # Pad with zeros
            return np.pad(audio, (0, target_length - len(audio)))
        else:
            # Trim
            return audio[:target_length]
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Audio signal
            
        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    
    def remove_silence(self, audio: np.ndarray, 
                       top_db: int = 20) -> np.ndarray:
        """
        Remove leading and trailing silence
        
        Args:
            audio: Audio signal
            top_db: Threshold in dB below reference to consider as silence
            
        Returns:
            Audio with silence removed
        """
        # Find non-silent intervals
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        if len(intervals) > 0:
            # Concatenate all non-silent parts
            return np.concatenate([audio[start:end] for start, end in intervals])
        return audio
    
    def butter_bandpass(self, lowcut: float, highcut: float, 
                        order: int = 5) -> Tuple:
        """
        Design a Butterworth bandpass filter
        
        Args:
            lowcut: Low frequency cutoff (Hz)
            highcut: High frequency cutoff (Hz)
            order: Filter order
            
        Returns:
            Filter coefficients (b, a)
        """
        nyquist = 0.5 * self.sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def apply_bandpass_filter(self, audio: np.ndarray,
                             lowcut: Optional[float] = None,
                             highcut: Optional[float] = None) -> np.ndarray:
        """
        Apply bandpass filter to isolate melodic content
        Optimized for Carnatic music (80-1800 Hz)
        
        Args:
            audio: Audio signal
            lowcut: Low frequency cutoff (Hz), defaults to self.lowcut
            highcut: High frequency cutoff (Hz), defaults to self.highcut
            
        Returns:
            Filtered audio
        """
        lowcut = lowcut or self.lowcut
        highcut = highcut or self.highcut
        
        b, a = self.butter_bandpass(lowcut, highcut, order=5)
        filtered = filtfilt(b, a, audio)
        return filtered
    
    def harmonic_percussive_separation(self, audio: np.ndarray,
                                       margin: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate harmonic (melodic) and percussive components
        Critical for Carnatic music to isolate voice/violin from mridangam
        
        Args:
            audio: Audio signal
            margin: Margin size for separation (higher = cleaner separation)
            
        Returns:
            Tuple of (harmonic_audio, percussive_audio)
        """
        # HPSS: Harmonic-Percussive Source Separation
        # Harmonic = voice, violin, flute, tanpura
        # Percussive = mridangam, kanjira, ghatam
        y_harmonic, y_percussive = librosa.effects.hpss(
            audio, 
            margin=margin
        )
        return y_harmonic, y_percussive
    
    def preprocess_for_raga_detection(self, audio: np.ndarray,
                                      apply_hpss: bool = True,
                                      apply_bandpass: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline optimized for raga detection
        
        Pipeline:
        1. Normalize audio
        2. Apply HPS to remove percussion (if enabled)
        3. Apply bandpass filter for melodic range (if enabled)
        4. Remove silence
        
        Args:
            audio: Input audio signal
            apply_hpss: Whether to apply harmonic-percussive separation
            apply_bandpass: Whether to apply bandpass filtering
            
        Returns:
            Preprocessed audio ready for feature extraction
        """
        # Step 1: Normalize
        audio = self.normalize_audio(audio)
        
        # Step 2: HPS - Isolate harmonic content (voice/violin)
        if apply_hpss:
            audio, _ = self.harmonic_percussive_separation(audio)
        
        # Step 3: Bandpass filter - Keep melodic frequency range
        if apply_bandpass:
            audio = self.apply_bandpass_filter(audio)
        
        # Step 4: Remove silence
        audio = self.remove_silence(audio, top_db=20)
        
        return audio
    
    def apply_pre_emphasis(self, audio: np.ndarray, 
                          coef: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter to amplify high frequencies
        
        Args:
            audio: Audio signal
            coef: Pre-emphasis coefficient
            
        Returns:
            Filtered audio
        """
        return np.append(audio[0], audio[1:] - coef * audio[:-1])
    
    def extract_mel_spectrogram(self, audio: np.ndarray,
                               n_mels: int = 128,
                               fmax: int = 8000) -> np.ndarray:
        """
        Extract mel spectrogram from audio
        
        Args:
            audio: Audio signal
            n_mels: Number of mel bands
            fmax: Maximum frequency
            
        Returns:
            Mel spectrogram in dB
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=n_mels,
            fmax=fmax
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def preprocess(self, audio_path: str, 
                   remove_silence: bool = True,
                   normalize: bool = True,
                   target_duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Complete preprocessing pipeline
        
        Args:
            audio_path: Path to audio file
            remove_silence: Whether to remove silence
            normalize: Whether to normalize audio
            target_duration: Target duration in seconds (overrides self.duration)
            
        Returns:
            Tuple of (preprocessed_audio, sample_rate)
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Remove silence if requested
        if remove_silence:
            audio = self.remove_silence(audio)
        
        # Normalize if requested
        if normalize:
            audio = self.normalize_audio(audio)
        
        # Pad or trim if target duration specified
        if target_duration:
            target_samples = int(target_duration * sr)
            audio = self.pad_or_trim(audio, target_samples)
        
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save preprocessed audio to file
        
        Args:
            audio: Audio signal
            output_path: Output file path
        """
        sf.write(output_path, audio, self.sr)


def augment_audio(audio: np.ndarray, sr: int) -> list:
    """
    Create augmented variations of audio for data augmentation
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        
    Returns:
        List of augmented audio samples
    """
    augmented = []
    
    # Original
    augmented.append(audio)
    
    # Time stretch (faster)
    augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
    
    # Time stretch (slower)
    augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
    
    # Pitch shift up (2 semitones)
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
    
    # Pitch shift down (2 semitones)
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))
    
    # Add slight noise
    noise = np.random.normal(0, 0.005, len(audio))
    augmented.append(audio + noise)
    
    return augmented


if __name__ == "__main__":
    # Example usage
    preprocessor = AudioPreprocessor(sample_rate=22050, duration=30)
    
    # Load and preprocess audio
    audio_path = "path/to/audio.mp3"
    try:
        audio, sr = preprocessor.preprocess(audio_path)
        print(f"Preprocessed audio shape: {audio.shape}")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {len(audio) / sr:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
