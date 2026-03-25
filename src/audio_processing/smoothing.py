"""
Note Smoothing and Gamaka Handling
Stabilizes note sequences and handles ornamentations (gamakas)
"""

import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import median_filter
from typing import List, Tuple


class NoteStabilizer:
    """Handles note stabilization and gamaka detection"""
    
    def __init__(self, stable_duration_ms: int = 100, 
                 min_note_duration_ms: int = 50):
        """
        Initialize note stabilizer
        
        Args:
            stable_duration_ms: Min duration for a note to be considered stable
            min_note_duration_ms: Minimum duration to consider as a note
        """
        self.stable_duration_ms = stable_duration_ms
        self.min_note_duration_ms = min_note_duration_ms
    
    def median_filter_pitch(self, pitch_contour: np.ndarray,
                           window_size: int = 5) -> np.ndarray:
        """
        Apply median filter to smooth pitch contour
        
        Args:
            pitch_contour: Array of pitch values
            window_size: Size of median filter window
            
        Returns:
            Smoothed pitch contour
        """
        # Skip NaN values
        valid_mask = ~np.isnan(pitch_contour)
        smoothed = pitch_contour.copy()
        
        if valid_mask.sum() > window_size:
            smoothed[valid_mask] = medfilt(
                pitch_contour[valid_mask], 
                kernel_size=window_size
            )
        
        return smoothed
    
    def savitzky_golay_filter(self, pitch_contour: np.ndarray,
                              window_length: int = 11,
                              polyorder: int = 3) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for smooth pitch tracking
        
        Args:
            pitch_contour: Array of pitch values
            window_length: Length of filter window (must be odd)
            polyorder: Order of polynomial
            
        Returns:
            Smoothed pitch contour
        """
        valid_mask = ~np.isnan(pitch_contour)
        smoothed = pitch_contour.copy()
        
        if valid_mask.sum() > window_length:
            smoothed[valid_mask] = savgol_filter(
                pitch_contour[valid_mask],
                window_length=window_length,
                polyorder=polyorder
            )
        
        return smoothed
    
    def detect_stable_regions(self, pitch_contour: np.ndarray,
                             hop_length: int = 512,
                             sample_rate: int = 22050,
                             cents_threshold: float = 30) -> List[Tuple[int, int, float]]:
        """
        Detect stable pitch regions (sustained notes)
        
        Args:
            pitch_contour: Array of pitch values in Hz
            hop_length: Hop length used in pitch detection
            sample_rate: Audio sample rate
            cents_threshold: Max deviation in cents to consider stable
            
        Returns:
            List of (start_idx, end_idx, mean_pitch) for stable regions
        """
        if len(pitch_contour) == 0:
            return []
        
        # Convert to cents for better comparison
        reference = np.nanmedian(pitch_contour)
        cents_contour = 1200 * np.log2(pitch_contour / reference)
        
        # Calculate frame duration in ms
        frame_duration_ms = (hop_length / sample_rate) * 1000
        min_frames = int(self.min_note_duration_ms / frame_duration_ms)
        
        stable_regions = []
        i = 0
        
        while i < len(cents_contour):
            if np.isnan(cents_contour[i]):
                i += 1
                continue
            
            # Start of potential stable region
            start = i
            mean_cents = cents_contour[i]
            
            # Extend region while deviation is small
            j = i + 1
            while j < len(cents_contour):
                if np.isnan(cents_contour[j]):
                    j += 1
                    continue
                
                # Check if still within threshold
                diff = abs(cents_contour[j] - mean_cents)
                if diff < cents_threshold:
                    j += 1
                else:
                    break
            
            # Check if region is long enough
            duration = j - start
            if duration >= min_frames:
                mean_pitch = np.nanmean(pitch_contour[start:j])
                stable_regions.append((start, j, mean_pitch))
            
            i = j if j > i else i + 1
        
        return stable_regions
    
    def identify_gamakas(self, pitch_contour: np.ndarray,
                        stable_regions: List[Tuple[int, int, float]],
                        vibrato_rate_hz: float = 5.0) -> List[Tuple[int, int, str]]:
        """
        Identify gamaka (ornamentations) between stable notes
        
        Args:
            pitch_contour: Array of pitch values
            stable_regions: List of stable note regions
            vibrato_rate_hz: Expected vibrato rate for gamaka detection
            
        Returns:
            List of (start_idx, end_idx, gamaka_type)
        """
        gamakas = []
        
        for i in range(len(stable_regions) - 1):
            _, end1, pitch1 = stable_regions[i]
            start2, _, pitch2 = stable_regions[i + 1]
            
            # Region between stable notes
            if start2 > end1:
                transition = pitch_contour[end1:start2]
                
                # Skip if too short
                if len(transition) < 3:
                    continue
                
                # Calculate pitch change
                pitch_diff_cents = 1200 * np.log2(pitch2 / pitch1)
                
                # Classify gamaka type
                if abs(pitch_diff_cents) < 50:
                    gamaka_type = 'jaru'  # Slide between same notes
                elif pitch_diff_cents > 0:
                    gamaka_type = 'kampita_upward'  # Upward oscillation
                else:
                    gamaka_type = 'kampita_downward'  # Downward oscillation
                
                # Check for oscillation
                if len(transition) >= 5:
                    diff = np.diff(transition[~np.isnan(transition)])
                    sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
                    if sign_changes >= 2:
                        gamaka_type = 'oscillation'  # Vibrato/trill
                
                gamakas.append((end1, start2, gamaka_type))
        
        return gamakas
    
    def extract_note_sequence(self, pitch_contour: np.ndarray,
                             cents_to_note_func,
                             hop_length: int = 512,
                             sample_rate: int = 22050) -> List[Tuple[str, float, float]]:
        """
        Extract sequence of stable notes with timings
        
        Args:
            pitch_contour: Array of pitch values in Hz
            cents_to_note_func: Function to convert cents to note name
            hop_length: Hop length used in pitch detection
            sample_rate: Audio sample rate
            
        Returns:
            List of (note_name, start_time_sec, duration_sec)
        """
        # Detect stable regions
        stable_regions = self.detect_stable_regions(
            pitch_contour, hop_length, sample_rate
        )
        
        # Convert to note sequence
        note_sequence = []
        frame_duration = hop_length / sample_rate
        
        for start_idx, end_idx, mean_pitch in stable_regions:
            # Convert pitch to cents (relative to some reference)
            reference = np.nanmedian(pitch_contour)
            cents = 1200 * np.log2(mean_pitch / reference)
            
            # Get note name
            note_name, _ = cents_to_note_func(cents)
            
            # Calculate timing
            start_time = start_idx * frame_duration
            duration = (end_idx - start_idx) * frame_duration
            
            note_sequence.append((note_name, start_time, duration))
        
        return note_sequence
    
    def remove_outliers(self, pitch_contour: np.ndarray,
                       std_threshold: float = 2.0) -> np.ndarray:
        """
        Remove outlier pitch values using statistical method
        
        Args:
            pitch_contour: Array of pitch values
            std_threshold: Number of standard deviations for outlier detection
            
        Returns:
            Pitch contour with outliers removed (set to NaN)
        """
        valid_mask = ~np.isnan(pitch_contour)
        valid_pitches = pitch_contour[valid_mask]
        
        if len(valid_pitches) < 3:
            return pitch_contour
        
        mean = np.mean(valid_pitches)
        std = np.std(valid_pitches)
        
        # Create output array
        cleaned = pitch_contour.copy()
        
        # Mark outliers as NaN
        outlier_mask = np.abs(pitch_contour - mean) > (std_threshold * std)
        cleaned[outlier_mask] = np.nan
        
        return cleaned


if __name__ == "__main__":
    # Example usage
    stabilizer = NoteStabilizer()
    
    # Example pitch contour
    pitch_contour = np.array([220, 222, 220, 240, 250, 247, 248, 247, 330, 329, 330])
    
    # Smooth pitch
    smoothed = stabilizer.median_filter_pitch(pitch_contour)
    print(f"Smoothed pitch: {smoothed}")
    
    # Detect stable regions
    stable = stabilizer.detect_stable_regions(pitch_contour)
    print(f"Stable regions: {stable}")
