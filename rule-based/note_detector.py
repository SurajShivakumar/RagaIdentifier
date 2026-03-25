"""
Detect musical notes from audio using pitch detection and frequency analysis.
"""

import numpy as np
import librosa
from typing import List, Tuple, Dict

class NoteDetector:
    """
    Detects musical notes from audio using fundamental frequency detection.
    """
    
    def __init__(self, sr=22050):
        self.sr = sr
        
        # Carnatic music note frequencies (in Hz) - tuned to C (261.63 Hz)
        self.note_frequencies = {
            "S": 261.63,   # Sa
            "R1": 275.00,  # Shuddha Rishabham (komal re)
            "R2": 293.66,  # Chatushruti Rishabham
            "R3": 311.13,  # Shatshruti Rishabham
            "G1": 293.66,  # Shuddha Gandharam
            "G2": 311.13,  # Sadharana Gandharam
            "G3": 329.63,  # Antara Gandharam
            "M1": 349.23,  # Shuddha Madhyamam
            "M2": 369.99,  # Prati Madhyamam
            "P": 392.00,   # Panchamam
            "D1": 412.50,  # Shuddha Dhaivatam (komal dha)
            "D2": 440.00,  # Chatushruti Dhaivatam
            "D3": 466.16,  # Shatshruti Dhaivatam
            "N1": 440.00,  # Shuddha Nishadam
            "N2": 466.16,  # Kaisiki Nishadam
            "N3": 493.88,  # Kakali Nishadam
            "S'": 523.25   # Upper Sa
        }
        
        # Create frequency to note mapping
        self.freq_to_note = {}
        for note, freq in self.note_frequencies.items():
            # Add multiple octaves
            for octave in range(-1, 3):
                octave_freq = freq * (2 ** octave)
                self.freq_to_note[octave_freq] = note
    
    def detect_notes_from_audio(self, audio_path: str, hop_length: int = 512) -> List[Tuple[float, str, float]]:
        """
        Detect notes from an audio file.
        
        Args:
            audio_path: Path to audio file
            hop_length: Hop length for pitch detection
            
        Returns:
            List of (time, note, confidence) tuples
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Detect pitches using pYIN algorithm (probabilistic YIN)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length
        )
        
        # Convert frame indices to time
        times = librosa.frames_to_time(range(len(f0)), sr=sr, hop_length=hop_length)
        
        # Convert frequencies to notes
        detected_notes = []
        for time, freq, voiced, confidence in zip(times, f0, voiced_flag, voiced_probs):
            if voiced and not np.isnan(freq):
                note = self._freq_to_note(freq)
                if note:
                    detected_notes.append((time, note, confidence))
        
        return detected_notes
    
    def _freq_to_note(self, freq: float, tolerance: float = 30) -> str:
        """
        Convert frequency to closest note name.
        
        Args:
            freq: Frequency in Hz
            tolerance: Tolerance in Hz for note matching
            
        Returns:
            Note name or None
        """
        closest_note = None
        min_diff = float('inf')
        
        for note_freq, note_name in self.freq_to_note.items():
            diff = abs(freq - note_freq)
            if diff < min_diff and diff < tolerance:
                min_diff = diff
                closest_note = note_name
        
        return closest_note
    
    def extract_note_sequence(self, detected_notes: List[Tuple[float, str, float]], 
                             min_duration: float = 0.1) -> List[str]:
        """
        Extract a sequence of notes, removing duplicates and short notes.
        
        Args:
            detected_notes: List of (time, note, confidence) tuples
            min_duration: Minimum duration for a note to be included
            
        Returns:
            List of note names in sequence
        """
        if not detected_notes:
            return []
        
        # Group consecutive same notes
        note_sequence = []
        current_note = None
        note_start_time = None
        
        for time, note, confidence in detected_notes:
            if note != current_note:
                # Save previous note if it lasted long enough
                if current_note and note_start_time and (time - note_start_time) >= min_duration:
                    note_sequence.append(current_note)
                
                current_note = note
                note_start_time = time
        
        # Add the last note
        if current_note:
            note_sequence.append(current_note)
        
        return note_sequence
    
    def get_note_histogram(self, note_sequence: List[str]) -> Dict[str, int]:
        """
        Get frequency count of each note in the sequence.
        
        Args:
            note_sequence: List of note names
            
        Returns:
            Dictionary mapping note names to their frequency counts
        """
        histogram = {}
        for note in note_sequence:
            histogram[note] = histogram.get(note, 0) + 1
        return histogram
