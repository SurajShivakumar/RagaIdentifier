"""
Main rule-based raga identification system.
"""

import os
import sys
from typing import Dict, Tuple

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from note_detector import NoteDetector
from pattern_matcher import RagaPatternMatcher

class RuleBasedRagaIdentifier:
    """
    Complete rule-based system for identifying ragas from audio files.
    """
    
    def __init__(self, sr=22050):
        """
        Initialize the identifier with note detector and pattern matcher.
        
        Args:
            sr: Sample rate for audio processing
        """
        self.note_detector = NoteDetector(sr=sr)
        self.pattern_matcher = RagaPatternMatcher()
    
    def identify_from_audio(self, audio_path: str, verbose: bool = False) -> Tuple[str, Dict[str, float]]:
        """
        Identify raga from an audio file.
        
        Args:
            audio_path: Path to the audio file
            verbose: If True, print detailed information
            
        Returns:
            Tuple of (predicted_raga, confidence_scores)
        """
        # Step 1: Detect notes from audio
        if verbose:
            print("🎵 Detecting notes from audio...")
        
        detected_notes = self.note_detector.detect_notes_from_audio(audio_path)
        
        if not detected_notes:
            if verbose:
                print("❌ No notes detected in audio")
            return "Unknown", {}
        
        # Step 2: Extract note sequence
        note_sequence = self.note_detector.extract_note_sequence(detected_notes)
        
        if verbose:
            print(f"📝 Detected note sequence: {' -> '.join(note_sequence[:20])}...")
            print(f"   Total notes detected: {len(note_sequence)}")
        
        # Step 3: Get note histogram
        note_histogram = self.note_detector.get_note_histogram(note_sequence)
        
        if verbose:
            print(f"📊 Note distribution:")
            for note, count in sorted(note_histogram.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {note}: {count} times")
        
        # Step 4: Identify raga using pattern matching
        if verbose:
            print("🔍 Matching against raga patterns...")
        
        confidence_scores = self.pattern_matcher.identify_raga(note_sequence, note_histogram)
        
        # Get the raga with highest confidence
        if confidence_scores:
            predicted_raga = max(confidence_scores, key=confidence_scores.get)
            
            if verbose:
                print(f"\n🎼 Results:")
                for raga, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {raga}: {score*100:.1f}%")
                
                print(f"\n✅ Predicted Raga: {predicted_raga} ({confidence_scores[predicted_raga]*100:.1f}%)")
                
                # Print explanation
                explanation = self.pattern_matcher.get_raga_explanation(
                    predicted_raga, note_sequence, note_histogram
                )
                print(f"\n{explanation}")
            
            return predicted_raga, confidence_scores
        else:
            if verbose:
                print("❌ Could not identify raga")
            return "Unknown", {}
    
    def get_supported_ragas(self):
        """Get list of ragas supported by this identifier."""
        from raga_definitions import get_all_ragas
        return get_all_ragas()
