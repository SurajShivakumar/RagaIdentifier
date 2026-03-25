"""
Pattern matching algorithms for raga identification based on note sequences.
"""

import os
import sys
from typing import List, Dict, Tuple

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from raga_definitions import RAGA_PATTERNS, get_all_ragas

class RagaPatternMatcher:
    """
    Matches detected note sequences against known raga patterns.
    """
    
    def __init__(self):
        self.ragas = RAGA_PATTERNS
    
    def identify_raga(self, note_sequence: List[str], note_histogram: Dict[str, int]) -> Dict[str, float]:
        """
        Identify the most likely raga from a note sequence.
        
        Args:
            note_sequence: Sequence of detected notes
            note_histogram: Frequency count of each note
            
        Returns:
            Dictionary mapping raga names to confidence scores (0-1)
        """
        scores = {}
        
        for raga_name in get_all_ragas():
            score = self._calculate_raga_score(raga_name, note_sequence, note_histogram)
            scores[raga_name] = score
        
        # Normalize scores to 0-1 range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v/max_score for k, v in scores.items()}
        
        return scores
    
    def _calculate_raga_score(self, raga_name: str, note_sequence: List[str], 
                             note_histogram: Dict[str, int]) -> float:
        """
        Calculate similarity score for a specific raga.
        """
        raga_info = self.ragas[raga_name]
        score = 0.0
        
        # 1. Check for characteristic notes (40% weight)
        important_notes = raga_info.get("important_notes", [])
        if important_notes:
            found_important = sum(1 for note in important_notes if note in note_histogram)
            score += (found_important / len(important_notes)) * 40
        
        # 2. Check for arohanam/avarohanam patterns (30% weight)
        arohanam_score = self._find_pattern_matches(note_sequence, raga_info["arohanam"])
        avarohanam_score = self._find_pattern_matches(note_sequence, raga_info["avarohanam"])
        score += (arohanam_score + avarohanam_score) / 2 * 30
        
        # 3. Check for characteristic phrases (20% weight)
        characteristic_phrases = raga_info.get("characteristic_phrases", [])
        if characteristic_phrases:
            phrase_matches = sum(
                self._find_pattern_matches(note_sequence, phrase) 
                for phrase in characteristic_phrases
            )
            avg_phrase_score = phrase_matches / len(characteristic_phrases)
            score += avg_phrase_score * 20
        
        # 4. Check vadi and samvadi prominence (10% weight)
        vadi = raga_info.get("vadi")
        samvadi = raga_info.get("samvadi")
        
        if vadi and samvadi and note_histogram:
            total_notes = sum(note_histogram.values())
            vadi_freq = note_histogram.get(vadi, 0) / total_notes if total_notes > 0 else 0
            samvadi_freq = note_histogram.get(samvadi, 0) / total_notes if total_notes > 0 else 0
            
            # Vadi and samvadi should be prominent
            prominence_score = (vadi_freq + samvadi_freq) * 100  # Scale up
            score += min(prominence_score, 10)  # Cap at 10 points
        
        return score
    
    def _find_pattern_matches(self, note_sequence: List[str], pattern: List[str]) -> float:
        """
        Find how well a pattern matches in the note sequence.
        Uses sliding window to find subsequence matches.
        
        Returns:
            Score between 0 and 1 indicating match quality
        """
        if not note_sequence or not pattern:
            return 0.0
        
        pattern_len = len(pattern)
        max_matches = 0
        
        # Slide through the note sequence
        for i in range(len(note_sequence) - pattern_len + 1):
            window = note_sequence[i:i + pattern_len]
            matches = sum(1 for w, p in zip(window, pattern) if w == p)
            max_matches = max(max_matches, matches)
        
        # Also check for partial matches (subsequences)
        for pattern_start in range(len(pattern)):
            for pattern_end in range(pattern_start + 2, len(pattern) + 1):
                sub_pattern = pattern[pattern_start:pattern_end]
                if self._is_subsequence(sub_pattern, note_sequence):
                    # Award points for finding subsequences
                    subsequence_score = len(sub_pattern) / pattern_len
                    max_matches = max(max_matches, int(subsequence_score * pattern_len))
        
        return max_matches / pattern_len
    
    def _is_subsequence(self, pattern: List[str], sequence: List[str]) -> bool:
        """
        Check if pattern appears as a subsequence in sequence.
        """
        pattern_idx = 0
        for note in sequence:
            if pattern_idx < len(pattern) and note == pattern[pattern_idx]:
                pattern_idx += 1
        return pattern_idx == len(pattern)
    
    def get_raga_explanation(self, raga_name: str, note_sequence: List[str], 
                            note_histogram: Dict[str, int]) -> str:
        """
        Generate an explanation of why a raga was identified.
        """
        raga_info = self.ragas.get(raga_name)
        if not raga_info:
            return f"Unknown raga: {raga_name}"
        
        explanation = [f"Identified as {raga_name}:"]
        
        # Check important notes
        important_notes = raga_info.get("important_notes", [])
        found_notes = [note for note in important_notes if note in note_histogram]
        if found_notes:
            explanation.append(f"  - Found characteristic notes: {', '.join(found_notes)}")
        
        # Check arohanam
        arohanam = raga_info["arohanam"]
        arohanam_score = self._find_pattern_matches(note_sequence, arohanam)
        explanation.append(f"  - Arohanam match: {arohanam_score*100:.1f}%")
        
        # Check avarohanam
        avarohanam = raga_info["avarohanam"]
        avarohanam_score = self._find_pattern_matches(note_sequence, avarohanam)
        explanation.append(f"  - Avarohanam match: {avarohanam_score*100:.1f}%")
        
        # Vadi and samvadi
        vadi = raga_info.get("vadi")
        samvadi = raga_info.get("samvadi")
        if vadi in note_histogram and samvadi in note_histogram:
            explanation.append(f"  - Vadi ({vadi}) and Samvadi ({samvadi}) present")
        
        return "\n".join(explanation)
