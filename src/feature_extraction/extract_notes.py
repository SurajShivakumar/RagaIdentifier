"""
Note Extraction and Raga Pattern Analysis
Extracts arohanam/avarohanam sequences and identifies raga characteristics
"""

import numpy as np
import json
import csv
from typing import List, Tuple, Dict, Optional
from collections import Counter


class NoteExtractor:
    """Extracts note sequences and raga patterns"""
    
    # Carnatic notes mapping
    CARNATIC_NOTES = {
        'S': 0,      # Shadja (Sa)
        'R1': 1,     # Shuddha Rishabha
        'R2': 2,     # Chatushruti Rishabha
        'G2': 3,     # Sadharana Gandhara
        'G3': 4,     # Antara Gandhara
        'M1': 5,     # Shuddha Madhyama
        'M2': 6,     # Prati Madhyama
        'P': 7,      # Panchama (Pa)
        'D1': 8,     # Shuddha Dhaivata
        'D2': 9,     # Chatushruti Dhaivata
        'N2': 10,    # Kaisiki Nishada
        'N3': 11     # Kakali Nishada
    }
    
    def __init__(self):
        """Initialize note extractor"""
        self.note_to_index = self.CARNATIC_NOTES
        self.index_to_note = {v: k for k, v in self.CARNATIC_NOTES.items()}
    
    def extract_arohanam(self, note_sequence: List[str]) -> List[str]:
        """
        Extract ascending note sequence (arohanam)
        
        Args:
            note_sequence: List of note names
            
        Returns:
            Arohanam sequence (ascending notes)
        """
        if not note_sequence:
            return []
        
        arohanam = []
        prev_idx = -1
        
        for note in note_sequence:
            if note not in self.note_to_index:
                continue
            
            current_idx = self.note_to_index[note]
            
            # Add if ascending or same
            if current_idx >= prev_idx:
                if not arohanam or arohanam[-1] != note:
                    arohanam.append(note)
                prev_idx = current_idx
        
        return arohanam
    
    def extract_avarohanam(self, note_sequence: List[str]) -> List[str]:
        """
        Extract descending note sequence (avarohanam)
        
        Args:
            note_sequence: List of note names
            
        Returns:
            Avarohanam sequence (descending notes)
        """
        if not note_sequence:
            return []
        
        avarohanam = []
        prev_idx = float('inf')
        
        for note in note_sequence:
            if note not in self.note_to_index:
                continue
            
            current_idx = self.note_to_index[note]
            
            # Add if descending or same
            if current_idx <= prev_idx:
                if not avarohanam or avarohanam[-1] != note:
                    avarohanam.append(note)
                prev_idx = current_idx
        
        return avarohanam
    
    def identify_scale_type(self, note_sequence: List[str]) -> str:
        """
        Identify if scale is audava (5), shadava (6), or sampurna (7)
        
        Args:
            note_sequence: List of unique notes
            
        Returns:
            Scale type classification
        """
        unique_notes = list(set(note_sequence))
        num_notes = len(unique_notes)
        
        if num_notes <= 5:
            return 'audava'  # Pentatonic
        elif num_notes == 6:
            return 'shadava'  # Hexatonic
        elif num_notes >= 7:
            return 'sampurna'  # Heptatonic
        else:
            return 'unknown'
    
    def extract_phrases(self, note_sequence: List[Tuple[str, float, float]],
                       min_phrase_length: int = 3) -> List[List[str]]:
        """
        Extract common phrases (sancharas) from note sequence with timings
        
        Args:
            note_sequence: List of (note, start_time, duration) tuples
            min_phrase_length: Minimum notes in a phrase
            
        Returns:
            List of note phrases
        """
        if len(note_sequence) < min_phrase_length:
            return []
        
        phrases = []
        
        # Extract overlapping n-grams
        for n in range(min_phrase_length, min(8, len(note_sequence) + 1)):
            for i in range(len(note_sequence) - n + 1):
                phrase = [note for note, _, _ in note_sequence[i:i+n]]
                phrases.append(phrase)
        
        return phrases
    
    def find_characteristic_phrases(self, phrases: List[List[str]],
                                   min_frequency: int = 2) -> Dict[str, int]:
        """
        Find frequently occurring characteristic phrases
        
        Args:
            phrases: List of note phrases
            min_frequency: Minimum occurrences to consider characteristic
            
        Returns:
            Dictionary of phrase -> frequency
        """
        # Convert phrases to strings for counting
        phrase_strings = [' '.join(phrase) for phrase in phrases]
        
        # Count occurrences
        phrase_counts = Counter(phrase_strings)
        
        # Filter by minimum frequency
        characteristic = {
            phrase: count 
            for phrase, count in phrase_counts.items() 
            if count >= min_frequency
        }
        
        return dict(sorted(characteristic.items(), 
                          key=lambda x: x[1], 
                          reverse=True))
    
    def identify_vadi_samvadi(self, note_sequence: List[str]) -> Dict[str, str]:
        """
        Identify vadi (most emphasized) and samvadi (second most) notes
        
        Args:
            note_sequence: List of notes
            
        Returns:
            Dictionary with vadi and samvadi notes
        """
        # Count note frequencies
        note_counts = Counter(note_sequence)
        
        # Get top 2
        most_common = note_counts.most_common(2)
        
        result = {}
        if len(most_common) >= 1:
            result['vadi'] = most_common[0][0]
        if len(most_common) >= 2:
            result['samvadi'] = most_common[1][0]
        
        return result
    
    def analyze_note_patterns(self, note_sequence: List[Tuple[str, float, float]]) -> Dict:
        """
        Complete analysis of note patterns
        
        Args:
            note_sequence: List of (note, start_time, duration) tuples
            
        Returns:
            Dictionary with comprehensive pattern analysis
        """
        # Extract just note names
        notes = [note for note, _, _ in note_sequence]
        
        # Get unique notes
        unique_notes = list(set(notes))
        
        # Extract arohanam and avarohanam
        arohanam = self.extract_arohanam(notes)
        avarohanam = self.extract_avarohanam(notes)
        
        # Identify scale type
        scale_type = self.identify_scale_type(unique_notes)
        
        # Extract phrases
        phrases = self.extract_phrases(note_sequence)
        
        # Find characteristic phrases
        characteristic_phrases = self.find_characteristic_phrases(phrases)
        
        # Identify vadi/samvadi
        vadi_samvadi = self.identify_vadi_samvadi(notes)
        
        return {
            'unique_notes': sorted(unique_notes),
            'note_count': len(unique_notes),
            'scale_type': scale_type,
            'arohanam': arohanam,
            'avarohanam': avarohanam,
            'vadi_samvadi': vadi_samvadi,
            'characteristic_phrases': characteristic_phrases,
            'total_notes': len(notes)
        }
    
    def save_to_csv(self, note_sequence: List[Tuple[str, float, float]], 
                   output_path: str):
        """
        Save note sequence to CSV file
        
        Args:
            note_sequence: List of (note, start_time, duration) tuples
            output_path: Path to output CSV file
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Note', 'Start_Time_Sec', 'Duration_Sec'])
            
            for note, start_time, duration in note_sequence:
                writer.writerow([note, f'{start_time:.3f}', f'{duration:.3f}'])
    
    def save_to_json(self, analysis_results: Dict, output_path: str):
        """
        Save analysis results to JSON file
        
        Args:
            analysis_results: Dictionary with analysis results
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
    
    def load_from_csv(self, csv_path: str) -> List[Tuple[str, float, float]]:
        """
        Load note sequence from CSV file
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of (note, start_time, duration) tuples
        """
        note_sequence = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                note = row['Note']
                start_time = float(row['Start_Time_Sec'])
                duration = float(row['Duration_Sec'])
                note_sequence.append((note, start_time, duration))
        
        return note_sequence
    
    def compare_with_raga_template(self, extracted_notes: List[str],
                                   template_arohanam: List[str],
                                   template_avarohanam: List[str]) -> float:
        """
        Compare extracted notes with known raga template
        
        Args:
            extracted_notes: Extracted note sequence
            template_arohanam: Expected arohanam for raga
            template_avarohanam: Expected avarohanam for raga
            
        Returns:
            Similarity score (0-1)
        """
        # Extract patterns from input
        arohanam = self.extract_arohanam(extracted_notes)
        avarohanam = self.extract_avarohanam(extracted_notes)
        
        # Calculate arohanam similarity
        arohanam_matches = sum(
            1 for note in arohanam if note in template_arohanam
        )
        arohanam_score = (arohanam_matches / len(template_arohanam) 
                         if template_arohanam else 0)
        
        # Calculate avarohanam similarity
        avarohanam_matches = sum(
            1 for note in avarohanam if note in template_avarohanam
        )
        avarohanam_score = (avarohanam_matches / len(template_avarohanam)
                           if template_avarohanam else 0)
        
        # Combined score
        return (arohanam_score + avarohanam_score) / 2


# Raga templates (examples)
RAGA_TEMPLATES = {
    'Mayamalavagowla': {
        'arohanam': ['S', 'R1', 'G3', 'M1', 'P', 'D1', 'N3', 'S'],
        'avarohanam': ['S', 'N3', 'D1', 'P', 'M1', 'G3', 'R1', 'S'],
        'jati': 'sampurna-sampurna'
    },
    'Sankarabharanam': {
        'arohanam': ['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N3', 'S'],
        'avarohanam': ['S', 'N3', 'D2', 'P', 'M1', 'G3', 'R2', 'S'],
        'jati': 'sampurna-sampurna'
    },
    'Kalyani': {
        'arohanam': ['S', 'R2', 'G3', 'M2', 'P', 'D2', 'N3', 'S'],
        'avarohanam': ['S', 'N3', 'D2', 'P', 'M2', 'G3', 'R2', 'S'],
        'jati': 'sampurna-sampurna'
    },
    'Mohanam': {
        'arohanam': ['S', 'R2', 'G3', 'P', 'D2', 'S'],
        'avarohanam': ['S', 'D2', 'P', 'G3', 'R2', 'S'],
        'jati': 'audava-audava'
    }
}


if __name__ == "__main__":
    # Example usage
    extractor = NoteExtractor()
    
    # Example note sequence with timings
    notes = [
        ('S', 0.0, 0.5),
        ('R2', 0.5, 0.3),
        ('G3', 0.8, 0.4),
        ('M1', 1.2, 0.6),
        ('P', 1.8, 0.5)
    ]
    
    # Analyze patterns
    analysis = extractor.analyze_note_patterns(notes)
    print(json.dumps(analysis, indent=2))
    
    # Save to files
    # extractor.save_to_csv(notes, 'notes.csv')
    # extractor.save_to_json(analysis, 'analysis.json')
