"""
Unit tests for note extraction module
"""

import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.feature_extraction.extract_notes import NoteExtractor, RAGA_TEMPLATES


class TestNoteExtractor(unittest.TestCase):
    """Test cases for NoteExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = NoteExtractor()
    
    def test_initialization(self):
        """Test extractor initialization"""
        self.assertIsNotNone(self.extractor.note_to_index)
        self.assertIsNotNone(self.extractor.index_to_note)
        self.assertEqual(len(self.extractor.note_to_index), 12)
    
    def test_extract_arohanam(self):
        """Test arohanam extraction"""
        # Simple ascending sequence
        notes = ['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N3', 'S']
        arohanam = self.extractor.extract_arohanam(notes)
        
        self.assertEqual(arohanam, notes)
    
    def test_extract_avarohanam(self):
        """Test avarohanam extraction"""
        # Simple descending sequence
        notes = ['S', 'N3', 'D2', 'P', 'M1', 'G3', 'R2', 'S']
        avarohanam = self.extractor.extract_avarohanam(notes)
        
        self.assertEqual(avarohanam, notes)
    
    def test_identify_scale_type(self):
        """Test scale type identification"""
        # Pentatonic (audava)
        notes = ['S', 'R2', 'G3', 'P', 'D2']
        scale_type = self.extractor.identify_scale_type(notes)
        self.assertEqual(scale_type, 'audava')
        
        # Hexatonic (shadava)
        notes = ['S', 'R2', 'G3', 'M1', 'P', 'D2']
        scale_type = self.extractor.identify_scale_type(notes)
        self.assertEqual(scale_type, 'shadava')
        
        # Heptatonic (sampurna)
        notes = ['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N3']
        scale_type = self.extractor.identify_scale_type(notes)
        self.assertEqual(scale_type, 'sampurna')
    
    def test_identify_vadi_samvadi(self):
        """Test vadi and samvadi identification"""
        # Note sequence with clear vadi/samvadi
        notes = ['S', 'P', 'S', 'P', 'S', 'G3', 'G3', 'P']
        
        result = self.extractor.identify_vadi_samvadi(notes)
        
        self.assertIn('vadi', result)
        self.assertIn('samvadi', result)
        # S and P are most frequent
        self.assertIn(result['vadi'], ['S', 'P'])
    
    def test_compare_with_raga_template(self):
        """Test raga template comparison"""
        # Perfect match with Mayamalavagowla
        notes = ['S', 'R1', 'G3', 'M1', 'P', 'D1', 'N3', 'S']
        
        template = RAGA_TEMPLATES['Mayamalavagowla']
        score = self.extractor.compare_with_raga_template(
            notes,
            template['arohanam'],
            template['avarohanam']
        )
        
        # Should have high similarity
        self.assertGreater(score, 0.8)
    
    def test_analyze_note_patterns(self):
        """Test comprehensive pattern analysis"""
        # Create test note sequence
        note_sequence = [
            ('S', 0.0, 0.5),
            ('R2', 0.5, 0.3),
            ('G3', 0.8, 0.4),
            ('M1', 1.2, 0.6),
            ('P', 1.8, 0.5)
        ]
        
        analysis = self.extractor.analyze_note_patterns(note_sequence)
        
        self.assertIn('unique_notes', analysis)
        self.assertIn('arohanam', analysis)
        self.assertIn('avarohanam', analysis)
        self.assertIn('scale_type', analysis)
        self.assertEqual(analysis['note_count'], 5)


class TestRagaTemplates(unittest.TestCase):
    """Test raga template database"""
    
    def test_raga_templates_exist(self):
        """Test that raga templates are defined"""
        self.assertGreater(len(RAGA_TEMPLATES), 0)
    
    def test_raga_template_structure(self):
        """Test structure of raga templates"""
        for raga_name, template in RAGA_TEMPLATES.items():
            self.assertIn('arohanam', template)
            self.assertIn('avarohanam', template)
            self.assertIn('jati', template)
            
            # Arohanam and avarohanam should be lists
            self.assertIsInstance(template['arohanam'], list)
            self.assertIsInstance(template['avarohanam'], list)
            
            # Should not be empty
            self.assertGreater(len(template['arohanam']), 0)
            self.assertGreater(len(template['avarohanam']), 0)


class TestNoteExtractionEdgeCases(unittest.TestCase):
    """Test edge cases for note extraction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = NoteExtractor()
    
    def test_empty_note_sequence(self):
        """Test with empty note sequence"""
        arohanam = self.extractor.extract_arohanam([])
        self.assertEqual(arohanam, [])
        
        avarohanam = self.extractor.extract_avarohanam([])
        self.assertEqual(avarohanam, [])
    
    def test_single_note(self):
        """Test with single note"""
        notes = ['S']
        arohanam = self.extractor.extract_arohanam(notes)
        self.assertEqual(arohanam, ['S'])
    
    def test_invalid_notes(self):
        """Test with invalid note names"""
        notes = ['S', 'INVALID', 'P', 'FAKE']
        arohanam = self.extractor.extract_arohanam(notes)
        
        # Should only include valid notes
        self.assertNotIn('INVALID', arohanam)
        self.assertNotIn('FAKE', arohanam)


if __name__ == '__main__':
    unittest.main()
