"""
Unit tests for pitch detection module
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.audio_processing.pitch_detect import PitchDetector


class TestPitchDetector(unittest.TestCase):
    """Test cases for PitchDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = PitchDetector(sample_rate=22050, fmin=80, fmax=600)
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.sr, 22050)
        self.assertEqual(self.detector.fmin, 80)
        self.assertEqual(self.detector.fmax, 600)
    
    def test_pitch_to_cents(self):
        """Test pitch to cents conversion"""
        tonic = 220.0  # A3
        octave_above = 440.0  # A4
        
        # One octave = 1200 cents
        cents = self.detector.pitch_to_cents(octave_above, tonic)
        self.assertAlmostEqual(cents, 1200.0, places=1)
        
        # Same pitch = 0 cents
        cents = self.detector.pitch_to_cents(tonic, tonic)
        self.assertAlmostEqual(cents, 0.0, places=1)
    
    def test_cents_to_note(self):
        """Test cents to note conversion"""
        # Sa (0 cents)
        note, deviation = self.detector.cents_to_note(0)
        self.assertEqual(note, 'S')
        
        # Pa (702 cents)
        note, deviation = self.detector.cents_to_note(702)
        self.assertEqual(note, 'P')
        
        # Octave Sa (1200 cents)
        note, deviation = self.detector.cents_to_note(1200)
        self.assertEqual(note, 'S')
    
    def test_smooth_pitch(self):
        """Test pitch smoothing"""
        # Create test pitch contour with noise
        pitch_contour = np.array([220, 222, 220, 240, 220, 218, 220])
        
        smoothed = self.detector.smooth_pitch(pitch_contour, kernel_size=3)
        
        # Smoothed should have same length
        self.assertEqual(len(smoothed), len(pitch_contour))
        
        # Smoothed should reduce variations
        self.assertLess(np.std(smoothed), np.std(pitch_contour))
    
    def test_estimate_tonic(self):
        """Test tonic estimation"""
        # Create pitch contour centered around 220 Hz (A3)
        np.random.seed(42)
        pitch_contour = np.random.normal(220, 5, 1000)
        
        tonic = self.detector.estimate_tonic(pitch_contour, method='median')
        
        # Tonic should be close to 220 Hz
        self.assertAlmostEqual(tonic, 220, delta=10)
    
    def test_invalid_pitch_handling(self):
        """Test handling of invalid pitch values"""
        # Test with NaN values
        cents = self.detector.pitch_to_cents(np.nan, 220)
        self.assertTrue(np.isnan(cents))
        
        # Test with zero frequency
        cents = self.detector.pitch_to_cents(0, 220)
        self.assertTrue(np.isnan(cents))


class TestPitchDetectionEdgeCases(unittest.TestCase):
    """Test edge cases for pitch detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = PitchDetector()
    
    def test_empty_pitch_contour(self):
        """Test with empty pitch contour"""
        pitch_contour = np.array([])
        
        with self.assertRaises(ValueError):
            self.detector.estimate_tonic(pitch_contour)
    
    def test_all_nan_pitch_contour(self):
        """Test with all NaN values"""
        pitch_contour = np.full(100, np.nan)
        
        with self.assertRaises(ValueError):
            self.detector.estimate_tonic(pitch_contour)
    
    def test_extreme_frequencies(self):
        """Test with extreme frequency values"""
        # Very low frequency
        cents = self.detector.pitch_to_cents(10, 220)
        self.assertTrue(isinstance(cents, float))
        
        # Very high frequency
        cents = self.detector.pitch_to_cents(10000, 220)
        self.assertTrue(isinstance(cents, float))


if __name__ == '__main__':
    unittest.main()
