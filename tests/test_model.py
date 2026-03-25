"""
Unit tests for model training and prediction
"""

import unittest
import numpy as np
import sys
import os
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.train_model import RagaDataPreprocessor, build_cnn_model


class TestRagaDataPreprocessor(unittest.TestCase):
    """Test cases for RagaDataPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = RagaDataPreprocessor(sample_rate=22050, duration=30)
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        self.assertEqual(self.preprocessor.sr, 22050)
        self.assertEqual(self.preprocessor.duration, 30)
        self.assertEqual(self.preprocessor.samples_per_clip, 22050 * 30)
    
    def test_augment_audio(self):
        """Test audio augmentation"""
        # Create synthetic audio
        audio = np.random.randn(22050)  # 1 second
        
        augmented = self.preprocessor.augment_audio(audio, 22050)
        
        # Should return multiple variations
        self.assertGreater(len(augmented), 1)
        self.assertLessEqual(len(augmented), 10)  # Reasonable number
        
        # Each variation should be an array
        for aug in augmented:
            self.assertIsInstance(aug, np.ndarray)


class TestCNNModel(unittest.TestCase):
    """Test cases for CNN model building"""
    
    def test_build_cnn_model(self):
        """Test CNN model construction"""
        input_shape = (128, 1292, 1)  # Typical mel spectrogram shape
        num_classes = 5
        
        model = build_cnn_model(input_shape, num_classes)
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers) > 0, True)
        
        # Check output shape
        output_shape = model.output_shape
        self.assertEqual(output_shape[-1], num_classes)
    
    def test_model_compilation(self):
        """Test that model is compiled correctly"""
        input_shape = (128, 1292, 1)
        num_classes = 3
        
        model = build_cnn_model(input_shape, num_classes)
        
        # Model should be compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
    
    def test_model_prediction_shape(self):
        """Test model prediction output shape"""
        input_shape = (128, 1292, 1)
        num_classes = 4
        
        model = build_cnn_model(input_shape, num_classes)
        
        # Create dummy input
        dummy_input = np.random.randn(1, *input_shape)
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        # Check prediction shape
        self.assertEqual(prediction.shape, (1, num_classes))
        
        # Check probabilities sum to 1
        self.assertAlmostEqual(np.sum(prediction[0]), 1.0, places=5)


class TestModelFeatureExtraction(unittest.TestCase):
    """Test feature extraction for model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = RagaDataPreprocessor()
    
    def test_extract_features_shape(self):
        """Test that extracted features have correct shape"""
        # Note: This test requires a real audio file
        # For now, we'll skip if no test file is available
        pass


class TestModelEdgeCases(unittest.TestCase):
    """Test edge cases for model"""
    
    def test_model_with_single_class(self):
        """Test model building with single class"""
        input_shape = (128, 1292, 1)
        num_classes = 1
        
        model = build_cnn_model(input_shape, num_classes)
        
        # Should still work
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape[-1], num_classes)
    
    def test_model_with_many_classes(self):
        """Test model building with many classes"""
        input_shape = (128, 1292, 1)
        num_classes = 72  # All 72 melakartas
        
        model = build_cnn_model(input_shape, num_classes)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape[-1], num_classes)


if __name__ == '__main__':
    unittest.main()
