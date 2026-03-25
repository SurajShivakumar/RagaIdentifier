"""
Quick test script to check what the ML model is predicting
"""

import os
import sys
import numpy as np
import pickle

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.audio_processing import AudioPreprocessor
from src.feature_extraction.advanced_features import AdvancedFeatureExtractor

def test_model_on_files():
    """Test the trained model on sample files"""
    
    # Load model
    import tensorflow as tf
    
    model_path = '../models/raga_model.h5'
    encoder_path = '../models/label_encoder.pkl'
    
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Model loaded. Classes: {label_encoder.classes_}")
    print(f"Model expects input shape: {model.input_shape}")
    
    # Initialize processors
    preprocessor = AudioPreprocessor(sample_rate=22050, duration=30)
    feature_extractor = AdvancedFeatureExtractor(sample_rate=22050)
    
    # Test files
    test_dir = '../data/curated'
    
    for raga in ['Begada', 'Shankarabharanam']:
        raga_dir = os.path.join(test_dir, raga)
        if not os.path.exists(raga_dir):
            continue
            
        files = [f for f in os.listdir(raga_dir) if f.endswith('.mp3')][:3]  # Test first 3
        
        print(f"\n{'='*60}")
        print(f"Testing {raga} files:")
        print(f"{'='*60}")
        
        for file in files:
            file_path = os.path.join(raga_dir, file)
            print(f"\nFile: {file}")
            
            try:
                # Load and process audio
                audio, sr = preprocessor.load_audio(file_path)
                mel_spec = feature_extractor.extract_mel_spectrogram(audio)
                mel_spec_normalized = feature_extractor.prepare_for_cnn(mel_spec, target_shape=(128, 128))
                
                # Prepare input
                mel_spec_input = np.expand_dims(mel_spec_normalized, axis=0)
                mel_spec_input = np.expand_dims(mel_spec_input, axis=-1)
                
                # Predict
                probabilities = model.predict(mel_spec_input, verbose=0)[0]
                
                # Show predictions
                for i, (cls, prob) in enumerate(zip(label_encoder.classes_, probabilities)):
                    print(f"  {cls}: {prob*100:.2f}%")
                
                predicted = label_encoder.classes_[np.argmax(probabilities)]
                actual = raga
                correct = "✅" if predicted == actual else "❌"
                print(f"  Predicted: {predicted} | Actual: {actual} {correct}")
                
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_model_on_files()
