"""
Test model predictions on actual files to debug why it's always predicting Shankarabharanam
"""

import numpy as np
import pickle
import librosa
import tensorflow as tf
from tensorflow import keras
import os

# Load model and label encoder
model = keras.models.load_model('models/best_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("Model loaded successfully")
print(f"Classes: {label_encoder.classes_}")
print(f"Model output shape: {model.output_shape}")

# Function to extract mel spectrogram (matching training code)
def extract_mel_spectrogram(audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """Extract mel spectrogram matching the training pipeline"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def resize_spectrogram(mel_spec, target_shape=(128, 128)):
    """Resize spectrogram to target shape"""
    from scipy.ndimage import zoom
    zoom_factors = (target_shape[0] / mel_spec.shape[0], target_shape[1] / mel_spec.shape[1])
    resized = zoom(mel_spec, zoom_factors, order=1)
    return resized

def test_file(file_path, true_label):
    """Test model on a single file"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(file_path)}")
    print(f"True label: {true_label}")
    
    # Load audio
    audio, sr = librosa.load(file_path, sr=22050, duration=30.0)
    print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
    
    # Extract mel spectrogram
    mel_spec = extract_mel_spectrogram(audio, sr)
    print(f"Raw mel spectrogram shape: {mel_spec.shape}")
    
    # Resize to match model input
    mel_spec_resized = resize_spectrogram(mel_spec, (128, 128))
    print(f"Resized mel spectrogram shape: {mel_spec_resized.shape}")
    
    # Normalize
    mel_spec_norm = (mel_spec_resized - np.mean(mel_spec_resized)) / (np.std(mel_spec_resized) + 1e-6)
    
    # Add batch and channel dimensions
    X = mel_spec_norm[np.newaxis, :, :, np.newaxis]
    print(f"Model input shape: {X.shape}")
    
    # Predict
    predictions = model.predict(X, verbose=0)
    print(f"\nRaw predictions: {predictions[0]}")
    
    predicted_idx = np.argmax(predictions[0])
    predicted_label = label_encoder.classes_[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    print(f"Predicted: {predicted_label} ({confidence:.2f}% confidence)")
    print(f"Correct: {'✓' if predicted_label == true_label else '✗'}")
    
    return predicted_label == true_label

# Test files
test_files = [
    ("data/raw/Begada/01 Begada --Intha Chaalamu (Varnam) TM Krishan.mp3", "Begada"),
    ("data/raw/Begada/02 Begada - Naadaloludai (Kriti Thyagarajar).mp3", "Begada"),
    ("data/raw/Begada/03 Begada - Nanati Bratuku (Kriti Thyagarajar).mp3", "Begada"),
]

# Find Shankarabharanam files
import glob
shankar_files = glob.glob("data/raw/Shankarabharanam/*.mp3")[:3]
for file in shankar_files:
    test_files.append((file, "Shankarabharanam"))

print("Starting model evaluation...")
print(f"Total test files: {len(test_files)}")

correct = 0
total = 0

for file_path, true_label in test_files:
    if os.path.exists(file_path):
        if test_file(file_path, true_label):
            correct += 1
        total += 1
    else:
        print(f"File not found: {file_path}")

print(f"\n{'='*60}")
print(f"RESULTS: {correct}/{total} correct ({100*correct/total:.1f}% accuracy)")
print(f"{'='*60}")
