"""
Train a MUCH simpler model for tiny datasets
This uses only ~1000 parameters instead of 159,746
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.audio_processing import AudioPreprocessor
from src.feature_extraction.advanced_features import AdvancedFeatureExtractor
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle

# Simplified model for tiny dataset
def create_tiny_model(num_classes):
    """Create a very small model suitable for tiny datasets"""
    model = keras.Sequential([
        # Input: 128x128x1
        keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.GlobalAveragePooling2D(),
        
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Load data
DATA_DIR = '../data/curated'
preprocessor = AudioPreprocessor(sample_rate=22050, duration=30)
feature_extractor = AdvancedFeatureExtractor(sample_rate=22050)

print("Loading audio files...")
X = []
y = []

for raga in ['Begada', 'Shankarabharanam']:
    raga_dir = os.path.join(DATA_DIR, raga)
    files = [f for f in os.listdir(raga_dir) if f.endswith('.mp3')]
    
    print(f"{raga}: {len(files)} files")
    
    for file in files:
        file_path = os.path.join(raga_dir, file)
        try:
            audio, sr = preprocessor.load_audio(file_path)
            mel_spec = feature_extractor.extract_mel_spectrogram(audio)
            mel_spec_normalized = feature_extractor.prepare_for_cnn(mel_spec, target_shape=(128, 128))
            
            X.append(mel_spec_normalized)
            y.append(raga)
        except Exception as e:
            print(f"Error loading {file}: {e}")

X = np.array(X)
X = np.expand_dims(X, axis=-1)
y = np.array(y)

print(f"\nDataset: {X.shape}, Labels: {y.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Create and compile model
model = create_tiny_model(num_classes=len(label_encoder.classes_))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nModel has {model.count_params()} parameters (vs 159,746 before)")
model.summary()

# Train with VERY strong regularization
print("\nTraining...")
history = model.fit(
    X, y_categorical,
    epochs=50,
    batch_size=4,
    validation_split=0.15,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)

# Save
model.save('../models/tiny_model.h5')
with open('../models/tiny_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n✅ Tiny model saved!")
print(f"   Location: models/tiny_model.h5")
print(f"   Parameters: {model.count_params()}")
