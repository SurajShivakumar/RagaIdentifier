"""
Training Pipeline for CRNN Raga Classifier
Handles data loading, augmentation, training, and evaluation
"""

import numpy as np
import os
import glob
from typing import Tuple, List, Dict, Optional
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.audio_processing import AudioPreprocessor, PitchDetector
from src.feature_extraction.advanced_features import AdvancedFeatureExtractor
from src.model.crnn_model import RagaCRNN, create_baseline_cnn


class RagaDataGenerator(keras.utils.Sequence):
    """
    Custom data generator with real-time augmentation
    Prevents loading all data into memory
    """
    
    def __init__(self, 
                 audio_paths: List[str],
                 labels: np.ndarray,
                 batch_size: int = 32,
                 target_shape: Tuple[int, int] = (128, 128),
                 augment: bool = True,
                 shuffle: bool = True):
        """
        Initialize data generator
        
        Args:
            audio_paths: List of audio file paths
            labels: One-hot encoded labels
            batch_size: Batch size
            target_shape: Target shape for mel spectrogram
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle data each epoch
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.augment = augment
        self.shuffle = shuffle
        
        self.preprocessor = AudioPreprocessor(sample_rate=22050, duration=30)
        self.feature_extractor = AdvancedFeatureExtractor(sample_rate=22050)
        
        self.indexes = np.arange(len(self.audio_paths))
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.audio_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get batch indices
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(batch_indexes)
        
        return X, y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_indexes):
        """Generate batch data"""
        X = np.empty((self.batch_size, *self.target_shape, 1))
        y = np.empty((self.batch_size, self.labels.shape[1]))
        
        for i, idx in enumerate(batch_indexes):
            # Load and preprocess audio
            try:
                audio, sr = self.preprocessor.load_audio(self.audio_paths[idx])
                
                # DISABLED aggressive preprocessing - testing raw audio
                # audio = self.preprocessor.preprocess_for_raga_detection(
                #     audio, apply_hpss=True, apply_bandpass=True
                # )
                
                # Apply augmentation
                if self.augment:
                    audio = self.__augment_audio(audio)
                
                # Extract mel spectrogram from raw audio
                mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
                
                # Prepare for CNN
                mel_spec = self.feature_extractor.prepare_for_cnn(mel_spec, self.target_shape)
                
                # Add channel dimension
                X[i,] = np.expand_dims(mel_spec, axis=-1)
                y[i,] = self.labels[idx]
                
            except Exception as e:
                print(f"⚠️ Skipping corrupted file: {self.audio_paths[idx]}")
                # Skip corrupted files - try next file in list
                for next_idx in range(len(self.audio_paths)):
                    if next_idx not in batch_indexes:
                        try:
                            audio, sr = self.preprocessor.load_audio(self.audio_paths[next_idx])
                            if self.augment:
                                audio = self.__augment_audio(audio)
                            mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
                            mel_spec = self.feature_extractor.prepare_for_cnn(mel_spec, self.target_shape)
                            X[i,] = np.expand_dims(mel_spec, axis=-1)
                            y[i,] = self.labels[next_idx]
                            break
                        except:
                            continue
                else:
                    # If all files fail, fill with zeros as last resort
                    X[i,] = np.zeros((*self.target_shape, 1))
                    y[i,] = self.labels[idx]
        
        return X, y
    
    def __augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random audio augmentations
        Improves model robustness and generalization
        Light augmentation for larger dataset
        """
        # Light time stretch - slight tempo variation
        if np.random.random() < 0.2:
            import librosa
            rate = np.random.uniform(0.95, 1.05)  # Very subtle
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # NO pitch shift - would change the raga notes
        
        # Light noise addition - simulates recording conditions
        if np.random.random() < 0.2:
            noise_factor = np.random.uniform(0.001, 0.003)
            noise = np.random.randn(len(audio))
            audio = audio + noise_factor * noise
        
        # Volume adjustment
        if np.random.random() < 0.4:
            volume_factor = np.random.uniform(0.8, 1.2)
            audio = audio * volume_factor
        
        return audio


class RagaTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, 
                 data_dir: str,
                 model_dir: str = 'models',
                 sample_rate: int = 22050,
                 duration: float = 30.0):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing raga audio files
            model_dir: Directory to save models
            sample_rate: Audio sample rate
            duration: Audio duration to use (seconds)
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.duration = duration
        
        os.makedirs(model_dir, exist_ok=True)
    
    def load_dataset(self, test_size: float = 0.2, 
                    val_size: float = 0.1) -> Dict:
        """
        Load and split dataset
        
        Expected directory structure:
        data_dir/
            Bhairavi/
                song1.mp3
                song2.mp3
            Kalyani/
                song1.mp3
            ...
        
        Args:
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            
        Returns:
            Dictionary with train/val/test splits and label encoder
        """
        print("📁 Loading dataset...")
        
        # Find all audio files organized by raga
        audio_paths = []
        raga_labels = []
        
        for raga_dir in glob.glob(os.path.join(self.data_dir, '*')):
            if not os.path.isdir(raga_dir):
                continue
            
            raga_name = os.path.basename(raga_dir)
            audio_files = glob.glob(os.path.join(raga_dir, '*.mp3')) + \
                         glob.glob(os.path.join(raga_dir, '*.wav')) + \
                         glob.glob(os.path.join(raga_dir, '*.m4a'))
            
            for audio_file in audio_files:
                audio_paths.append(audio_file)
                raga_labels.append(raga_name)
        
        print(f"✅ Found {len(audio_paths)} audio files across {len(set(raga_labels))} ragas")
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(raga_labels)
        
        # One-hot encode
        num_classes = len(label_encoder.classes_)
        one_hot_labels = keras.utils.to_categorical(encoded_labels, num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            audio_paths, one_hot_labels, test_size=test_size, 
            stratify=encoded_labels, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size,
            stratify=np.argmax(y_train, axis=1), random_state=42
        )
        
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"🏷️ Ragas: {', '.join(label_encoder.classes_)}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'num_classes': num_classes
        }
    
    def train(self, 
             dataset: Dict,
             model_type: str = 'crnn',
             epochs: int = 10,
             batch_size: int = 16,
             use_attention: bool = False,
             learning_rate: float = 0.001) -> Tuple[keras.Model, Dict]:
        """
        Train raga classification model
        
        Args:
            dataset: Dataset dictionary from load_dataset()
            model_type: 'crnn' or 'cnn'
            epochs: Number of training epochs
            batch_size: Batch size
            use_attention: Use attention mechanism (for CRNN only)
            
        Returns:
            Tuple of (trained_model, history)
        """
        print(f"\n🚀 Training {model_type.upper()} model...")
        
        # Create data generators
        train_gen = RagaDataGenerator(
            dataset['X_train'], dataset['y_train'],
            batch_size=batch_size, augment=True, shuffle=True
        )
        
        val_gen = RagaDataGenerator(
            dataset['X_val'], dataset['y_val'],
            batch_size=batch_size, augment=False, shuffle=False
        )
        
        # Build model
        input_shape = (128, 128, 1)
        
        if model_type == 'crnn':
            crnn = RagaCRNN(
                input_shape=input_shape,
                num_classes=dataset['num_classes'],
                learning_rate=learning_rate
            )
            
            if use_attention:
                model = crnn.build_attention_model()
            else:
                # Simpler architecture for 304 samples
                model = crnn.build_model(
                    cnn_filters=[32, 64],    # Reduced complexity
                    lstm_units=[64],          # Single LSTM layer
                    dropout_rate=0.2          # Reduced - model needs to learn first
                )
        else:
            model = create_baseline_cnn(
                input_shape=input_shape,
                num_classes=dataset['num_classes'],
                learning_rate=learning_rate
            )
        
        print(model.summary())
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-7
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.model_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        # Calculate class weights to handle imbalance
        from sklearn.utils.class_weight import compute_class_weight
        y_train_labels = np.argmax(dataset['y_train'], axis=1)
        classes = np.unique(y_train_labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train_labels
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"\n⚖️  Class weights (to balance dataset):")
        for idx, weight in class_weight_dict.items():
            print(f"   Class {idx}: {weight:.2f}")
        
        # Train with class weights
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict,  # Balance classes!
            verbose=1
        )
        
        # Save final model and label encoder
        model.save(os.path.join(self.model_dir, 'raga_model.h5'))
        
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(dataset['label_encoder'], f)
        
        print(f"\n✅ Model saved to {self.model_dir}")
        
        return model, history.history
    
    def evaluate(self, model: keras.Model, dataset: Dict, batch_size: int = 16):
        """Evaluate model on test set"""
        print("\n📊 Evaluating on test set...")
        
        test_gen = RagaDataGenerator(
            dataset['X_test'], dataset['y_test'],
            batch_size=batch_size, augment=False, shuffle=False
        )
        
        results = model.evaluate(test_gen, verbose=1)
        
        print(f"\n✅ Test Accuracy: {results[1]:.4f}")
        print(f"✅ Test Top-5 Accuracy: {results[2]:.4f}")
        
        return results


def main():
    """Example training script"""
    # Configuration - Train on only the 2 ragas with most data
    # Mohanam: 90 files, Shankarabharanam: 92 files = 182 total
    DATA_DIR = 'data/raw'
    MODEL_DIR = 'models'
    
    # Filter to only include the 2 largest ragas
    import os
    import shutil
    temp_dir = 'data/temp_2ragas'
    
    # Clean temp directory if it exists
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Copy only Mohanam and Shankarabharanam
    os.makedirs(temp_dir, exist_ok=True)
    for raga in ['Mohanam', 'Shankarabharanam']:
        src = os.path.join(DATA_DIR, raga)
        dst = os.path.join(temp_dir, raga)
        if os.path.exists(src):
            shutil.copytree(src, dst)
    
    print(f"🎯 Training on 2 ragas: Mohanam (90 files) + Shankarabharanam (92 files)")
    
    # Initialize trainer
    trainer = RagaTrainer(data_dir=temp_dir, model_dir=MODEL_DIR)
    
    # Load dataset
    dataset = trainer.load_dataset(test_size=0.15, val_size=0.15)
    
    # Train model with parameters optimized for 182 files
    model, history = trainer.train(
        dataset=dataset,
        model_type='cnn',  # CNN baseline
        epochs=50,  # Good for this dataset size
        batch_size=16,   # Larger batch for 182 files
        use_attention=False,
        learning_rate=0.0005  # Moderate learning rate
    )
    
    # Evaluate
    trainer.evaluate(model, dataset)
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
