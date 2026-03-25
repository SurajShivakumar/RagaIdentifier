"""
Configuration for Raga Identifier
Targeting 5 specific Carnatic ragas: Kalyani, Mohanam, Begada, Mayamalavagowla, Shankarabharanam
"""

# Target ragas for detection
TARGET_RAGAS = [
    "Kalyani",
    "Mohanam", 
    "Begada",
    "Mayamalavagowla",
    "Shankarabharanam"
]

NUM_CLASSES = len(TARGET_RAGAS)

# Raga characteristics (for reference and understanding)
RAGA_INFO = {
    "Kalyani": {
        "melakartha": 65,
        "arohanam": "S R2 G3 M2 P D2 N3 S'",
        "avarohanam": "S' N3 D2 P M2 G3 R2 S",
        "type": "sampurna",  # 7 notes
        "janya_of": None,  # It's a melakartha (parent scale)
        "key_phrases": ["M2 P D2", "G3 M2 P"],
        "characteristic_note": "M2",  # Prati Madhyamam (sharp 4th)
        "description": "Bright, majestic raga with Prati Madhyamam"
    },
    "Mohanam": {
        "melakartha": None,
        "arohanam": "S R2 G3 P D2 S'",
        "avarohanam": "S' D2 P G3 R2 S",
        "type": "audava",  # 5 notes (pentatonic)
        "janya_of": 28,  # Harikambhoji
        "key_phrases": ["S R2 G3 P", "P D2 S'"],
        "characteristic_note": "P",  # No Ma or Ni
        "description": "Pentatonic, joyful raga - no Ma or Ni"
    },
    "Begada": {
        "melakartha": None,
        "arohanam": "S R2 G2 M1 P D1 N2 S'",
        "avarohanam": "S' N2 D1 P M1 G2 R2 S",
        "type": "sampurna",
        "janya_of": 29,  # Dheerasankarabharanam
        "key_phrases": ["G2 M1 P D1", "D1 N2 S'"],
        "characteristic_note": "D1",  # Shuddha Dhaivat
        "description": "Devotional, serious raga with Shuddha Dhaivat"
    },
    "Mayamalavagowla": {
        "melakartha": 15,
        "arohanam": "S R1 G3 M1 P D1 N3 S'",
        "avarohanam": "S' N3 D1 P M1 G3 R1 S",
        "type": "sampurna",
        "janya_of": None,  # It's a melakartha
        "key_phrases": ["R1 G3 M1", "P D1 N3"],
        "characteristic_note": "R1",  # Shuddha Rishabham with Antara Gandharam
        "description": "Morning raga, serious and devotional, distinctive R1-G3 jump"
    },
    "Shankarabharanam": {
        "melakartha": 29,
        "arohanam": "S R2 G3 M1 P D2 N3 S'",
        "avarohanam": "S' N3 D2 P M1 G3 R2 S",
        "type": "sampurna",
        "janya_of": None,  # It's a melakartha (also called Dheerasankarabharanam)
        "key_phrases": ["S R2 G3", "M1 P D2 N3"],
        "characteristic_note": "M1",  # Shuddha Madhyamam (vs Kalyani's M2)
        "description": "King of ragas, equivalent to Bilaval/Major scale"
    }
}

# Audio preprocessing settings
AUDIO_CONFIG = {
    "sample_rate": 22050,
    "duration": 30.0,  # seconds - trim/pad to this length
    "bandpass_low": 80,    # Hz - removes low rumble
    "bandpass_high": 1800,  # Hz - Carnatic melodic range
    "hop_length": 512,
    "n_fft": 2048,
    "n_mels": 128,
    "apply_hps": True,  # Harmonic-Percussive Separation
    "apply_bandpass": True  # Bandpass filter
}

# Model architecture settings
MODEL_CONFIG = {
    "input_shape": (128, 128, 1),  # (height, width, channels) for mel-spectrogram
    "num_classes": NUM_CLASSES,
    "cnn_filters": [32, 64, 128],  # Convolutional layers
    "lstm_units": [128, 64],  # LSTM layers
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "use_attention": False  # Set True for attention mechanism
}

# Model training settings
TRAINING_CONFIG = {
    "batch_size": 16,
    "epochs": 10,  # Reduced from 50 - sufficient for good accuracy
    "validation_split": 0.1,  # 10% of training data for validation
    "test_split": 0.2,  # 20% of data for testing
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "min_audio_length": 10.0,  # seconds
    "max_audio_length": 180.0,  # 3 minutes
}

# Data augmentation settings
AUGMENTATION_CONFIG = {
    "pitch_shift_range": (-2, 2),  # semitones
    "time_stretch_range": (0.9, 1.1),  # speed factor
    "noise_factor_range": (0.001, 0.005),  # amplitude
    "volume_range": (0.7, 1.3),  # amplitude multiplier
    "apply_probability": 0.5  # 50% chance to apply each augmentation
}

# File paths
PATHS = {
    "data_raw": "data/raw",
    "data_processed": "data/processed",
    "data_notes": "data/notes",
    "data_training": "data/training_data",
    "models": "models",
    "logs": "logs",
    "temp": "tmp",
    "model_file": "models/raga_model.h5",
    "encoder_file": "models/label_encoder.pkl",
    "best_model": "models/best_model.h5"
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_file_size": 50 * 1024 * 1024,  # 50 MB
    "allowed_extensions": [".mp3", ".wav", ".m4a", ".flac"],
    "cors_origins": ["*"]  # For production, restrict to specific domains
}

# Pitch detection settings
PITCH_CONFIG = {
    "method": "pyin",  # 'pyin', 'crepe', or 'piptrack'
    "fmin": 80,  # Hz - minimum pitch
    "fmax": 600,  # Hz - maximum pitch (typical Carnatic vocal range)
    "hop_length": 512
}

# Feature extraction settings
FEATURE_CONFIG = {
    "extract_mel": True,
    "extract_cqt": True,
    "extract_chroma": True,
    "extract_pitch": True,
    "extract_tonal_histogram": True,
    "extract_swara_histogram": True,
    "tonal_bins": 120,  # 10 cents resolution
    "swara_tolerance": 50  # cents tolerance for swara assignment
}

# Display info
def print_config():
    """Print configuration summary"""
    print("=" * 60)
    print("🎵 RAGA IDENTIFIER CONFIGURATION")
    print("=" * 60)
    print(f"\n📊 Target Ragas ({NUM_CLASSES}):")
    for i, raga in enumerate(TARGET_RAGAS, 1):
        info = RAGA_INFO[raga]
        print(f"  {i}. {raga:20s} - {info['description']}")
    print(f"\n🎼 Audio Settings:")
    print(f"  Sample Rate: {AUDIO_CONFIG['sample_rate']} Hz")
    print(f"  Duration: {AUDIO_CONFIG['duration']} seconds")
    print(f"  Bandpass: {AUDIO_CONFIG['bandpass_low']}-{AUDIO_CONFIG['bandpass_high']} Hz")
    print(f"  HPS Enabled: {AUDIO_CONFIG['apply_hps']}")
    print(f"\n🧠 Model Settings:")
    print(f"  Architecture: CRNN (CNN + LSTM)")
    print(f"  Input Shape: {MODEL_CONFIG['input_shape']}")
    print(f"  CNN Filters: {MODEL_CONFIG['cnn_filters']}")
    print(f"  LSTM Units: {MODEL_CONFIG['lstm_units']}")
    print(f"\n🎓 Training Settings:")
    print(f"  Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  Test Split: {TRAINING_CONFIG['test_split'] * 100}%")
    print(f"  Val Split: {TRAINING_CONFIG['validation_split'] * 100}%")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
