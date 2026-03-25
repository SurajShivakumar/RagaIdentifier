# Project Structure Overview

## Complete Directory Tree

```
raga-identifier/
│
├── .gitignore                      # Git ignore rules
├── README.md                       # Main documentation
├── QUICKSTART.md                   # Quick start guide
├── MIGRATION.md                    # Migration guide from old structure
├── requirements.txt                # Python dependencies
├── example_usage.py                # Example workflow script
│
├── data/                           # Audio data
│   ├── README.md
│   ├── raw/                        # Original recordings
│   │   ├── Begada/
│   │   ├── Shankarabharanam/
│   │   └── ARCHIVE/                # Archived ragas
│   ├── curated/                    # Manually curated minimal dataset
│   │   ├── Begada/                 # 15 high-quality files
│   │   ├── Shankarabharanam/       # 13 high-quality files
│   │   └── dataset_info.json       # Dataset metadata
│   ├── processed/                  # Preprocessed audio
│   │   └── .gitkeep
│   ├── notes/                      # Extracted note sequences
│   │   └── .gitkeep
│   └── training_data/              # Prepared training data
│       └── .gitkeep
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_pitch_detection.ipynb
│   └── 03_note_extraction.ipynb
│
├── src/                            # Source code
│   ├── __init__.py
│   │
│   ├── audio_processing/           # Audio processing module
│   │   ├── __init__.py
│   │   ├── preprocess.py          # Audio preprocessing
│   │   ├── pitch_detect.py        # Pitch detection & tonic ID
│   │   └── smoothing.py           # Note stabilization & gamakas
│   │
│   ├── feature_extraction/         # Feature extraction module
│   │   ├── __init__.py
│   │   └── extract_notes.py       # Arohanam/avarohanam extraction
│   │
│   ├── model/                      # ML model module
│   │   ├── __init__.py
│   │   ├── crnn_model.py          # CRNN architecture definition
│   │   └── train_crnn.py          # Training pipeline
│   │
│   ├── feature_extraction/         # Feature extraction
│   │   ├── __init__.py
│   │   ├── extract_notes.py       # Note extraction
│   │   └── advanced_features.py   # Mel-spectrogram features
│   │
│   └── utils/                      # Utilities
│       ├── __init__.py
│       └── helpers.py             # Helper functions
│
├── models/                         # Trained models
│   ├── README.md
│   ├── best_model.h5              # Best validation accuracy model
│   ├── raga_model.h5              # Final trained model
│   ├── label_encoder.pkl          # Label encoder
│   └── logs/                      # TensorBoard logs
│
├── scripts/                        # Utility scripts
│   ├── README.md
│   ├── dunya_data_fetcher.py      # Dunya API metadata fetcher
│   ├── list_dunya_ragas.py        # List available ragas
│   ├── analyze_audio_quality.py   # Audio quality analysis
│   ├── identify_corrupted_files.py # Find corrupted files
│   ├── curate_dataset.py          # Interactive curation tool
│   └── create_minimal_dataset.py  # Create minimal dataset
│
├── api/                            # REST API
│   ├── app.py                     # FastAPI application
│   ├── requirements.txt           # API-specific dependencies
│   └── static/
│       └── index.html             # Web UI
│
├── mobile_app/                     # Mobile app (placeholder)
│   └── .gitkeep
│
└── tests/                          # Unit tests
    ├── __init__.py
    ├── test_pitch.py              # Pitch detection tests
    ├── test_notes.py              # Note extraction tests
    ├── test_model.py              # Model tests
    └── test_model_predictions.py # Test model on sample files
```

## Module Responsibilities

### 📦 `src/audio_processing/`
**Purpose**: Low-level audio processing

- **preprocess.py**
  - Audio loading and normalization
  - Silence removal
  - Mel spectrogram extraction
  - Data augmentation

- **pitch_detect.py**
  - Fundamental frequency (F0) detection
  - Tonic (Sa) identification
  - Pitch-to-cents conversion
  - Note name mapping

- **smoothing.py**
  - Pitch contour smoothing
  - Stable note region detection
  - Gamaka identification
  - Outlier removal

### 🎵 `src/feature_extraction/`
**Purpose**: Musical feature extraction

- **extract_notes.py**
  - Arohanam (ascending scale) extraction
  - Avarohanam (descending scale) extraction
  - Scale type identification (audava/shadava/sampurna)
  - Characteristic phrase detection
  - Vadi/Samvadi identification
  - Raga template comparison

### 🤖 `src/model/`
**Purpose**: Machine learning models

- **train_model.py**
  - Dataset preparation
  - CNN model architecture
  - Training pipeline
  - Model evaluation
  - Model persistence

- **match_raga.py**
  - ML-based prediction
  - Rule-based matching
  - Batch prediction
  - Result formatting

### 🛠️ `src/utils/`
**Purpose**: Common utilities

- **helpers.py**
  - File I/O utilities
  - Audio file discovery
  - Time formatting
  - Note conversions
  - Similarity calculations

### 🌐 `api/`
**Purpose**: REST API backend

- **app.py**
  - FastAPI endpoints
  - File upload handling
  - ML prediction endpoint
  - Rule-based analysis endpoint
  - Combined identification

### 📓 `notebooks/`
**Purpose**: Interactive analysis

- **01_data_exploration.ipynb**
  - Dataset statistics
  - Waveform visualization
  - Spectrogram analysis

- **02_pitch_detection.ipynb**
  - Pitch detection experiments
  - Tonic identification
  - Pitch contour visualization

- **03_note_extraction.ipynb**
  - Note sequence extraction
  - Pattern analysis
  - Raga matching

### 🧪 `tests/`
**Purpose**: Unit tests

- **test_pitch.py** - Pitch detection tests
- **test_notes.py** - Note extraction tests
- **test_model.py** - Model training/prediction tests

## File Count Summary

- **Python Modules**: 15 files
- **Notebooks**: 3 files
- **Tests**: 3 files
- **Documentation**: 5 files (README, QUICKSTART, MIGRATION, etc.)
- **Configuration**: 3 files (.gitignore, requirements.txt, api/requirements.txt)

**Total**: ~30 organized files

## Key Features by Location

### Core Functionality
- Audio preprocessing: `src/audio_processing/preprocess.py`
- Pitch detection: `src/audio_processing/pitch_detect.py`
- Note extraction: `src/feature_extraction/extract_notes.py`
- ML training: `src/model/train_model.py`
- Prediction: `src/model/match_raga.py`

### User-Facing
- Quick start: `QUICKSTART.md`
- Example script: `example_usage.py`
- API server: `api/app.py`
- Notebooks: `notebooks/*.ipynb`

### Development
- Tests: `tests/*.py`
- Documentation: `README.md`, `MIGRATION.md`
- Configuration: `requirements.txt`

## Data Flow

```
Audio File
    ↓
[Audio Preprocessing] → preprocess.py
    ↓
[Pitch Detection] → pitch_detect.py
    ↓
[Note Stabilization] → smoothing.py
    ↓
[Feature Extraction] → extract_notes.py
    ↓
    ├── [Rule-Based Matching] → match_raga.py
    └── [ML Prediction] → train_model.py → match_raga.py
         ↓
    Raga Identification Result
```

## Usage Patterns

### 1. Training
```
data/raw/ → train_model.py → models/raga_model.h5
```

### 2. Prediction
```
audio.mp3 → match_raga.py → prediction results
```

### 3. Analysis
```
audio.mp3 → pitch_detect.py → extract_notes.py → analysis results
```

### 4. API
```
HTTP Request → api/app.py → src/model/ → JSON Response
```

## Next Steps

1. ✅ Structure created
2. ✅ Code refactored
3. ✅ Tests written
4. ✅ Documentation added
5. ⏭️ Add training data
6. ⏭️ Train model
7. ⏭️ Test prediction
8. ⏭️ Deploy API

---

**Note**: The old `src/ml/` directory can be removed once you're comfortable with the new structure. See `MIGRATION.md` for details.
