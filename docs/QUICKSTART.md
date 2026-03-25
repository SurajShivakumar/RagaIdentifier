# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies

```powershell
# Install Python packages
pip install -r requirements.txt
```

### 2. Prepare Data

Place your audio files in the appropriate folders:

```
data/raw/
  ├── Shankarabharanam/
  │   └── your_audio.mp3
  ├── Kalyani/
  │   └── your_audio.mp3
  └── ...
```

You need at least 2 raga folders with at least 1 audio file each.

## Option 1: Use Existing Code (No Retraining)

If you have trained models from before, just use them:

```python
from src.model.match_raga import RagaPredictor

predictor = RagaPredictor()
results = predictor.predict('path/to/audio.mp3')
predictor.print_results(results)
```

## Option 2: Train New Model

### Step 1: Train the Model (10-30 minutes)

```powershell
python src/model/train_model.py
```

This will:
- Load audio from `data/raw/`
- Train a CNN model
- Save to `models/raga_model.h5`

### Step 2: Test Prediction

```python
from src.model.match_raga import RagaPredictor

predictor = RagaPredictor()
results = predictor.predict('data/raw/Shankarabharanam/test.mp3')
predictor.print_results(results)
```

## Option 3: Rule-Based Analysis (No Training)

Analyze audio without ML:

```python
from src.audio_processing import PitchDetector
from src.feature_extraction import NoteExtractor, RAGA_TEMPLATES
import librosa

# Load audio
y, sr = librosa.load('your_audio.mp3', duration=30)

# Detect pitch
detector = PitchDetector()
pitch = detector.detect_pitch_yin(y)
tonic = detector.estimate_tonic(pitch)

print(f"Tonic: {tonic:.2f} Hz")

# Extract and analyze notes
extractor = NoteExtractor()
# ... (see notebooks for full workflow)
```

## Option 4: Use API

### Start Server

```powershell
cd api
python app.py
```

### Make Request

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("audio.mp3", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

Or use curl:

```powershell
curl -X POST -F "file=@audio.mp3" http://localhost:8000/predict
```

## Explore with Notebooks

```powershell
jupyter notebook notebooks/
```

Open `01_data_exploration.ipynb` to start.

## Troubleshooting

### Import Errors

Make sure you're in the project root directory:

```powershell
cd c:\raga-identifier
python -c "import sys; print(sys.path)"
```

### Model Not Found

Train the model first:

```powershell
python src/model/train_model.py
```

### Audio Loading Errors

Install FFmpeg:

```powershell
choco install ffmpeg
```

## Next Steps

1. Add more training data for better accuracy
2. Experiment with notebooks
3. Try the API endpoints
4. Run unit tests: `pytest tests/`

## Support

Check `README.md` for detailed documentation.
