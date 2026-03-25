# Raga Identifier

An intelligent system for identifying Carnatic music ragas from audio recordings using both rule-based analysis and machine learning.

## 🎵 Features

- **Audio Processing**: Advanced pitch detection, tonic identification, and note extraction
- **Pattern Recognition**: Arohanam/Avarohanam extraction and characteristic phrase identification
- **Machine Learning**: CNN-based raga classification from mel spectrograms
- **Rule-Based Matching**: Template matching with known raga patterns
- **REST API**: FastAPI backend for integration with mobile/web applications
- **Analysis Tools**: Jupyter notebooks for data exploration and experimentation

## 📁 Project Structure

```
raga-identifier/
│
├── api/                    # REST API & Web UI
├── data/
│   ├── raw/               # Original recordings
│   └── curated/           # High-quality minimal dataset
├── docs/                  # Documentation (NEW)
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks
├── scripts/               # Utility scripts (NEW)
├── src/                   # Source code
│   ├── audio_processing/
│   ├── feature_extraction/
│   └── model/            # CRNN training
└── tests/                 # Unit tests

See `docs/STRUCTURE.md` for detailed structure documentation.
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio file handling)

### Install Dependencies

```powershell
# Clone the repository
git clone https://github.com/yourusername/raga-identifier.git
cd raga-identifier

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Install FFmpeg (Windows)

```powershell
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

## 📚 Quick Start

### Train Model

```bash
python src/model/train_crnn.py
```

### Start Web Server

```bash
uvicorn api.app:app --reload
```

Visit http://localhost:8000 to use the web interface.

### Detailed Usage

See `docs/QUICKSTART.md` for detailed usage instructions.

```
data/raw/
  ├── Shankarabharanam/
  │   ├── song1.mp3
  │   └── song2.mp3
  ├── Kalyani/
  │   ├── song1.mp3
  │   └── song2.mp3
  └── ...
```

### 2. Train ML Model

```powershell
python src/model/train_model.py
```

This will:
- Load audio files from `data/raw/`
- Extract mel spectrogram features
- Train a CNN classifier
- Save the model to `models/raga_model.h5`

### 3. Predict Raga from Audio

```python
from src.model.match_raga import RagaPredictor

# Initialize predictor
predictor = RagaPredictor()

# Predict raga
results = predictor.predict('path/to/audio.mp3')
predictor.print_results(results)
```

### 4. Rule-Based Analysis

```python
from src.audio_processing import PitchDetector
from src.feature_extraction import NoteExtractor, RAGA_TEMPLATES
import librosa

# Load audio
y, sr = librosa.load('path/to/audio.mp3', duration=30)

# Detect pitch and extract notes
detector = PitchDetector()
pitch_contour = detector.detect_pitch_yin(y)
tonic = detector.estimate_tonic(pitch_contour)

# Extract note patterns
extractor = NoteExtractor()
# ... (see notebooks for complete workflow)
```

### 5. Start API Server

```powershell
cd api
pip install -r requirements.txt
python app.py
## 📊 Current Status

- **Architecture**: CRNN (CNN + LSTM) model for mel-spectrogram analysis
- **Training Data**: Curated dataset with 28 high-quality files (Begada, Shankarabharanam)
- **Web Interface**: Drag-and-drop UI with spectrogram visualization
- **API**: FastAPI backend serving predictions

## 📚 Documentation

See the `docs/` folder for detailed documentation:
- **QUICKSTART.md** - Getting started guide
- **STRUCTURE.md** - Project structure details
- **ML_PIPELINE.md** - Machine learning pipeline
- **IMPLEMENTATION.md** - Implementation details

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Librosa library for audio processing
- TensorFlow/Keras for machine learning
- Carnatic music community for domain knowledge

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🗺️ Roadmap

- [ ] Support for more ragas (all 72 melakartas)
- [ ] Real-time audio analysis
- [ ] Mobile app development
- [ ] Gamaka-aware note detection
- [ ] Multi-language support
- [ ] Cloud deployment

---

**Note**: This project is for educational and research purposes. Accuracy depends on the quality and quantity of training data.
