# 🚀 Quick Start: Advanced Raga Detection

This guide shows how to use the new ML-powered raga detection system.

---

## 📋 Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: For best pitch detection
pip install crepe
```

---

## 🎯 Option 1: Use the Web UI (Easiest)

### 1. Start the server
```bash
# From project root
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Open your browser
```
http://localhost:8000
```

### 3. Upload audio
- **Drag & drop** your audio file onto the purple zone
- Or **click** to browse files
- Supported: MP3, WAV, M4A, FLAC

### 4. Choose method
- **ML Model**: Uses advanced CRNN with HPS + bandpass filtering
- **Rule-Based**: Uses traditional pattern matching (fallback)

### 5. Get results!
- Top predicted raga with confidence score
- Top 5 predictions with visual progress bars

---

## 🐍 Option 2: Python API

### Example 1: Simple Prediction
```python
import requests

# Upload file
with open('my_song.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

result = response.json()
print(f"Raga: {result['predicted_raga']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top 5: {result['top_predictions']}")
```

### Example 2: Direct Python Usage
```python
from src.audio_processing import AudioPreprocessor
from src.feature_extraction.advanced_features import AdvancedFeatureExtractor
import tensorflow as tf
import numpy as np
import pickle

# Initialize
preprocessor = AudioPreprocessor(sample_rate=22050, duration=30)
feature_extractor = AdvancedFeatureExtractor(sample_rate=22050)

# Load model
model = tf.keras.models.load_model('models/raga_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Process audio
audio, sr = preprocessor.load_audio('test_song.mp3')

# Apply advanced preprocessing (HPS + bandpass)
audio_clean = preprocessor.preprocess_for_raga_detection(
    audio, 
    apply_hpss=True,      # Remove mridangam
    apply_bandpass=True   # 80-1800 Hz melodic range
)

# Extract features
mel_spec = feature_extractor.extract_mel_spectrogram(audio_clean)
mel_spec = feature_extractor.prepare_for_cnn(mel_spec, (128, 128))

# Add dimensions for model
mel_spec = mel_spec[np.newaxis, ..., np.newaxis]

# Predict
probs = model.predict(mel_spec)[0]
top_idx = np.argmax(probs)

print(f"🎵 Predicted Raga: {label_encoder.classes_[top_idx]}")
print(f"✅ Confidence: {probs[top_idx]:.2%}")

# Top 5
for i, idx in enumerate(np.argsort(probs)[::-1][:5], 1):
    print(f"{i}. {label_encoder.classes_[idx]}: {probs[idx]:.2%}")
```

---

## 🎓 Training Your Own Model

### Step 1: Organize Data
```
data/raw/
    Bhairavi/
        concert1.mp3
        concert2.mp3
    Kalyani/
        concert1.mp3
    Todi/
        concert1.mp3
    ...
```

**Tips:**
- Minimum 20-30 songs per raga
- 30 seconds to 3 minutes each
- Clean recordings preferred
- Multiple artists = better generalization

### Step 2: Train
```bash
python src/model/train_crnn.py
```

**Or customize:**
```python
from src.model.train_crnn import RagaTrainer

trainer = RagaTrainer(
    data_dir='data/raw',
    model_dir='models',
    sample_rate=22050,
    duration=30.0
)

# Load dataset
dataset = trainer.load_dataset(
    test_size=0.2,   # 20% for testing
    val_size=0.1     # 10% of train for validation
)

# Train CRNN
model, history = trainer.train(
    dataset=dataset,
    model_type='crnn',      # or 'cnn' for baseline
    epochs=50,
    batch_size=16,
    use_attention=False     # Set True for attention mechanism
)

# Evaluate
results = trainer.evaluate(model, dataset)
```

### Step 3: Model is saved automatically
```
models/
    raga_model.h5          # Trained model
    label_encoder.pkl      # Raga name encoder
    best_model.h5          # Best checkpoint
    logs/                  # TensorBoard logs
```

### Step 4: View training progress
```bash
tensorboard --logdir=models/logs
```

---

## 🎹 Understanding the Pipeline

### What happens when you upload audio?

```
1️⃣ LOAD AUDIO
   ↓ librosa.load() at 22050 Hz
   
2️⃣ HARMONIC-PERCUSSIVE SEPARATION
   ↓ Isolate voice/violin (harmonic)
   ↓ Remove mridangam (percussive)
   
3️⃣ BANDPASS FILTER (80-1800 Hz)
   ↓ Keep melodic frequency range
   ↓ Remove noise outside this range
   
4️⃣ EXTRACT MEL-SPECTROGRAM
   ↓ Time-frequency representation
   ↓ Shape: 128x128 (height x width)
   
5️⃣ NORMALIZE & RESIZE
   ↓ Scale to [0, 1]
   ↓ Fixed 128x128 dimensions
   
6️⃣ CNN LAYERS
   ↓ Extract spatial patterns
   ↓ Learn gamakas, phrases
   
7️⃣ LSTM LAYERS
   ↓ Capture temporal sequences
   ↓ Learn raga grammar
   
8️⃣ DENSE LAYERS
   ↓ Classification
   ↓ Output: Probability per raga
   
9️⃣ SOFTMAX
   ↓ Normalize to probabilities
   
🎯 RESULT: Closest Raga + Confidence
```

---

## 🔧 Advanced Features

### Custom Preprocessing
```python
from src.audio_processing import AudioPreprocessor

preprocessor = AudioPreprocessor(sample_rate=22050)

# Just HPS
audio_harmonic, audio_percussive = preprocessor.harmonic_percussive_separation(audio)

# Just bandpass
audio_filtered = preprocessor.apply_bandpass_filter(audio, lowcut=80, highcut=1800)

# Full pipeline with custom params
audio_clean = preprocessor.preprocess_for_raga_detection(
    audio,
    apply_hpss=True,
    apply_bandpass=True
)
```

### Extract Multiple Features
```python
from src.feature_extraction.advanced_features import AdvancedFeatureExtractor
from src.audio_processing import PitchDetector

feature_extractor = AdvancedFeatureExtractor(sample_rate=22050)
pitch_detector = PitchDetector(method='pyin')

# Extract pitch contour
pitch_contour = pitch_detector.detect_pitch_auto(audio_clean)

# Estimate tonic (simplified - needs better algorithm)
tonic_hz = np.median(pitch_contour['pitch'][pitch_contour['pitch'] > 0])

# Extract all features
features = feature_extractor.extract_all_features(
    audio_clean,
    pitch_contour=pitch_contour['pitch'],
    tonic_hz=tonic_hz
)

print("Available features:")
for key, value in features.items():
    print(f"  {key}: {type(value)}")

# Access specific features
mel_spec = features['mel_spectrogram']
cqt = features['cqt']
chroma = features['chroma']
tonal_hist = features['tonal_histogram']
swara_hist = features['swara_histogram']
pitch_stats = features['pitch_statistics']
```

### Use CREPE for Pitch Detection
```python
from src.audio_processing import PitchDetector

# Install first: pip install crepe
pitch_detector = PitchDetector(method='crepe')

pitch, confidence = pitch_detector.detect_pitch_crepe(audio_clean)

# Filter by confidence
reliable_pitch = pitch[confidence > 0.8]
```

---

## ⚡ Performance Tips

### For faster training:
- Reduce `batch_size` (if OOM errors)
- Reduce `epochs` (try 30 instead of 50)
- Use CPU if GPU unavailable (slower but works)

### For better accuracy:
- More training data (50+ songs per raga)
- Longer audio clips (30-60 seconds)
- Enable data augmentation (default: on)
- Try attention mechanism
- Experiment with different architectures

### For faster inference:
- Use smaller model (fewer CNN/LSTM layers)
- Reduce audio duration (15-20 seconds)
- Batch multiple predictions together

---

## 🐛 Common Issues

### "Model not found"
**Solution:** Train the model first
```bash
python src/model/train_crnn.py
```

### "CREPE not available"
**Solution:** Either install it or use pYIN (default)
```bash
pip install crepe  # Optional
```

### "Out of memory"
**Solution:** Reduce batch size
```python
trainer.train(..., batch_size=8)  # Instead of 16
```

### "Poor accuracy"
**Possible causes:**
1. Not enough training data (need 30+ songs per raga)
2. Data quality issues (low bitrate, noise)
3. Imbalanced dataset (unequal songs per raga)
4. Need more training epochs

---

## 📊 Monitoring Training

### Watch live progress:
```bash
# Start TensorBoard
tensorboard --logdir=models/logs

# Open: http://localhost:6006
```

### Check metrics:
- **Accuracy:** How often model predicts correctly
- **Loss:** Lower is better
- **Val_accuracy:** Most important (test on unseen data)
- **Top-5 accuracy:** How often true raga is in top 5

### Good signs:
- ✅ Training and validation accuracy both increasing
- ✅ Loss decreasing steadily
- ✅ Val_accuracy > 80% (for 5+ ragas)

### Bad signs:
- ❌ Val_accuracy plateaus or decreases (overfitting)
- ❌ Large gap between train and val accuracy (overfitting)
- ❌ Loss not decreasing (learning rate too high/low)

---

## 🎯 Next Steps

1. ✅ Collect more training data
2. ✅ Train your model on your dataset
3. ✅ Test via web UI
4. ✅ Integrate into your application
5. ✅ Fine-tune hyperparameters
6. ✅ Add more ragas as you collect data

---

## 📖 Documentation

- **Full ML Pipeline:** See `ML_PIPELINE.md`
- **Project Structure:** See `STRUCTURE.md`
- **Migration Guide:** See `MIGRATION.md`
- **API Docs:** http://localhost:8000/docs (when server running)

---

**Happy Raga Detecting! 🎵**
