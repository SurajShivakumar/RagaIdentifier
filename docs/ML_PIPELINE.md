# 🎵 Advanced ML Pipeline for Raga Detection

## Overview

This implementation uses state-of-the-art machine learning techniques specifically designed for **Carnatic music raga identification**. The pipeline addresses the unique challenges of concert recordings with multiple instruments (tanpura, violin, mridangam, etc.) and complex gamakas.

---

## 🔊 Pipeline Architecture

```
Audio Input
    ↓
[1] HARMONIC-PERCUSSIVE SEPARATION (HPS)
    ├─ Harmonic: voice, violin, flute, tanpura
    └─ Percussive: mridangam, kanjira (discarded)
    ↓
[2] BANDPASS FILTER (80-1800 Hz)
    └─ Isolates melodic frequency range
    ↓
[3] FEATURE EXTRACTION
    ├─ Mel-Spectrogram (for CNN)
    ├─ CQT (Constant-Q Transform)
    ├─ Tonal Histogram
    ├─ Swara Histogram
    └─ Pitch Statistics (gamakas, vibrato)
    ↓
[4] CRNN MODEL (CNN + LSTM)
    ├─ CNN: Extract time-frequency patterns
    └─ LSTM: Capture sequential phrases
    ↓
[5] OUTPUT: Raga Probabilities
    └─ Closest Raga with confidence score
```

---

## 🧩 Key Components

### 1. **Audio Preprocessing** (`src/audio_processing/preprocess.py`)

#### Harmonic-Percussive Separation (HPS)
```python
audio_harmonic, audio_percussive = librosa.effects.hpss(audio, margin=2.0)
```
- **Why?** Carnatic concerts have loud mridangam that confuses pitch detection
- **Result:** Isolates melodic content (voice, violin) from percussion
- **Impact:** Dramatically improves accuracy on live recordings

#### Bandpass Filter (80-1800 Hz)
```python
filtered = apply_bandpass_filter(audio, lowcut=80, highcut=1800)
```
- **Why?** Melodic instruments sit in this frequency range
- **Removes:** Low-frequency rumble, high-frequency noise
- **Preserves:** Tanpura (100-150 Hz), Voice/Violin (200-2000 Hz)

### 2. **Pitch Detection** (`src/audio_processing/pitch_detect.py`)

Three algorithms available:

#### **pYIN** (Default - Best for Gamakas)
```python
pitch_detector = PitchDetector(method='pyin')
pitch_contour = pitch_detector.detect_pitch_pyin(audio)
```
- Probabilistic YIN algorithm
- Robust for continuous pitch tracking
- Handles fast oscillations (gamakas) well

#### **CREPE** (State-of-the-art)
```python
pitch_detector = PitchDetector(method='crepe')
pitch, confidence = pitch_detector.detect_pitch_crepe(audio)
```
- CNN-based pitch detection
- Best accuracy for complex ornamentations
- Requires: `pip install crepe`

### 3. **Feature Extraction** (`src/feature_extraction/advanced_features.py`)

#### Mel-Spectrogram (CNN Input)
- Time-frequency heatmap
- Captures timbral and temporal patterns
- Standard input for audio CNNs

#### CQT (Constant-Q Transform)
- Logarithmic frequency scale
- Aligns with musical notes
- Better than Mel for pitch-based music

#### Tonal Histogram
- Distribution of pitch classes
- Shows which notes are used and how often
- Distinguishes pentatonic vs heptatonic ragas

#### Swara Histogram
```python
swara_probs = extract_swara_histogram(pitch_contour, tonic_hz)
# {'S': 0.25, 'R2': 0.15, 'G3': 0.18, 'M1': 0.12, ...}
```
- Maps pitches to Carnatic swaras
- Probability distribution of note usage
- Key feature for raga classification

### 4. **CRNN Model** (`src/model/crnn_model.py`)

#### Architecture
```
Input: Mel-Spectrogram (128x128x1)
    ↓
CNN Block (3 layers)
    Conv2D(32) → BatchNorm → MaxPool → Dropout
    Conv2D(64) → BatchNorm → MaxPool → Dropout
    Conv2D(128) → BatchNorm → MaxPool → Dropout
    ↓
Reshape for LSTM
    ↓
LSTM Block (2 layers)
    LSTM(128) → BatchNorm
    LSTM(64) → BatchNorm
    ↓
Dense Classification
    Dense(128) → Dropout
    Dense(64)
    ↓
Output: Softmax(num_ragas)
```

**Why CNN + LSTM?**
- **CNN:** Learns local time-frequency patterns (gamakas, characteristic phrases)
- **LSTM:** Captures long-range dependencies (phrase sequences, raga grammar)
- **Combined:** Best of both worlds for raga detection

#### With Attention (Optional)
```python
crnn.build_attention_model()
```
- Attention mechanism focuses on important phrases
- Helps model learn which parts matter most
- Improves interpretability

---

## 🚀 Usage

### Training a Model

1. **Organize your data:**
```
data/raw/
    Bhairavi/
        song1.mp3
        song2.mp3
    Kalyani/
        song1.mp3
        song2.mp3
    Todi/
        ...
```

2. **Train the CRNN:**
```python
python src/model/train_crnn.py
```

3. **Or use the API:**
```python
from src.model.train_crnn import RagaTrainer

trainer = RagaTrainer(data_dir='data/raw', model_dir='models')
dataset = trainer.load_dataset(test_size=0.2, val_size=0.1)
model, history = trainer.train(
    dataset=dataset,
    model_type='crnn',
    epochs=50,
    batch_size=16,
    use_attention=False
)
```

### Making Predictions

#### Via Web UI
1. Start server: `python -m uvicorn api.app:app --reload`
2. Open: http://localhost:8000
3. Drag and drop audio file
4. Get instant raga prediction!

#### Via Python
```python
from src.audio_processing import AudioPreprocessor
from src.feature_extraction.advanced_features import AdvancedFeatureExtractor
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model('models/raga_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Preprocess audio
preprocessor = AudioPreprocessor(sample_rate=22050, duration=30)
audio, sr = preprocessor.load_audio('test_audio.mp3')
audio_clean = preprocessor.preprocess_for_raga_detection(audio)

# Extract features
feature_extractor = AdvancedFeatureExtractor(sample_rate=22050)
mel_spec = feature_extractor.extract_mel_spectrogram(audio_clean)
mel_spec = feature_extractor.prepare_for_cnn(mel_spec, (128, 128))
mel_spec = mel_spec[np.newaxis, ..., np.newaxis]  # Add batch & channel dims

# Predict
probs = model.predict(mel_spec)[0]
predicted_raga = label_encoder.classes_[np.argmax(probs)]
confidence = probs.max()

print(f"Predicted Raga: {predicted_raga} ({confidence:.2%} confidence)")
```

---

## 📊 Why ML Works Better Than Rule-Based Algorithms

### Problems with Classical Algorithms:

1. **Scale-based matching fails** - Ragas with same notes (aroha-avaroha) differ in phrasing
2. **Gamaka confusion** - Oscillations misinterpreted as wrong notes
3. **Mridangam interference** - Percussion transients corrupt pitch detection
4. **Phrase insensitivity** - Can't capture characteristic sangatis
5. **Variation intolerance** - Human performance variations break rules

### ML Advantages:

✅ **Learns gamaka shapes** - CNN captures oscillation patterns  
✅ **Phrase-aware** - LSTM learns sequential dependencies  
✅ **Noise-robust** - HPS removes percussion automatically  
✅ **Handles variations** - Data augmentation teaches robustness  
✅ **Multi-instrument** - Works with voice, violin, flute blend  
✅ **Statistical patterns** - Learns what's characteristic, not just allowed

---

## 🎯 Performance Optimization

### Data Augmentation (Applied During Training)
- **Time stretch:** 0.9x - 1.1x speed
- **Pitch shift:** ±2 semitones
- **Noise addition:** Simulates recording quality
- **Volume adjustment:** 0.7x - 1.3x amplitude

### Regularization Techniques
- **Dropout:** 0.3 (prevents overfitting)
- **Batch Normalization:** Stabilizes training
- **Early Stopping:** Stops when validation plateaus
- **Learning Rate Reduction:** Adaptive learning

### Model Checkpointing
```python
ModelCheckpoint(
    'models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)
```

---

## 📦 Dependencies

### Core
```bash
pip install librosa>=0.10.0
pip install tensorflow>=2.13.0
pip install scikit-learn>=1.3.0
pip install scipy>=1.11.0
pip install numpy>=1.24.0
```

### Optional (for better pitch detection)
```bash
pip install crepe  # CNN-based pitch detector
```

### API
```bash
pip install fastapi>=0.104.0
pip install uvicorn[standard]>=0.24.0
pip install python-multipart>=0.0.6
```

---

## 🔬 Model Variants

### 1. **CRNN (Recommended)**
- Best accuracy for raga detection
- Captures both patterns and phrases
- Training time: ~2-3 hours on GPU

### 2. **CRNN with Attention**
- Interpretable (shows important parts)
- Slightly better on complex ragas
- Training time: ~3-4 hours on GPU

### 3. **CNN Baseline**
- Faster training (~1 hour)
- Good for quick experiments
- Lower accuracy than CRNN

---

## 🎓 Further Improvements

### Short-term:
- [ ] Add transformer-based model
- [ ] Implement tonic tracking
- [ ] Add phrase extraction module
- [ ] Multi-task learning (raga + tonic)

### Long-term:
- [ ] Real-time streaming detection
- [ ] Gamaka classification
- [ ] Raga improvisation generation
- [ ] Multi-label (multiple ragas in one recording)

---

## 📚 References

1. **Harmonic-Percussive Separation:**
   - Driedger et al., "Median/Average Filtering" (2014)
   
2. **CREPE Pitch Detection:**
   - Kim et al., "CREPE: A Convolutional Representation for Pitch Estimation" (2018)

3. **CRNN for Audio:**
   - Choi et al., "Convolutional Recurrent Neural Networks for Music Classification" (2017)

4. **Attention Mechanisms:**
   - Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)

---

## 🐛 Troubleshooting

### Model not loading?
- Check paths: `models/raga_model.h5` and `models/label_encoder.pkl`
- Train first: `python src/model/train_crnn.py`

### Poor accuracy?
- More training data needed (aim for 50+ songs per raga)
- Enable data augmentation
- Try different model architectures
- Check audio quality (low bitrate affects features)

### Out of memory?
- Reduce batch_size in training
- Use shorter audio duration (15-20 seconds)
- Smaller model (fewer CNN filters, LSTM units)

---

## 💡 Tips for Best Results

1. **Clean data:** Remove intros, outros, applause
2. **Consistent quality:** Similar bitrate across recordings
3. **Balanced dataset:** Equal songs per raga
4. **Validate split:** Test on different artists than training
5. **Hyperparameter tuning:** Experiment with learning rate, dropout

---

**Built with ❤️ for Carnatic music lovers and ML enthusiasts!**
