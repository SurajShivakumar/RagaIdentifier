# Models Directory

This directory stores trained machine learning models.

## Files

- `raga_model.h5` - Trained CNN model (created after training)
- `label_encoder.pkl` - Label encoder for raga names (created after training)

## Training

To train the model:

```powershell
python src/model/train_model.py
```

The model will be saved here automatically.

## Usage

```python
from src.model.match_raga import RagaPredictor

predictor = RagaPredictor()
results = predictor.predict('audio.mp3')
```
