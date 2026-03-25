# Migration Guide - Old to New Structure

## Summary of Changes

The project has been reorganized from a simple `src/ml/` structure to a comprehensive modular architecture.

## Old Structure → New Structure

### Audio Processing
- **Old**: Mixed in with training code
- **New**: `src/audio_processing/`
  - `preprocess.py` - Audio preprocessing functions
  - `pitch_detect.py` - Pitch detection and tonic identification
  - `smoothing.py` - Note stabilization and gamaka handling

### Feature Extraction
- **Old**: Not separated
- **New**: `src/feature_extraction/`
  - `extract_notes.py` - Arohanam/avarohanam extraction, pattern analysis

### Model Code
- **Old**: `src/ml/train_raga_classifier.py`, `src/ml/predict_raga.py`
- **New**: `src/model/`
  - `train_model.py` - Refactored training code
  - `match_raga.py` - Prediction and matching code

### Utilities
- **Old**: Not present
- **New**: `src/utils/`
  - `helpers.py` - Common utility functions

## Old Files Status

The old `src/ml/` directory contains:
- `train_raga_classifier.py` - **Replaced by** `src/model/train_model.py`
- `predict_raga.py` - **Replaced by** `src/model/match_raga.py`
- `quick_train.py` - Data augmentation functions now in `src/audio_processing/preprocess.py`
- `README.md` - Information merged into main `README.md`

## Migration Actions

### Option 1: Use New Structure (Recommended)

Simply use the new modular code:

```python
# Old way
from src.ml.predict_raga import RagaPredictor

# New way
from src.model.match_raga import RagaPredictor
```

### Option 2: Keep Both (Temporary)

Keep old files for reference during transition. The new code is functionally equivalent but better organized.

### Option 3: Delete Old Files

Once you're confident with the new structure:

```powershell
# Remove old ml directory
Remove-Item -Recurse -Force src/ml
```

## Key Improvements in New Structure

1. **Modular Design**: Each component has clear responsibility
2. **Better Testing**: Separate test files for each module
3. **API Support**: FastAPI backend ready to use
4. **Documentation**: Jupyter notebooks for experimentation
5. **Scalability**: Easy to add new features

## Updating Your Code

If you have existing scripts using the old structure:

### Old Code
```python
from src.ml.train_raga_classifier import RagaDataPreprocessor, train_raga_classifier
from src.ml.predict_raga import RagaPredictor
```

### New Code
```python
from src.model.train_model import RagaDataPreprocessor, train_raga_classifier
from src.model.match_raga import RagaPredictor
```

## Model Compatibility

The new training code produces **the same model format** as before:
- `raga_model.h5` (Keras model)
- `label_encoder.pkl` (Label encoder)

**Your existing models will work with the new code!**

## Data Migration

No data migration needed. The new structure uses the same `data/` directory:

```
data/
  ├── raw/              # Same as before
  ├── processed/        # New: for preprocessed audio
  └── notes/            # New: for extracted note sequences
```

## Quick Migration Steps

1. **Install new dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Test with existing model** (if you have one)
   ```python
   from src.model.match_raga import RagaPredictor
   predictor = RagaPredictor()  # Uses models/ directory
   ```

3. **Or train new model**
   ```powershell
   python src/model/train_model.py
   ```

4. **Remove old files** (when ready)
   ```powershell
   Remove-Item -Recurse src/ml
   ```

## Need Help?

- Check `QUICKSTART.md` for usage examples
- See `example_usage.py` for complete workflow
- Explore `notebooks/` for detailed analysis

## Rollback Plan

If you need to rollback:

1. Old files are still in `src/ml/`
2. Just use the old import paths
3. No data has been modified

## Summary

✅ **New structure is production-ready**
✅ **Backward compatible with existing models**
✅ **More features and better organized**
✅ **Old code preserved for reference**

**Recommendation**: Start using the new structure for all new work. Keep old files temporarily if needed for reference.
