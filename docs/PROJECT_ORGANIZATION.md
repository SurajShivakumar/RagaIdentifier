# Project Organization Summary

## Recent Changes

### Created `scripts/` Directory
Moved all utility scripts to a dedicated folder:
- Data collection scripts (Dunya API)
- Data quality analysis tools
- Dataset curation utilities
- Testing scripts

### Created `data/curated/` Directory
Minimal high-quality dataset for testing:
- **Begada**: 15 files (~74 MB)
- **Shankarabharanam**: 13 files (~160 MB)
- Total: **28 carefully selected files**

### Updated Training Configuration
- Using `data/curated/` instead of `data/raw/`
- Reduced augmentation (minimal volume adjustment only)
- Smaller batch size (4) for small dataset
- More epochs (20) for better learning

### Cleaned Up
- Removed temporary HTML files
- Removed incomplete Dunya downloads (audio requires web auth)
- Organized test scripts

## Current Project Structure

```
raga-identifier/
├── api/                    # Web server + UI
├── data/
│   ├── raw/               # Original files (66 Begada, 91 Shankarabharanam)
│   └── curated/           # Minimal dataset (15 Begada, 13 Shankarabharanam)
├── models/                # Trained models (.h5 files)
├── scripts/               # Utility scripts (NEW)
├── src/                   # Core source code
│   ├── audio_processing/
│   ├── feature_extraction/
│   ├── model/            # CRNN architecture + training
│   └── utils/
└── tests/                 # Unit tests

## Next Steps

1. **Train on curated dataset**:
   ```bash
   python src/model/train_crnn.py
   ```

2. **Monitor training**: Watch for validation accuracy > 70%

3. **Test predictions**:
   ```bash
   python tests/test_model_predictions.py
   ```

4. **Start web server**:
   ```bash
   uvicorn api.app:app --reload
   ```

## Key Files

- `src/model/train_crnn.py` - Training pipeline (configured for curated data)
- `src/model/crnn_model.py` - Neural network architecture
- `scripts/create_minimal_dataset.py` - Creates curated dataset
- `api/app.py` - FastAPI server with web UI
- `tests/test_model_predictions.py` - Test model accuracy

## Notes

- Curated dataset is small but clean - better for testing architecture
- Augmentation disabled to preserve raga characteristics
- If model works (>70% accuracy), expand dataset gradually
- Dunya API provides metadata only, not audio files
