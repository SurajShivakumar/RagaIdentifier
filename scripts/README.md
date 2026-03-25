# Utility Scripts

This directory contains various utility scripts for data management and analysis.

## Data Collection
- `dunya_data_fetcher.py` - Fetch raga metadata from Dunya API (audio requires web auth)
- `list_dunya_ragas.py` - List all available ragas in Dunya database

## Data Quality
- `analyze_audio_quality.py` - Identify low-quality or corrupted audio files
- `identify_corrupted_files.py` - Check for corrupted MP3 files
- `curate_dataset.py` - Interactive tool to manually curate dataset
- `create_minimal_dataset.py` - Create a minimal curated dataset with selected files

## Setup & Examples
- `setup_data.py` - Data setup utilities
- `example_usage.py` - Example workflow

## Testing & Debugging
- `test_dunya_api.py` - Test Dunya API endpoints
- `test_audio_download.py` - Test audio download functionality

## Usage

Most scripts can be run directly:
```bash
python scripts/analyze_audio_quality.py
python scripts/create_minimal_dataset.py
```
