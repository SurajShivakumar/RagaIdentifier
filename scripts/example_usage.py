"""
Example script demonstrating raga identification workflow
"""

import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.audio_processing import PitchDetector, AudioPreprocessor
from src.feature_extraction import NoteExtractor, RAGA_TEMPLATES
from src.model.match_raga import RagaPredictor, match_raga_by_notes
from src.utils.helpers import print_banner, get_audio_files
import librosa
import numpy as np


def analyze_audio_file(audio_path: str):
    """
    Complete raga analysis pipeline
    
    Args:
        audio_path: Path to audio file
    """
    print_banner("RAGA IDENTIFICATION DEMO")
    print(f"\nAnalyzing: {audio_path}\n")
    
    # Step 1: Load audio
    print("Step 1: Loading audio...")
    y, sr = librosa.load(audio_path, duration=30)
    print(f"  ✓ Loaded {len(y)/sr:.1f} seconds @ {sr} Hz")
    
    # Step 2: Detect pitch
    print("\nStep 2: Detecting pitch...")
    detector = PitchDetector(sample_rate=sr)
    pitch_contour = detector.detect_pitch_yin(y)
    smoothed_pitch = detector.smooth_pitch(pitch_contour)
    print(f"  ✓ Detected {np.sum(~np.isnan(smoothed_pitch))} pitch frames")
    
    # Step 3: Estimate tonic
    print("\nStep 3: Estimating tonic (Sa)...")
    tonic = detector.estimate_tonic(smoothed_pitch)
    print(f"  ✓ Tonic: {tonic:.2f} Hz")
    
    # Step 4: Extract notes
    print("\nStep 4: Extracting notes...")
    from src.audio_processing.smoothing import NoteStabilizer
    stabilizer = NoteStabilizer()
    
    stable_regions = stabilizer.detect_stable_regions(
        smoothed_pitch,
        hop_length=512,
        sample_rate=sr
    )
    
    note_sequence = []
    frame_duration = 512 / sr
    
    for start_idx, end_idx, mean_pitch in stable_regions:
        cents = 1200 * np.log2(mean_pitch / tonic)
        note_name, _ = detector.cents_to_note(cents)
        start_time = start_idx * frame_duration
        duration = (end_idx - start_idx) * frame_duration
        note_sequence.append((note_name, start_time, duration))
    
    print(f"  ✓ Extracted {len(note_sequence)} stable notes")
    
    # Step 5: Analyze patterns
    print("\nStep 5: Analyzing raga patterns...")
    extractor = NoteExtractor()
    analysis = extractor.analyze_note_patterns(note_sequence)
    
    print(f"  ✓ Unique notes: {', '.join(analysis['unique_notes'])}")
    print(f"  ✓ Scale type: {analysis['scale_type']}")
    print(f"  ✓ Arohanam: {' → '.join(analysis['arohanam'])}")
    print(f"  ✓ Avarohanam: {' → '.join(analysis['avarohanam'])}")
    
    # Step 6: Match with templates
    print("\nStep 6: Matching with known ragas...")
    notes = [note for note, _, _ in note_sequence]
    matches = match_raga_by_notes(notes, RAGA_TEMPLATES)
    
    print("\n  Top 5 Matches (Rule-Based):")
    for i, (raga, score) in enumerate(matches[:5], 1):
        bar = '█' * int(score * 20)
        print(f"    {i}. {raga:20s} {score:5.1%} {bar}")
    
    # Step 7: ML Prediction (if model available)
    print("\nStep 7: ML Prediction...")
    try:
        predictor = RagaPredictor()
        ml_results = predictor.predict(audio_path, top_k=5)
        
        print(f"\n  🎵 ML Predicted Raga: {ml_results['top_raga']}")
        print(f"     Confidence: {ml_results['confidence']:.1f}%")
        
        print("\n  Top 5 ML Predictions:")
        for i, pred in enumerate(ml_results['top_predictions'], 1):
            print(f"    {i}. {pred['raga']:20s} {pred['confidence']:5.1f}%")
    
    except FileNotFoundError:
        print("  ⚠ ML model not trained yet. Run: python src/model/train_model.py")
    except Exception as e:
        print(f"  ⚠ ML prediction error: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


def main():
    """Main function"""
    # Check for audio files
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("\nPlease create the directory and add audio files:")
        print("  data/raw/Shankarabharanam/song.mp3")
        return
    
    # Get all audio files
    audio_files = get_audio_files(data_dir, recursive=True)
    
    if not audio_files:
        print("❌ No audio files found in data/raw/")
        print("\nPlease add audio files organized by raga:")
        print("  data/raw/")
        print("    ├── Shankarabharanam/")
        print("    │   └── song1.mp3")
        print("    └── Kalyani/")
        print("        └── song1.mp3")
        return
    
    print(f"Found {len(audio_files)} audio file(s)")
    
    # Analyze first file
    sample_file = audio_files[0]
    print(f"\nAnalyzing sample file: {sample_file}\n")
    
    try:
        analyze_audio_file(sample_file)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
