"""
Test script for rule-based raga identification.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rule_based.identifier import RuleBasedRagaIdentifier

def test_on_audio_file(audio_path: str):
    """Test the identifier on a single audio file."""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(audio_path)}")
    print(f"{'='*60}\n")
    
    # Create identifier
    identifier = RuleBasedRagaIdentifier()
    
    # Identify raga
    predicted_raga, confidence_scores = identifier.identify_from_audio(audio_path, verbose=True)
    
    print(f"\n{'='*60}\n")

def test_on_directory(directory: str):
    """Test the identifier on all audio files in a directory."""
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(directory).glob(f"*{ext}"))
    
    print(f"Found {len(audio_files)} audio files in {directory}\n")
    
    identifier = RuleBasedRagaIdentifier()
    results = []
    
    for audio_file in audio_files:
        print(f"\n{'='*60}")
        print(f"Testing: {audio_file.name}")
        print(f"{'='*60}")
        
        predicted_raga, confidence_scores = identifier.identify_from_audio(str(audio_file), verbose=False)
        results.append({
            'file': audio_file.name,
            'predicted': predicted_raga,
            'confidence': confidence_scores.get(predicted_raga, 0) * 100
        })
        
        print(f"✅ Result: {predicted_raga} ({confidence_scores.get(predicted_raga, 0)*100:.1f}%)")
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(f"{result['file']:<40} -> {result['predicted']:<20} ({result['confidence']:.1f}%)")

if __name__ == "__main__":
    print("🎵 Rule-Based Raga Identifier Test")
    print("=" * 60)
    
    # Get supported ragas
    identifier = RuleBasedRagaIdentifier()
    print(f"\nSupported Ragas: {', '.join(identifier.get_supported_ragas())}\n")
    
    # Test on a specific file or directory
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            test_on_audio_file(path)
        elif os.path.isdir(path):
            test_on_directory(path)
        else:
            print(f"Error: {path} is not a valid file or directory")
    else:
        print("Usage: python test_identifier.py <audio_file_or_directory>")
        print("\nExample:")
        print("  python test_identifier.py ../data/curated/Shankarabharanam/")
        print("  python test_identifier.py ../data/curated/Mayamalavagowla/song.mp3")
