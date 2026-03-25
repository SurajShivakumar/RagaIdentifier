"""
Identify corrupted or problematic audio files
"""
import os
import glob
import librosa
import warnings
warnings.filterwarnings('ignore')

def test_audio_file(file_path):
    """Test if an audio file can be loaded"""
    try:
        # Try to load the file
        audio, sr = librosa.load(file_path, sr=22050, duration=5.0)
        
        # Check if we got valid audio
        if len(audio) < 22050:  # Less than 1 second
            return False, "Too short"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)

# Test all audio files
data_dir = 'data/raw'
ragas = ['Begada', 'Shankarabharanam']

corrupted_files = []
good_files = []

for raga in ragas:
    raga_dir = os.path.join(data_dir, raga)
    if not os.path.exists(raga_dir):
        continue
    
    files = glob.glob(os.path.join(raga_dir, '*.mp3'))
    print(f"\nTesting {raga} ({len(files)} files)...")
    
    for file_path in files:
        is_valid, message = test_audio_file(file_path)
        
        if not is_valid:
            corrupted_files.append((file_path, message))
            print(f"  ❌ {os.path.basename(file_path)}: {message}")
        else:
            good_files.append(file_path)

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"  ✅ Good files: {len(good_files)}")
print(f"  ❌ Corrupted files: {len(corrupted_files)}")
print(f"{'='*60}")

if corrupted_files:
    print("\nCorrupted files to delete:")
    for file_path, error in corrupted_files:
        print(f"  {file_path}")
    
    print("\nDelete commands:")
    for file_path, error in corrupted_files:
        print(f'Remove-Item "{file_path}"')
