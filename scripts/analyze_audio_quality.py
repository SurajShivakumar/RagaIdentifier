"""
Analyze audio quality and identify low-quality recordings
"""
import os
import numpy as np
import librosa
from pathlib import Path
import json

DATA_DIR = "data/raw"
OUTPUT_FILE = "audio_quality_report.json"

def analyze_audio_quality(audio_path):
    """
    Analyze audio file quality metrics
    
    Returns:
        dict with quality metrics
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None, duration=None)
        
        # Duration
        duration = len(y) / sr
        
        # RMS Energy (loudness)
        rms = np.sqrt(np.mean(y**2))
        
        # Zero crossing rate (noise indicator)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Spectral centroid (brightness/quality)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Check for silence/corruption (too many zeros)
        zero_percentage = np.sum(y == 0) / len(y)
        
        # Signal to noise ratio estimate
        # Use spectral flatness - higher = more noise-like
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Dynamic range
        dynamic_range = np.max(np.abs(y)) - np.min(np.abs(y))
        
        return {
            "duration": float(duration),
            "rms_energy": float(rms),
            "zero_crossing_rate": float(zcr),
            "spectral_centroid": float(spectral_centroid),
            "zero_percentage": float(zero_percentage),
            "spectral_flatness": float(spectral_flatness),
            "dynamic_range": float(dynamic_range),
            "sample_rate": int(sr),
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def analyze_all_ragas():
    """Analyze all audio files and identify low quality ones"""
    
    results = {}
    low_quality_files = []
    
    print("=" * 70)
    print("AUDIO QUALITY ANALYSIS")
    print("=" * 70)
    
    for raga in os.listdir(DATA_DIR):
        raga_path = os.path.join(DATA_DIR, raga)
        if not os.path.isdir(raga_path):
            continue
        
        print(f"\n📁 Analyzing {raga}...")
        
        audio_files = [f for f in os.listdir(raga_path) 
                      if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))]
        
        raga_results = []
        
        for audio_file in audio_files:
            audio_path = os.path.join(raga_path, audio_file)
            
            print(f"  Analyzing: {audio_file}...", end=" ")
            
            metrics = analyze_audio_quality(audio_path)
            metrics["filename"] = audio_file
            metrics["path"] = audio_path
            
            if metrics["success"]:
                print("✓")
            else:
                print(f"✗ Error: {metrics['error']}")
            
            raga_results.append(metrics)
        
        results[raga] = raga_results
        
        # Calculate statistics for this raga
        successful = [r for r in raga_results if r["success"]]
        
        if successful:
            # Calculate means and stds
            rms_values = [r["rms_energy"] for r in successful]
            duration_values = [r["duration"] for r in successful]
            flatness_values = [r["spectral_flatness"] for r in successful]
            zero_pct_values = [r["zero_percentage"] for r in successful]
            
            mean_rms = np.mean(rms_values)
            std_rms = np.std(rms_values)
            mean_duration = np.mean(duration_values)
            mean_flatness = np.mean(flatness_values)
            std_flatness = np.std(flatness_values)
            
            print(f"\n  📊 {raga} Statistics:")
            print(f"     Files: {len(successful)}")
            print(f"     Avg Duration: {mean_duration:.1f}s")
            print(f"     Avg RMS Energy: {mean_rms:.4f}")
            print(f"     Avg Spectral Flatness: {mean_flatness:.4f} (lower=better)")
            
            # Identify low quality files
            print(f"\n  ⚠️  Low Quality Files:")
            found_low_quality = False
            
            for r in successful:
                issues = []
                
                # Too quiet (more than 2 std below mean)
                if r["rms_energy"] < (mean_rms - 2 * std_rms):
                    issues.append("too quiet")
                
                # Too noisy (high spectral flatness)
                if r["spectral_flatness"] > (mean_flatness + 2 * std_flatness):
                    issues.append("too noisy")
                
                # Too short
                if r["duration"] < 20:
                    issues.append("too short (<20s)")
                
                # Too much silence
                if r["zero_percentage"] > 0.1:
                    issues.append("excessive silence")
                
                # Very low energy (corrupted?)
                if r["rms_energy"] < 0.001:
                    issues.append("extremely low energy")
                
                if issues:
                    found_low_quality = True
                    print(f"     - {r['filename']}")
                    print(f"       Issues: {', '.join(issues)}")
                    print(f"       RMS: {r['rms_energy']:.4f}, Duration: {r['duration']:.1f}s")
                    
                    low_quality_files.append({
                        "raga": raga,
                        "filename": r['filename'],
                        "path": r['path'],
                        "issues": issues,
                        "metrics": {
                            "rms_energy": r["rms_energy"],
                            "duration": r["duration"],
                            "spectral_flatness": r["spectral_flatness"],
                            "zero_percentage": r["zero_percentage"]
                        }
                    })
            
            if not found_low_quality:
                print(f"     ✓ All files have acceptable quality")
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            "full_analysis": results,
            "low_quality_files": low_quality_files
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY")
    print("=" * 70)
    print(f"Total low quality files found: {len(low_quality_files)}")
    print(f"\nFull report saved to: {OUTPUT_FILE}")
    
    if low_quality_files:
        print("\n⚠️  RECOMMENDATION:")
        print("Delete these low quality files:")
        print("\nPowerShell commands:")
        for item in low_quality_files:
            # Escape path for PowerShell
            path = item['path'].replace("\\", "\\")
            print(f'Remove-Item "{path}"  # {raga}: {", ".join(item["issues"])}')
    
    return results, low_quality_files


if __name__ == "__main__":
    analyze_all_ragas()
