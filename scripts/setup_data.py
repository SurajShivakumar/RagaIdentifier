"""
Data Setup Helper for 5 Target Ragas
Checks data structure and provides guidance
"""
import os
from pathlib import Path
from config import TARGET_RAGAS, PATHS, RAGA_INFO

def create_data_structure():
    """Create folder structure for the 5 ragas"""
    base_path = Path(PATHS["data_raw"])
    
    print("\n📁 Creating data structure...")
    print("=" * 60)
    
    for raga in TARGET_RAGAS:
        raga_path = base_path / raga
        raga_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {raga_path}")
    
    print("\n✅ Data structure ready!")

def check_data_status():
    """Check how many audio files are in each raga folder"""
    base_path = Path(PATHS["data_raw"])
    
    print("\n📊 DATA STATUS REPORT")
    print("=" * 60)
    
    total_files = 0
    status = []
    
    for raga in TARGET_RAGAS:
        raga_path = base_path / raga
        
        if not raga_path.exists():
            count = 0
            status_emoji = "❌"
        else:
            # Count audio files
            audio_files = []
            for ext in ['.mp3', '.wav', '.m4a', '.flac']:
                audio_files.extend(raga_path.glob(f'*{ext}'))
            count = len(audio_files)
            total_files += count
            
            if count == 0:
                status_emoji = "⚠️ "
            elif count < 20:
                status_emoji = "🟡"
            else:
                status_emoji = "✅"
        
        status.append((raga, count, status_emoji))
        print(f"{status_emoji} {raga:20s} {count:3d} files")
    
    print("=" * 60)
    print(f"📈 TOTAL: {total_files} audio files across {len(TARGET_RAGAS)} ragas")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if total_files == 0:
        print("❌ No audio files found!")
        print("   → Add audio files to: data/raw/<RagaName>/")
    elif total_files < 50:
        print("⚠️  Limited training data detected")
        print(f"   → Current: {total_files} files")
        print(f"   → Recommended: 100+ files (20 per raga)")
    elif total_files < 100:
        print("🟡 Moderate training data")
        print(f"   → Current: {total_files} files")
        print(f"   → For better results: 150+ files (30 per raga)")
    else:
        print(f"✅ Good training data: {total_files} files")
        print("   → Ready to train!")
    
    # Check balance
    counts = [c for _, c, _ in status]
    if counts and max(counts) > 0:
        min_count = min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 0
        max_count = max(counts)
        if max_count > min_count * 2:
            print("\n⚠️  IMBALANCED DATASET DETECTED")
            print("   → Some ragas have much more data than others")
            print("   → This may affect accuracy")
            print("   → Recommendation: Balance the dataset")
    
    return status, total_files

def print_data_collection_guide():
    """Print guide for collecting audio data"""
    print("\n" + "=" * 60)
    print("📚 DATA COLLECTION GUIDE")
    print("=" * 60)
    
    print("\n🎯 TARGET: 20-30 audio clips per raga (1-3 minutes each)")
    print("\n📁 Place files in:")
    print("   data/raw/<RagaName>/")
    print("   └── Example: data/raw/Kalyani/concert1.mp3")
    
    print("\n🎵 Where to find recordings:")
    print("   1. YouTube (search: '<raga> concert', '<raga> alapana')")
    print("   2. Personal collection of concert recordings")
    print("   3. Carnatic music archives (CompMusic, Saraga)")
    
    print("\n✅ Audio quality tips:")
    print("   • Clean recordings preferred (less background noise)")
    print("   • Concert recordings work well")
    print("   • Mix of alapanas, compositions, neravals")
    print("   • Multiple artists = better generalization")
    print("   • Vocal or instrumental (violin, flute) both work")
    
    print("\n🎼 Target Ragas Info:")
    for raga in TARGET_RAGAS:
        info = RAGA_INFO[raga]
        print(f"\n   {raga}:")
        print(f"      Notes: {info['arohanam']}")
        print(f"      Key: {info['description']}")
        if info['characteristic_note']:
            print(f"      Distinguish by: {info['characteristic_note']}")
    
    print("\n💡 PRO TIP:")
    print("   Use yt-dlp to download audio from YouTube:")
    print('   yt-dlp -x --audio-format mp3 -o "data/raw/Kalyani/%(title)s.%(ext)s" "URL"')
    
    print("=" * 60)

def main():
    """Main function"""
    print("\n🎵 RAGA IDENTIFIER - DATA SETUP")
    
    # Create structure
    create_data_structure()
    
    # Check status
    status, total = check_data_status()
    
    # Print guide
    print_data_collection_guide()
    
    # Next steps
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS:")
    print("=" * 60)
    if total == 0:
        print("1. Collect audio files for each raga")
        print("2. Place them in data/raw/<RagaName>/")
        print("3. Re-run this script to check status")
        print("4. When ready, train: python src/model/train_crnn.py")
    elif total < 50:
        print("1. Add more audio files (aim for 100+ total)")
        print("2. Re-run to check: python setup_data.py")
        print("3. Train with current data: python src/model/train_crnn.py")
    else:
        print("✅ Data looks good!")
        print("1. Train the model: python src/model/train_crnn.py")
        print("2. Start API: uvicorn api.app:app --reload")
        print("3. Open: http://localhost:8000")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
