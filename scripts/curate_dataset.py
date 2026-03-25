"""
Interactive data curation tool
Helps you listen to files and verify they're correctly labeled
"""

import os
import glob
import json
from pathlib import Path

def curate_dataset():
    """Interactive tool to curate raga dataset"""
    
    data_dir = "data/raw"
    ragas = [d for d in os.listdir(data_dir) 
             if os.path.isdir(os.path.join(data_dir, d)) and d != 'ARCHIVE']
    
    print("=" * 60)
    print("DATASET CURATION TOOL")
    print("=" * 60)
    print("\nThis tool will help you:")
    print("1. Review each raga folder")
    print("2. Identify files to keep or remove")
    print("3. Create a curated subset for training")
    print()
    
    curated_files = {}
    
    for raga in ragas:
        raga_dir = os.path.join(data_dir, raga)
        files = glob.glob(os.path.join(raga_dir, '*.mp3'))
        
        print(f"\n{'='*60}")
        print(f"RAGA: {raga}")
        print(f"{'='*60}")
        print(f"Total files: {len(files)}")
        print()
        
        # Show all files with indices
        for i, file_path in enumerate(files, 1):
            filename = os.path.basename(file_path)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  [{i:2d}] {filename} ({size_mb:.1f} MB)")
        
        print()
        print("Options:")
        print("  k - Keep ALL files from this raga")
        print("  s - SELECT specific files to keep (enter numbers separated by spaces)")
        print("  r - REMOVE specific files (enter numbers separated by spaces)")
        print("  skip - Skip this raga entirely")
        
        choice = input(f"\nYour choice for {raga}: ").strip().lower()
        
        if choice == 'k':
            curated_files[raga] = files
            print(f"✅ Keeping all {len(files)} files from {raga}")
        
        elif choice == 's':
            indices_str = input("Enter file numbers to KEEP (e.g., 1 3 5-10 15): ").strip()
            selected_indices = parse_range(indices_str)
            selected_files = [files[i-1] for i in selected_indices if 1 <= i <= len(files)]
            curated_files[raga] = selected_files
            print(f"✅ Selected {len(selected_files)} files from {raga}")
        
        elif choice == 'r':
            indices_str = input("Enter file numbers to REMOVE (e.g., 1 3 5-10 15): ").strip()
            remove_indices = parse_range(indices_str)
            kept_files = [f for i, f in enumerate(files, 1) if i not in remove_indices]
            curated_files[raga] = kept_files
            print(f"✅ Keeping {len(kept_files)} files from {raga} (removed {len(remove_indices)})")
        
        elif choice == 'skip':
            print(f"⏭️  Skipping {raga}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("CURATION SUMMARY")
    print(f"{'='*60}")
    
    total_files = 0
    for raga, files in curated_files.items():
        print(f"  {raga}: {len(files)} files")
        total_files += len(files)
    
    print(f"\nTotal curated files: {total_files}")
    
    # Save curation config
    save_choice = input("\nSave this configuration? (y/n): ").strip().lower()
    if save_choice == 'y':
        config = {
            'ragas': {raga: [os.path.basename(f) for f in files] 
                     for raga, files in curated_files.items()}
        }
        
        with open('data/curation_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration saved to data/curation_config.json")
        
        # Option to copy curated files to new directory
        copy_choice = input("\nCopy curated files to data/curated? (y/n): ").strip().lower()
        if copy_choice == 'y':
            import shutil
            for raga, files in curated_files.items():
                curated_dir = os.path.join('data/curated', raga)
                os.makedirs(curated_dir, exist_ok=True)
                
                for file_path in files:
                    dest_path = os.path.join(curated_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)
            
            print(f"✅ Curated files copied to data/curated/")

def parse_range(range_str):
    """Parse range string like '1 3 5-10 15' into list of integers"""
    indices = set()
    parts = range_str.split()
    
    for part in parts:
        if '-' in part:
            start, end = part.split('-')
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    
    return sorted(indices)


if __name__ == '__main__':
    curate_dataset()
