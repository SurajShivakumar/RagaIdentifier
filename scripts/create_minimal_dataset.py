"""
Create a minimal curated dataset with 15 high-quality files per raga
Select files that are clear vocal recordings, not too long, good quality
"""

import os
import shutil
from pathlib import Path

# Create curated directory
curated_dir = Path("data/curated")
curated_dir.mkdir(exist_ok=True)

# Define manually selected files (avoiding corrupted ones, choosing varied lengths)
curated_files = {
    "Begada": [
        "01 Begada --Intha Chaalamu (Varnam) TM Krishan.mp3",  # 3.3 MB
        "01-Anudinamu-Begada-PSI.mp3",  # 8.8 MB
        "01-IntaChalamu - Begada.mp3",  # 3 MB
        "01-Inta_Chalamu-Begada-VKuppaiyer.mp3",  # 6 MB
        "01-maraciTluNDEdi-bEgaDA.mp3",  # 5.8 MB
        "02   Tanavari Tanamu.mp3",  # 2.9 MB
        "02-anudinamunu_kAvumayya-bEgaDA.mp3",  # 10.7 MB
        "02-MMI-01---Vallabha-Nayakasya---Begada.mp3",  # 6.5 MB
        "03-kalayAmi_raghurAmaM-bEgaDA-swAtitirunAL.mp3",  # 6.2 MB
        "03-Sankari Neeve-Begada.mp3",  # 6.6 MB
        "04 Nadopasana - Begada.mp3",  # 6.3 MB
        "07-Inamu-Begada-MDR.mp3",  # 2 MB
        "08-Lokavana Chathura-Begada.mp3",  # 1.9 MB
        "09-Kadaikan_vaithennai-Begada.mp3",  # 4.1 MB
        "12-Abhimanamennadu-Begada.mp3",  # 6.2 MB
    ],
    "Shankarabharanam": [
        "004-BakthiBhiksha-Sankarabaranam.mp3",  # 13.7 MB
        "01 Varnam - Sankarabaranm.mp3",  # 6.7 MB
        "01-muthukumAraIyanE-sankarAbharanam-rAmasVmi sivan.mp3",  # 17.1 MB
        "02-Ivaruku_juchinadi_Sankarabharanam_Thyagarajar.mp3",  # 14.9 MB
        "020 EMI NERAMU.mp3",  # 10.5 MB
        "03 Swararaga - Sankarabaranam.mp3",  # 4.1 MB
        "03-enduku-sankarabharanam-ARI.mp3",  # 16 MB
        "04 Sankarabaranam--Sri mahalakshmi.mp3",  # 9.6 MB
        "04 Tillana Sankarabharanam.mp3",  # 4.4 MB
        "04-SSI-MSS-KSN-Dakshinamurthe-Sankarabharanam.mp3",  # 15.5 MB
        "06-Akshayalinga-Sankarabharanam.mp3",  # 26.3 MB
        "06-Alarulu-Sankarabharanam.mp3",  # 20.9 MB
        "06a-EdutaNilichite--Sankarabaranam.mp3",  # 10.3 MB
        "08-Brova_Barama-Sankarabharanam-Adi-tyAgarAja.mp3",  # Need to check if exists
        "09-Emi Neremu-Sankarabharanam.mp3",  # Need to check if exists
    ]
}

# Copy files to curated directory
print("Creating minimal curated dataset...\n")
total_copied = 0

for raga, files in curated_files.items():
    raga_source = Path(f"data/raw/{raga}")
    raga_dest = curated_dir / raga
    raga_dest.mkdir(exist_ok=True)
    
    copied = 0
    print(f"=== {raga} ===")
    
    for filename in files:
        source_path = raga_source / filename
        
        if source_path.exists():
            dest_path = raga_dest / filename
            
            # Skip if already exists
            if not dest_path.exists():
                shutil.copy2(source_path, dest_path)
            
            size_mb = source_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename} ({size_mb:.1f} MB)")
            copied += 1
        else:
            print(f"  ❌ Not found: {filename}")
    
    print(f"Copied {copied}/{len(files)} files for {raga}\n")
    total_copied += copied

print(f"{'='*60}")
print(f"Total files in curated dataset: {total_copied}")
print(f"Location: {curated_dir.absolute()}")
print(f"{'='*60}")

# Create a summary file
summary = {
    "total_files": total_copied,
    "ragas": {raga: len(files) for raga, files in curated_files.items()},
    "note": "Manually selected high-quality files, avoiding corrupted files"
}

import json
with open(curated_dir / "dataset_info.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Dataset info saved to data/curated/dataset_info.json")
