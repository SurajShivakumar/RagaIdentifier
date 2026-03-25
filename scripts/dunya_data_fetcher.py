"""
Fetch Carnatic music data from Dunya API
Download compositions for specific ragas with proper metadata
"""

import requests
import os
import json
from typing import List, Dict
import time
import unicodedata
try:
    from config import TARGET_RAGAS, PATHS  # type: ignore
except Exception:
    TARGET_RAGAS = []
    PATHS = {"data_raw": "data/raw"}

class DunyaFetcher:
    def __init__(self, api_token: str, download_audio: bool = False):
        """
        Initialize Dunya API client
        
        Args:
            api_token: Your Dunya API token
        """
        self.api_token = api_token
        self.base_url = "https://dunya.compmusic.upf.edu/api/carnatic"
        self.headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json"
        }
        self.download_audio_enabled = download_audio
    
    def search_ragas(self, raga_name: str) -> List[Dict]:
        """Search for ragas by name across all pages"""
        url = f"{self.base_url}/raaga"
        matches: List[Dict] = []
        offset = 0
        page_size = 100

        while True:
            params = {"limit": page_size, "offset": offset}
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code != 200:
                print(f"Error fetching ragas: {response.status_code}")
                break

            data = response.json()
            results = data.get('results', [])
            if not results:
                break

            # Filter by name with unicode diacritics normalization
            def norm(s: str) -> str:
                s = unicodedata.normalize('NFD', s)
                s = ''.join(c for c in s if not unicodedata.combining(c))
                return s.lower()

            name_l = norm(raga_name)
            for r in results:
                rname = r.get('name', '')
                if name_l in norm(rname):
                    matches.append(r)

            if len(results) < page_size:
                break
            offset += page_size

        return matches
    
    def get_raga_recordings(self, raga_id: str, limit: int = 50) -> List[Dict]:
        """
        Get recordings for a specific raga
        
        Args:
            raga_id: Dunya raga UUID
            limit: Maximum number of recordings to fetch
        """
        recordings = []
        offset = 0
        
        while len(recordings) < limit:
            url = f"{self.base_url}/recording"
            params = {
                "raaga": raga_id,
                "limit": 100,
                "offset": offset
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"Error fetching recordings: {response.status_code}")
                break
            
            data = response.json()
            batch = data.get('results', [])
            
            if not batch:
                break
            
            recordings.extend(batch)
            offset += 100
            
            if len(batch) < 100:  # Last page
                break
        
        return recordings[:limit]
    
    def get_recording_details(self, recording_id: str) -> Dict:
        """Get detailed information about a recording"""
        url = f"{self.base_url}/recording/{recording_id}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print(f"Error fetching recording details: {response.status_code}")
            return {}
        
        return response.json()
    
    def download_audio(self, recording_id: str, output_path: str) -> bool:
        """
        Download audio file for a recording
        
        Args:
            recording_id: Dunya recording UUID
            output_path: Where to save the file
        """
        # Use the working endpoint (without /api prefix)
        audio_url = f"https://dunya.compmusic.upf.edu/carnatic/recording/{recording_id}/audio"
        
        # Respect flag: disable audio downloads unless explicitly enabled
        if not self.download_audio_enabled:
            return False

        # Download the audio
        try:
            response = requests.get(audio_url, headers=self.headers, stream=True)
            
            if response.status_code != 200:
                print(f"Error downloading audio: {response.status_code}")
                return False
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        
        except Exception as e:
            print(f"Error downloading {recording_id}: {e}")
            return False
    
    def fetch_raga_dataset(self, raga_name: str, output_dir: str, max_recordings: int = 50):
        """
        Fetch complete dataset for a raga
        
        Args:
            raga_name: Name of the raga (e.g., 'Shankarabharanam', 'Begada')
            output_dir: Directory to save audio files
            max_recordings: Maximum number of recordings to download
        """
        print(f"\n{'='*60}")
        print(f"Fetching data for raga: {raga_name}")
        print(f"{'='*60}")
        
        # Search for the raga
        print("Searching for raga...")
        ragas = self.search_ragas(raga_name)
        
        if not ragas:
            print(f"❌ No ragas found matching '{raga_name}'")
            return
        
        # Use the first match (or let user select)
        raga = ragas[0]
        print(f"✅ Found raga: {raga['name']} (ID: {raga['uuid']})")
        
        # Get recordings
        print(f"Fetching recordings (max {max_recordings})...")
        recordings = self.get_raga_recordings(raga['uuid'], limit=max_recordings)
        print(f"✅ Found {len(recordings)} recordings")
        
        # Create output directory
        raga_dir = os.path.join(output_dir, raga['name'])
        os.makedirs(raga_dir, exist_ok=True)
        
        # Download recordings (audio disabled by default; always save metadata)
        metadata = []
        successful = 0
        
        for i, recording in enumerate(recordings, 1):
            recording_id = recording['mbid']
            title = recording.get('title', f'Recording_{i}')
            
            # Clean filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
            filename = f"{i:03d}_{safe_title[:50]}.mp3"
            output_path = os.path.join(raga_dir, filename)
            
            # Skip if already downloaded
            if os.path.exists(output_path):
                print(f"[{i}/{len(recordings)}] ⏭️  Skipping (already exists): {title}")
                successful += 1
                continue
            
            if self.download_audio_enabled:
                print(f"[{i}/{len(recordings)}] ⬇️  Downloading: {title}")
            else:
                print(f"[{i}/{len(recordings)}] ⏭️  Skipping audio (metadata only): {title}")

            downloaded = self.download_audio(recording_id, output_path)
            if downloaded:
                successful += 1
                print(f"[{i}/{len(recordings)}] ✅ Downloaded: {filename}")

            metadata.append({
                'filename': filename if downloaded else None,
                'title': title,
                'recording_id': recording_id,
                'raga': raga['name']
            })
            
            # Rate limiting
            time.sleep(0.5)
        
        # Save metadata
        metadata_path = os.path.join(raga_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Downloaded {successful}/{len(recordings)} recordings to {raga_dir}")
        print(f"✅ Metadata saved to {metadata_path}")


def main():
    """Fetch Dunya data for target ragas (metadata only by default)"""
    # Prefer env var for token
    api_token = os.environ.get("DUNYA_API_TOKEN")

    if not api_token:
        print("❌ API token is required. Set DUNYA_API_TOKEN environment variable.")
        return

    # Initialize fetcher
    # Audio download is disabled by default; enable via DUNYA_DOWNLOAD_AUDIO=1
    download_flag = os.environ.get("DUNYA_DOWNLOAD_AUDIO") in {"1", "true", "True"}
    fetcher = DunyaFetcher(api_token, download_audio=download_flag)

    # Determine output directory under data/raw/dunya
    base_raw = PATHS.get("data_raw", "data/raw")
    output_dir = os.path.join(base_raw, "dunya")
    os.makedirs(output_dir, exist_ok=True)

    # Build ragas list from config; default to empty
    ragas_to_fetch = [(raga, 50) for raga in TARGET_RAGAS] if TARGET_RAGAS else []

    if not ragas_to_fetch:
        print("⚠️ No target ragas found. Update config.TARGET_RAGAS or pass list.")
        return

    # Fetch data for each raga
    for raga_name, max_recordings in ragas_to_fetch:
        fetcher.fetch_raga_dataset(raga_name, output_dir, max_recordings)

    print(f"\n{'='*60}")
    print("✅ Fetch complete (metadata)")
    print(f"Data saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
