"""
Test Dunya API endpoints to find the correct way to download audio
"""

import requests
import json

api_token = "de0d218c94386e22087702321841805dbaf7011b"
base_url = "https://dunya.compmusic.upf.edu/api/carnatic"
headers = {
    "Authorization": f"Token {api_token}",
    "Content-Type": "application/json"
}

# Get a sample recording to inspect the structure
print("Fetching sample recording details...")
url = f"{base_url}/recording"
params = {"raaga": "f972db4d-5d16-4f9a-9841-f313e1601aaa", "limit": 1}
response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    recordings = data.get('results', [])
    
    if recordings:
        recording = recordings[0]
        mbid = recording['mbid']
        print(f"\nSample recording: {recording.get('title', 'Unknown')}")
        print(f"MBID: {mbid}")
        
        # Get detailed info
        detail_url = f"{base_url}/recording/{mbid}"
        detail_response = requests.get(detail_url, headers=headers)
        
        if detail_response.status_code == 200:
            details = detail_response.json()
            print("\nRecording details structure:")
            print(json.dumps(details, indent=2))
            
            # Try different audio endpoints
            print("\n" + "="*60)
            print("Testing audio download endpoints:")
            print("="*60)
            
            test_urls = [
                f"{base_url}/recording/{mbid}/audio",
                f"https://dunya.compmusic.upf.edu/carnatic/recording/{mbid}/audio",
                f"https://dunya.compmusic.upf.edu/document/{mbid}/audio",
            ]
            
            for test_url in test_urls:
                print(f"\nTrying: {test_url}")
                r = requests.head(test_url, headers=headers)
                print(f"  Status: {r.status_code}")
                if r.status_code == 200:
                    print(f"  ✅ This endpoint works!")
                    print(f"  Content-Type: {r.headers.get('Content-Type')}")
                    print(f"  Content-Length: {r.headers.get('Content-Length')}")
        else:
            print(f"Error getting details: {detail_response.status_code}")
else:
    print(f"Error: {response.status_code}")
