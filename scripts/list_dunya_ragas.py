"""
List all available ragas in Dunya Carnatic database
"""

import requests
import json

api_token = "de0d218c94386e22087702321841805dbaf7011b"
base_url = "https://dunya.compmusic.upf.edu/api/carnatic"
headers = {
    "Authorization": f"Token {api_token}",
    "Content-Type": "application/json"
}

print("Fetching all ragas from Dunya...")
url = f"{base_url}/raaga"
response = requests.get(url, headers=headers)

if response.status_code != 200:
    print(f"Error: {response.status_code}")
    print(response.text)
else:
    data = response.json()
    ragas = data.get('results', [])
    
    print(f"\nFound {len(ragas)} ragas in database\n")
    print("Available ragas:")
    print("=" * 60)
    
    # Sort by name
    ragas_sorted = sorted(ragas, key=lambda x: x.get('name', ''))
    
    for raga in ragas_sorted:
        name = raga.get('name', 'Unknown')
        uuid = raga.get('uuid', 'N/A')
        print(f"  {name} (ID: {uuid})")
    
    # Look for our target ragas
    print("\n" + "=" * 60)
    print("Searching for target ragas:")
    print("=" * 60)
    
    targets = ['shankar', 'begada', 'kalyani', 'todi', 'bhairavi', 'mohanam', 'mayamalavagowla']
    
    for target in targets:
        matches = [r for r in ragas if target.lower() in r.get('name', '').lower()]
        if matches:
            print(f"\n'{target}' matches:")
            for match in matches:
                print(f"  - {match['name']} (ID: {match['uuid']})")
