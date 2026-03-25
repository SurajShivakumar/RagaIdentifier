"""
Test downloading audio from Dunya - check HTML response for actual audio URL
"""

import requests
from bs4 import BeautifulSoup

api_token = "de0d218c94386e22087702321841805dbaf7011b"
headers = {
    "Authorization": f"Token {api_token}",
    "Content-Type": "application/json"
}

recording_id = "88166f7e-a85d-4c7a-91ec-2f16831b7e79"

# Try the audio endpoint
url = f"https://dunya.compmusic.upf.edu/carnatic/recording/{recording_id}/audio"
print(f"Fetching: {url}")

response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")
print(f"Content-Length: {len(response.content)} bytes")

# Save response to check
with open('test_response.html', 'wb') as f:
    f.write(response.content)

# Parse HTML to find audio source
soup = BeautifulSoup(response.content, 'html.parser')

# Look for audio tags
audio_tags = soup.find_all('audio')
print(f"\nFound {len(audio_tags)} audio tags")

for audio in audio_tags:
    src = audio.get('src')
    print(f"  Audio src: {src}")

# Look for source tags
source_tags = soup.find_all('source')
print(f"\nFound {len(source_tags)} source tags")

for source in source_tags:
    src = source.get('src')
    print(f"  Source src: {src}")

# Look for any links to MP3 files
links = soup.find_all('a', href=True)
mp3_links = [link['href'] for link in links if '.mp3' in link['href'].lower() or 'audio' in link['href'].lower()]
print(f"\nPotential audio links:")
for link in mp3_links[:5]:
    print(f"  {link}")

# Try to find download link
download_link = soup.find('a', text=lambda t: t and 'download' in t.lower())
if download_link:
    print(f"\nDownload link: {download_link.get('href')}")
