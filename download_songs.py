import os
import requests
import pandas as pd
import time


API_KEY = "25f54c3d"  
OUTPUT_DIR = "songs"
METADATA_FILE = "songs_metadata.csv"
TARGET_SONGS = 300
MIN_DURATION = 120             


os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_tracks(offset=0, limit=200):
    url = "https://api.jamendo.com/v3.0/tracks/"
    params = {
        "client_id": API_KEY,
        "format": "json",
        "limit": limit,
        "offset": offset,
        "audioformat": "mp32",
        "audiodlformat": "mp32",
        "order": "popularity_total",
        "duration_between": "120_600",  
        "include": "musicinfo",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("results", [])

def download_track(track, idx):
    audio_url = track.get("audiodownload") or track.get("audio")
    if not audio_url:
        return False

    filename = os.path.join(OUTPUT_DIR, f"{idx:03d}_{track['id']}.mp3")
    if os.path.exists(filename):
        print(f"    Already exists: {filename}")
        return filename

    try:
        r = requests.get(audio_url, stream=True, timeout=30)
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename
    except Exception as e:
        print(f"   Failed to download {track['name']}: {e}")
        return False

def get_existing_metadata():
    """Load existing metadata and track IDs"""
    if os.path.exists(METADATA_FILE):
        df = pd.read_csv(METADATA_FILE)
        return df.to_dict('records'), set(df['track_id'].astype(str))
    return [], set()

def extract_track_id_from_filename(filename):
    """Extract track_id from filename like '101_123456.mp3' -> '123456'"""
    try:
        parts = os.path.basename(filename).split('_')
        if len(parts) == 2 and parts[1].endswith('.mp3'):
            return parts[1][:-4]
        return None
    except:
        return None

def fetch_track_metadata(track_id):
    """Fetch metadata for specific track_id from Jamendo API"""
    url = "https://api.jamendo.com/v3.0/tracks/"
    params = {
        "client_id": API_KEY,
        "format": "json",
        "id": track_id,
        "include": "musicinfo"
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            track = results[0]
            duration = int(track.get("duration", 0))
            if duration >= MIN_DURATION:
                # Find matching file (handle duplicates)
                song_files = [f for f in os.listdir(OUTPUT_DIR) if f'{track["id"]}' in f and f.endswith('.mp3')]
                path = os.path.join(OUTPUT_DIR, song_files[0]) if song_files else ""
                return {
                    "track_id": track["id"],
                    "title": track["name"],
                    "artist": track["artist_name"],
                    "genre": (track.get("musicinfo", {}).get("tags", {}).get("genres") or ["Unknown"])[0],
                    "duration": duration,
                    "path": path,
                    "license": track.get("license_ccurl", "")
                }
    except Exception as e:
        print(f"  Failed to fetch metadata for {track_id}: {e}")
    return None

def main():
    print("🎵 Jamendo Music Downloader - Incremental Mode")
    print(f"   Target: {TARGET_SONGS} songs total\n")

    # Load existing metadata
    existing_records, existing_track_ids = get_existing_metadata()
    print(f"✅ Found {len(existing_records)} existing metadata entries")

    # Scan songs/ for files missing metadata
    if not os.path.exists(OUTPUT_DIR):
        print("❌ No songs directory found")
        return
    
    song_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp3')]
    missing_track_ids = []
    
    for song_file in song_files:
        track_id = extract_track_id_from_filename(song_file)
        if track_id and str(track_id) not in existing_track_ids:
            missing_track_ids.append(str(track_id))
    
    print(f"🔍 Found {len(missing_track_ids)} songs without metadata")

    if not missing_track_ids:
        print("✅ All songs have metadata!")
        return

    # Fetch metadata for missing songs
    new_records = []
    max_new = TARGET_SONGS - len(existing_records)
    for i, track_id in enumerate(missing_track_ids[:max_new], 1):
        print(f"[{len(existing_records)+i}/{TARGET_SONGS}] Fetching metadata: {track_id}")
        metadata = fetch_track_metadata(track_id)
        if metadata:
            new_records.append(metadata)
            print(f"   ✓ {metadata['title'][:50]}...")
        else:
            print(f"   ✗ Failed/Skipped")
        time.sleep(0.5)  # API rate limit

    # Save combined metadata
    if new_records:
        all_records = existing_records + new_records
        df = pd.DataFrame(all_records)
        df.to_csv(METADATA_FILE, index=False)
        print(f"\n🎉 Added {len(new_records)} new entries!")
        print(f"📊 Total songs with metadata: {len(all_records)}")
        print(f"💾 Saved to {METADATA_FILE}")
    else:
        print("\nℹ️ No new metadata added")

if __name__ == "__main__":
    main()
