import os
import requests
import pandas as pd
import time


API_KEY = "25f54c3d"  
OUTPUT_DIR = "songs"
METADATA_FILE = "songs_metadata.csv"
TARGET_SONGS = 100
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

def main():
    print(" Jamendo Music Downloader")
    print(f"   Target: {TARGET_SONGS} songs >= {MIN_DURATION//60} minutes\n")

    records = []
    offset = 0
    batch_size = 200

    while len(records) < TARGET_SONGS:
        print(f" Fetching tracks (offset={offset})...")
        tracks = fetch_tracks(offset=offset, limit=batch_size)

        if not tracks:
            print("No more tracks available.")
            break

        for track in tracks:
            if len(records) >= TARGET_SONGS:
                break

            duration = int(track.get("duration", 0))
            if duration < MIN_DURATION:
                continue

            idx = len(records) + 1
            print(f"[{idx}/{TARGET_SONGS}] Downloading: {track['name']} by {track['artist_name']} ({duration//60}m {duration%60}s)")

            filepath = download_track(track, idx)
            if filepath:
                records.append({
                    "track_id": track["id"],
                    "title": track["name"],
                    "artist": track["artist_name"],
                    "genre": (track.get("musicinfo", {}).get("tags", {}).get("genres") or ["Unknown"])[0],
                    "duration": duration,
                    "path": filepath,
                    "license": track.get("license_ccurl", "")
                })
                time.sleep(0.3)

        offset += batch_size

    if records:
        df = pd.DataFrame(records)
        df.to_csv(METADATA_FILE, index=False)
        print(f"\n Downloaded {len(records)} songs to '{OUTPUT_DIR}/'")
        print(f" Metadata saved to '{METADATA_FILE}'")
    else:
        print("\n No songs downloaded. Check your API key.")

if __name__ == "__main__":
    main()
