# Music Recommender - Fix Metadata Mismatch

## Current Status
- ✅ app.py Streamlit error fixed (session_state initialization moved to top)
- ✅ download_songs.py updated for incremental metadata (handles 200+ songs)
- 200+ songs in `songs/`, only 100 in `songs_metadata.csv`

## Plan Steps
- [ ] 1. Edit download_songs.py for incremental metadata generation
- [ ] 2. Run `python download_songs.py` to populate metadata for songs 101-200+
- [ ] 3. Verify `len(pd.read_csv('songs_metadata.csv')) >= 200`
- [ ] 4. Run `python train_jamendo.py` to retrain with full dataset  
- [ ] 5. Test `streamlit run app.py` with improved recommendations
- [ ] 6. Complete

**Next:** Edit download_songs.py
