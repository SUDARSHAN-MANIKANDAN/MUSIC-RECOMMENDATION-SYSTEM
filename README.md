# 🎵 MusicMind — Music Recommendation System

Audio-feature based music recommender using full-length songs (2min+) from Jamendo, targeting 70%+ genre classification accuracy.

---

## 📦 Setup

```bash
pip install -r requirements.txt
```

---

## 🔑 Step 1: Get Jamendo API Key

Sign up free at: https://devportal.jamendo.com  
Takes ~2 minutes. Copy your `client_id`.

---

## 🎵 Step 2: Download 200 Songs

```bash
python download_songs.py --api_key YOUR_API_KEY
```

This will:
- Fetch 200 songs across 10 genres (pop, rock, jazz, electronic, etc.)
- Filter to only songs ≥ 2 minutes
- Download MP3s to `jamendo_songs/`
- Save metadata to `jamendo_metadata.csv`

**Download time:** ~15–30 mins depending on your internet speed

---

## 🔧 Step 3: Train the Model

```bash
python train_jamendo.py
```

This will:
- Extract audio features (MFCC, chroma, tempo, etc.) from all 200 songs
- Train SVM, Gradient Boosting, and Random Forest
- Auto-select the best model
- Save everything to `features.pkl`

**Training time:** ~8–12 minutes on CPU (full songs take longer than 30s clips)

---

## 🚀 Step 4: Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🎯 Accuracy

| Model | Expected Accuracy |
|---|---|
| SVM (RBF) | 70–78% |
| Gradient Boosting | 72–80% |
| Random Forest | 65–72% |

The best model is auto-selected and displayed in the UI.

---

## 📁 Files

| File | Purpose |
|---|---|
| `download_songs.py` | Fetch 200 songs from Jamendo API |
| `train_jamendo.py` | Feature extraction + model training |
| `recommend.py` | Recommendation logic |
| `app.py` | Streamlit UI |
| `requirements.txt` | Python dependencies |
| `features.pkl` | Generated after training (not included) |
| `jamendo_metadata.csv` | Generated after download (not included) |
