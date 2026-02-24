import streamlit as st
import os
import pickle
from recommend import get_recommendations, get_model_accuracy, get_model_name, get_all_songs

st.set_page_config(page_title="🎵 Music Recommender", layout="wide", page_icon="🎵")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #0a0a0f;
        color: #f0eee6;
    }
    .stApp { background: #0a0a0f; }

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bceff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4e;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffd93d;
        font-family: 'Space Mono', monospace;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #888;
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .song-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    .song-card:hover {
        border-color: #ff6b6b;
        transform: translateX(5px);
        box-shadow: -5px 0 25px rgba(255,107,107,0.2);
    }
    .selected-card {
        background: linear-gradient(135deg, #1e1a2e 0%, #1e1630 100%);
        border: 1px solid #6bceff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 0 20px rgba(107,206,255,0.1);
    }
    .genre-tag {
        background: rgba(255,107,107,0.15);
        border: 1px solid rgba(255,107,107,0.4);
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        color: #ff6b6b;
        font-family: 'Space Mono', monospace;
    }
    .rank-badge {
        background: rgba(255,107,107,0.2);
        color: #ff6b6b;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 4px 10px;
        border-radius: 8px;
        margin-right: 8px;
    }
    .accuracy-badge {
        background: linear-gradient(90deg, rgba(107,206,255,0.15), rgba(255,107,107,0.15));
        border: 1px solid rgba(107,206,255,0.3);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #6bceff;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff6b6b, #ffd93d) !important;
        color: #0a0a0f !important;
        font-weight: 800 !important;
        font-family: 'Syne', sans-serif !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255,107,107,0.3) !important;
    }
    .stSelectbox label, .stSlider label {
        color: #aaa !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    hr { border-color: #2d2d4e !important; }
    .no-model-warning {
        background: rgba(255,107,107,0.1);
        border: 1px solid rgba(255,107,107,0.3);
        border-radius: 10px;
        padding: 1.5rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎵 MusicMind</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">audio-feature based music recommendation · fma dataset</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Check if model exists ────────────────────────────────────────────────────
if not os.path.exists("features.pkl"):
    st.markdown("""
    <div class="no-model-warning">
        ⚠️  <strong>No trained model found.</strong><br><br>
        Run the following command to train the model first:<br><br>
        <code>python train.py fma_small fma_metadata/tracks.csv</code><br><br>
        This will take ~3–6 minutes for 200 songs.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def cached_load():
    records = get_all_songs()
    accuracy = get_model_accuracy()
    model_name = get_model_name()
    return records, accuracy, model_name

records, accuracy, model_name = cached_load()
song_titles = [r["title"] for r in records]
genres = sorted(set(r["genre"] for r in records))

# ── Stats Row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(records)}</div><div class="stat-label">Songs</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(genres)}</div><div class="stat-label">Genres</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{accuracy:.0%}</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-card"><div class="stat-value" style="font-size:1rem;padding-top:0.5rem">{model_name}</div><div class="stat-label">Best Model</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ── Controls ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### 🎧 Select a Song")
    
    # Optional genre filter
    filter_genre = st.selectbox("Filter by genre (optional)", ["All Genres"] + genres)
    filtered_titles = song_titles if filter_genre == "All Genres" else [
        r["title"] for r in records if r["genre"] == filter_genre
    ]
    
    selected_song = st.selectbox("Choose a song you like", filtered_titles)
    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

with col_right:
    st.markdown("#### 🎼 Selected Track")
    sel = next((r for r in records if r["title"] == selected_song), None)
    if sel:
        st.markdown(f"""
        <div class="selected-card">
            <strong style="font-size:1.1rem">{sel['title']}</strong><br>
            <span style="color:#aaa">{sel['artist']}</span><br><br>
            <span class="genre-tag">{sel['genre']}</span>
        </div>
        """, unsafe_allow_html=True)
        if sel.get("path") and os.path.exists(sel["path"]):
            st.audio(sel["path"])
        else:
            st.caption("🔇 Audio file not found at stored path")

st.markdown("<br>", unsafe_allow_html=True)

# ── Recommend Button ──────────────────────────────────────────────────────────
if st.button("🔍 Find Similar Songs", use_container_width=True):
    with st.spinner("Analyzing audio features and finding matches..."):
        recs = get_recommendations(selected_song, top_n)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 🎶 Songs similar to *{selected_song}*")
    st.markdown(f'<div class="accuracy-badge">🤖 Recommendations powered by {model_name} · {accuracy:.1%} genre accuracy</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not recs:
        st.error("No recommendations found. The song may not be in the dataset.")
    else:
        for i, rec in enumerate(recs, 1):
            bar_pct = min(int(rec["similarity"]), 100)
            bar_color = "#ff6b6b" if bar_pct > 70 else "#ffd93d" if bar_pct > 50 else "#6bceff"

            st.markdown(f"""
            <div class="song-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem">
                    <div>
                        <span class="rank-badge">#{i}</span>
                        <strong style="font-size:1.05rem">{rec['title']}</strong><br>
                        <span style="color:#aaa; font-size:0.9rem; margin-left:2.5rem">{rec['artist']}</span>
                    </div>
                    <div style="text-align:right">
                        <span class="genre-tag">{rec['genre']}</span><br>
                        <span style="font-family:'Space Mono',monospace; color:#ffd93d; font-size:0.85rem; margin-top:4px; display:block">
                            {rec['similarity']}% match
                        </span>
                    </div>
                </div>
                <div style="background:#2d2d4e; border-radius:3px; height:5px; margin-top:1rem">
                    <div style="width:{bar_pct}%; height:5px; background:{bar_color}; border-radius:3px; transition:width 0.5s ease"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if rec.get("path") and os.path.exists(rec["path"]):
                st.audio(rec["path"])

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.72rem; color:#444">
    Built with FMA Dataset · librosa · scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
