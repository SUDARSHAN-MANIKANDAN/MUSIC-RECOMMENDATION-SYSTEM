import streamlit as st
import os
import random
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from recommend import get_recommendations, get_model_accuracy, get_model_name, get_all_songs

# ── PAGE CONFIG ─────────────────────────────────────
st.set_page_config(page_title="HarmonyBeats", layout="wide")

# ── STYLING ─────────────────────────────────────────
st.markdown("""
<style>
body {background-color:#0e1117; color:white;}
h1 {text-align:center; color:#00e0ff;}
button {border-radius:8px !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>HarmonyBeats</h1>", unsafe_allow_html=True)

# ── SESSION STATE ───────────────────────────────────
if "liked" not in st.session_state: st.session_state.liked = []
if "queue" not in st.session_state: st.session_state.queue = []
if "playlists" not in st.session_state: st.session_state.playlists = {}
if "current_song" not in st.session_state: st.session_state.current_song = None
if "recs" not in st.session_state: st.session_state.recs = []

# ── LOAD DATA ───────────────────────────────────────
@st.cache_data
def load():
    records = get_all_songs()
    return records, get_model_accuracy(), get_model_name()

records, accuracy, model_name = load()

all_titles = [r["title"] for r in records]
genre_map = {r["title"]: r for r in records}
all_genres = sorted(set(r["genre"] for r in records))

if st.session_state.current_song is None:
    st.session_state.current_song = all_titles[0]

# ── NAVIGATION ──────────────────────────────────────
page = st.sidebar.radio("Navigate", ["Home", "Liked Songs", "Playlists"])
st.sidebar.caption(f"{model_name} · {accuracy:.2%}")
st.sidebar.write(f"Queue: {len(st.session_state.queue)}")
st.sidebar.markdown("---")
st.sidebar.subheader("Queue")

if not st.session_state.queue:
    st.sidebar.write("Queue is empty")
else:
    for i, song in enumerate(st.session_state.queue):
        col1, col2, col3 = st.sidebar.columns([3,1,1])

        with col1:
            st.write(song)

        with col2:
            if st.button("▶", key=f"qplay_{i}"):
                st.session_state.current_song = song
                st.rerun()

        with col3:
            if st.button("✖", key=f"qremove_{i}"):
                st.session_state.queue.pop(i)
                st.rerun()

# ── WAVEFORM ────────────────────────────────────────
def plot_waveform(path):
    try:
        y, sr = librosa.load(path, duration=30)
        fig, ax = plt.subplots(figsize=(6,1))
        ax.plot(y, color="#00e0ff")
        ax.axis("off")
        return fig
    except:
        return None

# ════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════
if page == "Home":

    col1, col2, col3 = st.columns([3,2,1])

    with col1:
        search = st.text_input("Search", "", label_visibility="collapsed")

    with col2:
        genre = st.selectbox("Genre", ["All"] + all_genres, label_visibility="collapsed")

    with col3:
        top_n = st.slider("Top", 3, 10, 5, label_visibility="collapsed")

    filtered = records
    if search:
        q = search.lower()
        filtered = [r for r in records if q in r["title"].lower()]

    if genre != "All":
        filtered = [r for r in filtered if r["genre"] == genre]

    titles = [r["title"] for r in filtered]

    # 🔥 FIXED SELECTBOX
    current = st.session_state.current_song
    idx = titles.index(current) if current in titles else 0

    selected = st.selectbox("Song", titles, index=idx, label_visibility="collapsed")

    if selected != st.session_state.current_song:
        st.session_state.current_song = selected

    record = genre_map[st.session_state.current_song]

    st.subheader("Now Playing")
    st.write(f"{record['title']} — {record['artist']}")

    # controls
    c1,c2,c3,c4 = st.columns(4)

    with c1:
        if st.button("Like"):
            if selected not in st.session_state.liked:
                st.session_state.liked.append(selected)

    with c2:
        if st.button("＋"):
            if selected not in st.session_state.queue:
                st.session_state.queue.append(selected)

    with c3:
        if st.button("Random"):
            st.session_state.current_song = random.choice(titles)
            st.rerun()

    with c4:
        if st.button("Next"):
            if st.session_state.queue:
                st.session_state.current_song = st.session_state.queue.pop(0)
                st.rerun()

    # audio
    path = record.get("path","")
    if path and os.path.exists(path):
        fig = plot_waveform(path)
        if fig:
            st.pyplot(fig)
        st.audio(path)
    st.markdown("### Add to Playlist")

    if st.session_state.playlists:
        selected_playlist = st.selectbox(
            "Select Playlist",
            list(st.session_state.playlists.keys()),
            label_visibility="collapsed"
        )

        if st.button("Add to Playlist"):
            if st.session_state.current_song not in st.session_state.playlists[selected_playlist]:
                st.session_state.playlists[selected_playlist].append(st.session_state.current_song)
    else:
        st.write("No playlists available. Create one in Playlists page.")
    # recommendations
    st.markdown("---")
    if st.button("Recommend"):
        st.session_state.recs = get_recommendations(selected, top_n)

    for i,rec in enumerate(st.session_state.recs):
        c1,c2,c3 = st.columns([4,1,1])

        with c1:
            st.write(rec["title"])

        with c2:
            if st.button("▶", key=f"rplay{i}"):
                st.session_state.current_song = rec["title"]
                st.rerun()

        with c3:
            if st.button("＋", key=f"rqueue{i}"):
                if rec["title"] not in st.session_state.queue:
                    st.session_state.queue.append(rec["title"])

# ════════════════════════════════════════════════════
# LIKED
# ════════════════════════════════════════════════════
elif page == "Liked Songs":

    for i,song in enumerate(st.session_state.liked):
        c1,c2,c3 = st.columns([4,1,1])

        with c1: st.write(song)

        with c2:
            if st.button("▶", key=f"lp{i}"):
                st.session_state.current_song = song
                st.rerun()

        with c3:
            if st.button("✖", key=f"lr{i}"):
                st.session_state.liked.remove(song)
                st.rerun()

# ════════════════════════════════════════════════════
# PLAYLISTS
# ════════════════════════════════════════════════════
else:

    name = st.text_input("New Playlist")

    if st.button("Create"):
        if name and name not in st.session_state.playlists:
            st.session_state.playlists[name] = []

    for pname,songs in st.session_state.playlists.items():
        st.markdown(f"### {pname}")

        for i,song in enumerate(songs):
            c1,c2,c3 = st.columns([4,1,1])

            with c1: st.write(song)

            with c2:
                if st.button("▶", key=f"pp{i}{pname}"):
                    st.session_state.current_song = song
                    st.rerun()

            with c3:
                if st.button("✖", key=f"pr{i}{pname}"):
                    st.session_state.playlists[pname].remove(song)
                    st.rerun()