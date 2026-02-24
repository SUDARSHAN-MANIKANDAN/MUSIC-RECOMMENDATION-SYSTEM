import pytest
import numpy as np
import pickle
import os
import time
import tempfile
import soundfile as sf
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def dummy_audio_file():
    """30s silent WAV file"""
    dummy = np.zeros(44100 * 30)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, dummy, 44100)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def short_audio_file():
    """60s audio — below 2 min threshold"""
    dummy = np.zeros(44100 * 60)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, dummy, 44100)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def noisy_audio_file():
    """30s random noise WAV"""
    noise = np.random.randn(44100 * 30) * 0.1
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, noise, 44100)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def long_audio_file():
    """3 min audio — above 2 min threshold"""
    audio = np.random.randn(44100 * 180) * 0.1
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, 44100)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def loaded_data():
    if not os.path.exists("features.pkl"):
        pytest.skip("features.pkl not found — train the model first")
    with open("features.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture
def first_song_title(loaded_data):
    return loaded_data["records"][0]["title"]

@pytest.fixture
def last_song_title(loaded_data):
    return loaded_data["records"][-1]["title"]


# ═══════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════

class TestFeatureExtraction:

    def test_returns_numpy_array(self, long_audio_file):
        from train_jamendo import extract_features
        feats = extract_features(long_audio_file)
        assert isinstance(feats, np.ndarray), "Features must be a numpy array"

    def test_feature_shape_is_consistent(self, long_audio_file, noisy_audio_file):
        """Two different audio files must produce same length feature vector"""
        from train_jamendo import extract_features
        # noisy_audio_file is only 30s so it will be rejected,
        # create another long noisy file
        noise = np.random.randn(44100 * 180) * 0.05
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, noise, 44100)
            tmp = f.name
        try:
            f1 = extract_features(long_audio_file)
            f2 = extract_features(tmp)
            assert f1.shape == f2.shape, "Feature vectors must have consistent shape"
        finally:
            os.unlink(tmp)

    def test_no_nan_values(self, long_audio_file):
        from train_jamendo import extract_features
        feats = extract_features(long_audio_file)
        assert not np.isnan(feats).any(), "Features must not contain NaN"

    def test_no_inf_values(self, long_audio_file):
        from train_jamendo import extract_features
        feats = extract_features(long_audio_file)
        assert not np.isinf(feats).any(), "Features must not contain Inf"

    def test_short_audio_rejected(self, short_audio_file):
        """Files under 2 minutes must return None"""
        from train_jamendo import extract_features
        feats = extract_features(short_audio_file)
        assert feats is None, "Audio under 2 min should return None"

    def test_nonexistent_file_returns_none(self):
        from train_jamendo import extract_features
        feats = extract_features("fake_path/nonexistent.mp3")
        assert feats is None

    def test_features_not_all_zeros(self, long_audio_file):
        from train_jamendo import extract_features
        feats = extract_features(long_audio_file)
        assert not np.all(feats == 0), "Feature vector should not be all zeros"

    def test_different_audio_different_features(self, long_audio_file):
        from train_jamendo import extract_features
        noise2 = np.random.randn(44100 * 180) * 0.5
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, noise2, 44100)
            tmp = f.name
        try:
            f1 = extract_features(long_audio_file)
            f2 = extract_features(tmp)
            assert not np.allclose(f1, f2), "Different audio should give different features"
        finally:
            os.unlink(tmp)

    def test_feature_vector_minimum_length(self, long_audio_file):
        from train_jamendo import extract_features
        feats = extract_features(long_audio_file)
        assert len(feats) >= 100, "Feature vector seems too short"

    def test_corrupt_file_returns_none(self):
        from train_jamendo import extract_features
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"this is not valid audio data at all!!!")
            tmp = f.name
        try:
            feats = extract_features(tmp)
            assert feats is None, "Corrupt file should return None"
        finally:
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════
# 2. MODEL & DATA INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════

class TestModelIntegrity:

    def test_pkl_has_model_key(self, loaded_data):
        assert "model" in loaded_data

    def test_pkl_has_scaler_key(self, loaded_data):
        assert "scaler" in loaded_data

    def test_pkl_has_label_encoder(self, loaded_data):
        assert "label_encoder" in loaded_data

    def test_pkl_has_accuracy(self, loaded_data):
        assert "accuracy" in loaded_data

    def test_pkl_has_records(self, loaded_data):
        assert "records" in loaded_data

    def test_records_not_empty(self, loaded_data):
        assert len(loaded_data["records"]) > 0

    def test_minimum_200_songs(self, loaded_data):
        count = len(loaded_data["records"])
        assert count >= 200, f"Expected 200 songs, found only {count}"

    def test_accuracy_above_70_percent(self, loaded_data):
        acc = loaded_data["accuracy"]
        assert acc >= 0.70, f"Accuracy {acc:.2%} is below 70% threshold!"

    def test_accuracy_in_valid_range(self, loaded_data):
        acc = loaded_data["accuracy"]
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} is out of valid range"

    def test_all_records_have_required_fields(self, loaded_data):
        required = {"title", "artist", "genre", "path", "features_scaled"}
        for i, r in enumerate(loaded_data["records"]):
            missing = required - r.keys()
            assert not missing, f"Record #{i} missing fields: {missing}"

    def test_no_duplicate_song_titles(self, loaded_data):
        titles = [r["title"] for r in loaded_data["records"]]
        assert len(titles) == len(set(titles)), "Duplicate song titles found in dataset"

    def test_feature_vectors_consistent_length(self, loaded_data):
        lengths = set(len(r["features_scaled"]) for r in loaded_data["records"])
        assert len(lengths) == 1, f"Inconsistent feature vector lengths: {lengths}"

    def test_genres_not_all_unknown(self, loaded_data):
        genres = [r["genre"] for r in loaded_data["records"]]
        unknown_ratio = genres.count("Unknown") / len(genres)
        assert unknown_ratio < 0.5, f"{unknown_ratio:.0%} of songs have Unknown genre — too many!"

    def test_at_least_2_unique_genres(self, loaded_data):
        genres = set(r["genre"] for r in loaded_data["records"])
        assert len(genres) >= 2, "Need at least 2 genres for meaningful recommendations"

    def test_label_encoder_covers_all_genres(self, loaded_data):
        le = loaded_data["label_encoder"]
        genres_in_records = set(r["genre"] for r in loaded_data["records"])
        for g in genres_in_records:
            assert g in le.classes_, f"Genre '{g}' missing from label encoder"

    def test_model_can_predict(self, loaded_data):
        model = loaded_data["model"]
        scaler = loaded_data["scaler"]
        sample = loaded_data["records"][0]["features_scaled"].reshape(1, -1)
        pred = model.predict(sample)
        assert pred is not None
        assert len(pred) == 1

    def test_scaler_has_correct_shape(self, loaded_data):
        scaler = loaded_data["scaler"]
        expected_len = len(loaded_data["records"][0]["features_scaled"])
        assert scaler.n_features_in_ == expected_len


# ═══════════════════════════════════════════════════════════════
# 3. RECOMMENDATION ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestRecommendationEngine:

    def test_returns_list(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=5)
        assert isinstance(recs, list)

    def test_correct_count_3(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=3)
        assert len(recs) == 3

    def test_correct_count_5(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=5)
        assert len(recs) == 5

    def test_correct_count_10(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=10)
        assert len(recs) == 10

    def test_no_self_recommendation(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=5)
        rec_titles = [r["title"] for r in recs]
        assert first_song_title not in rec_titles, "Song should never recommend itself!"

    def test_similarity_in_valid_range(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=5)
        for r in recs:
            assert 0 <= r["similarity"] <= 100, f"Similarity {r['similarity']} out of 0-100 range"

    def test_required_fields_present(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=3)
        required = {"title", "artist", "genre", "similarity"}
        for r in recs:
            assert required.issubset(r.keys()), f"Recommendation missing fields: {required - r.keys()}"

    def test_invalid_song_returns_empty_list(self):
        from recommend import get_recommendations
        recs = get_recommendations("ZZZZ_THIS_DOES_NOT_EXIST_9999", top_n=5)
        assert recs == [], "Invalid song title should return empty list"

    def test_no_duplicate_recommendations(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=10)
        titles = [r["title"] for r in recs]
        assert len(titles) == len(set(titles)), "Recommendations contain duplicates"

    def test_different_inputs_different_outputs(self, first_song_title, last_song_title):
        from recommend import get_recommendations
        recs1 = set(r["title"] for r in get_recommendations(first_song_title, top_n=5))
        recs2 = set(r["title"] for r in get_recommendations(last_song_title, top_n=5))
        assert recs1 != recs2, "Different songs should produce different recommendations"

    def test_recommendations_are_strings(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=3)
        for r in recs:
            assert isinstance(r["title"], str)
            assert isinstance(r["artist"], str)
            assert isinstance(r["genre"], str)

    def test_similarity_is_float(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=3)
        for r in recs:
            assert isinstance(r["similarity"], (int, float))

    def test_empty_string_returns_empty(self):
        from recommend import get_recommendations
        recs = get_recommendations("", top_n=5)
        assert recs == [], "Empty string should return empty list"


# ═══════════════════════════════════════════════════════════════
# 4. PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════════

class TestPerformance:

    def test_recommendations_under_2_seconds(self, first_song_title):
        from recommend import get_recommendations
        start = time.time()
        get_recommendations(first_song_title, top_n=10)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Took {elapsed:.2f}s — recommendations must be under 2s"

    def test_feature_extraction_under_10_seconds(self, long_audio_file):
        from train_jamendo import extract_features
        start = time.time()
        extract_features(long_audio_file)
        elapsed = time.time() - start
        assert elapsed < 10.0, f"Feature extraction took {elapsed:.2f}s — too slow!"

    def test_pkl_loads_under_3_seconds(self):
        if not os.path.exists("features.pkl"):
            pytest.skip("features.pkl not found")
        start = time.time()
        with open("features.pkl", "rb") as f:
            pickle.load(f)
        elapsed = time.time() - start
        assert elapsed < 3.0, f"Model load took {elapsed:.2f}s — too slow!"

    def test_repeated_recommendations_consistent(self, first_song_title):
        """Same input must always give same output"""
        from recommend import get_recommendations
        recs1 = [r["title"] for r in get_recommendations(first_song_title, top_n=5)]
        recs2 = [r["title"] for r in get_recommendations(first_song_title, top_n=5)]
        assert recs1 == recs2, "Recommendations are not deterministic!"


# ═══════════════════════════════════════════════════════════════
# 5. METADATA & DOWNLOAD TESTS
# ═══════════════════════════════════════════════════════════════

class TestMetadata:

    def test_metadata_file_exists(self):
        assert os.path.exists("songs_metadata.csv"), "songs_metadata.csv not found — run download_songs.py first"

    def test_required_columns_exist(self):
        if not os.path.exists("songs_metadata.csv"):
            pytest.skip("No metadata file")
        df = pd.read_csv("songs_metadata.csv")
        required = {"track_id", "title", "artist", "genre", "duration", "path"}
        missing = required - set(df.columns)
        assert not missing, f"Metadata missing columns: {missing}"

    def test_all_songs_meet_min_duration(self):
        if not os.path.exists("songs_metadata.csv"):
            pytest.skip("No metadata file")
        df = pd.read_csv("songs_metadata.csv")
        short = df[df["duration"] < 120]
        assert len(short) == 0, f"{len(short)} songs are under 2 minutes!"

    def test_no_missing_titles(self):
        if not os.path.exists("songs_metadata.csv"):
            pytest.skip("No metadata file")
        df = pd.read_csv("songs_metadata.csv")
        assert df["title"].notna().all(), "Some songs have missing titles"

    def test_no_missing_artists(self):
        if not os.path.exists("songs_metadata.csv"):
            pytest.skip("No metadata file")
        df = pd.read_csv("songs_metadata.csv")
        assert df["artist"].notna().all(), "Some songs have missing artists"

    def test_no_duplicate_track_ids(self):
        if not os.path.exists("songs_metadata.csv"):
            pytest.skip("No metadata file")
        df = pd.read_csv("songs_metadata.csv")
        assert df["track_id"].nunique() == len(df), "Duplicate track IDs found"

    def test_audio_files_exist_on_disk(self):
        if not os.path.exists("songs_metadata.csv"):
            pytest.skip("No metadata file")
        df = pd.read_csv("songs_metadata.csv")
        missing = [p for p in df["path"] if not os.path.exists(p)]
        assert len(missing) == 0, f"{len(missing)} audio files are missing from disk: {missing[:5]}"

    def test_target_song_count_reached(self):
        if not os.path.exists("songs_metadata.csv"):
            pytest.skip("No metadata file")
        df = pd.read_csv("songs_metadata.csv")
        assert len(df) >= 200, f"Only {len(df)} songs downloaded, need at least 200"

    def test_duration_values_are_positive(self):
        if not os.path.exists("songs_metadata.csv"):
            pytest.skip("No metadata file")
        df = pd.read_csv("songs_metadata.csv")
        assert (df["duration"] > 0).all(), "Some songs have zero or negative duration"