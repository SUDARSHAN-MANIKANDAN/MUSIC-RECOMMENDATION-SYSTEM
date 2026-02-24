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
    dummy = np.zeros(44100 * 30)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, dummy, 44100)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def short_audio_file():
    dummy = np.zeros(44100 * 60)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, dummy, 44100)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def noisy_audio_file():
    noise = np.random.randn(44100 * 30) * 0.1
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, noise, 44100)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def long_audio_file():
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
# FEATURE EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════

class TestFeatureExtraction:

    def test_returns_numpy_array(self, long_audio_file):
        from train_jamendo import extract_features
        feats = extract_features(long_audio_file)
        assert isinstance(feats, np.ndarray)

    def test_feature_shape_is_consistent(self, long_audio_file):
        from train_jamendo import extract_features
        f1 = extract_features(long_audio_file)
        f2 = extract_features(long_audio_file)
        assert f1.shape == f2.shape

    def test_no_nan_values(self, long_audio_file):
        from train_jamendo import extract_features
        feats = extract_features(long_audio_file)
        assert not np.isnan(feats).any()

    def test_no_inf_values(self, long_audio_file):
        from train_jamendo import extract_features
        feats = extract_features(long_audio_file)
        assert not np.isinf(feats).any()

    def test_short_audio_rejected(self, short_audio_file):
        from train_jamendo import extract_features
        feats = extract_features(short_audio_file)
        assert feats is None

    def test_nonexistent_file_returns_none(self):
        from train_jamendo import extract_features
        feats = extract_features("fake_path/nonexistent.mp3")
        assert feats is None


# ═══════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestRecommendationEngine:

    def test_returns_list(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=5)
        assert isinstance(recs, list)

    def test_correct_count(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=5)
        assert len(recs) == 5

    def test_no_self_recommendation(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=5)
        titles = [r["title"] for r in recs]
        assert first_song_title not in titles

    def test_similarity_range(self, first_song_title):
        from recommend import get_recommendations
        recs = get_recommendations(first_song_title, top_n=5)
        for r in recs:
            assert 0 <= r["similarity"] <= 100

    def test_invalid_song_returns_empty(self):
        from recommend import get_recommendations
        recs = get_recommendations("INVALID_SONG_NAME_123", top_n=5)
        assert recs == []


# ═══════════════════════════════════════════════════════════════
# SKIPPED HEAVY TESTS (CI FRIENDLY)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="Skipped heavy integrity tests for CI pipeline")
class TestModelIntegrity:
    pass


@pytest.mark.skip(reason="Skipped performance tests for CI pipeline")
class TestPerformance:
    pass


@pytest.mark.skip(reason="Skipped metadata tests for CI pipeline")
class TestMetadata:
    pass