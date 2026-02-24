import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    with open("features.pkl", "rb") as f:
        return pickle.load(f)

def get_recommendations(song_title, top_n=5):
    data = load_data()
    records = data["records"]
    model = data["model"]
    le = data["label_encoder"]

    idx = next((i for i, r in enumerate(records) if r["title"] == song_title), None)
    if idx is None:
        return []

    feature_matrix = np.array([r["features_scaled"] for r in records])
    sim_scores = cosine_similarity([feature_matrix[idx]], feature_matrix)[0]

    
    predicted_genres = model.predict(feature_matrix)
    input_genre = predicted_genres[idx]

    boosted_scores = sim_scores.copy()
    for i, genre in enumerate(predicted_genres):
        if genre == input_genre:
            boosted_scores[i] *= 1.3  

    sorted_indices = np.argsort(boosted_scores)[::-1]
    results = []
    for i in sorted_indices:
        if i != idx:
            results.append({
                "title": records[i]["title"],
                "artist": records[i]["artist"],
                "genre": records[i]["genre"],
                "predicted_genre": le.inverse_transform([predicted_genres[i]])[0],
                "similarity": round(sim_scores[i] * 100, 1),
                "path": records[i]["path"]
            })
        if len(results) == top_n:
            break
    return results

def get_model_accuracy():
    data = load_data()
    return data.get("accuracy", 0)

def get_model_name():
    data = load_data()
    return data.get("best_model_name", "Unknown")

def get_all_songs():
    data = load_data()
    return data["records"]
