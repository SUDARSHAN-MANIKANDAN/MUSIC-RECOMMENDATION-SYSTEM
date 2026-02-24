import os
import pickle
import warnings
import numpy as np
import pandas as pd
import librosa
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

METADATA_FILE  = "songs_metadata.csv"
MIN_DURATION   = 120
MIN_GENRE_SIZE = 5


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=180, mono=True)
        if librosa.get_duration(y=y, sr=sr) < MIN_DURATION:
            return None

        mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
        mel       = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)
        rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zcr       = librosa.feature.zero_crossing_rate(y)
        rms       = librosa.feature.rms(y=y)
        tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
        harmonic  = librosa.effects.harmonic(y)
        tonnetz   = librosa.feature.tonnetz(y=harmonic, sr=sr)

        features = np.hstack([
            np.mean(mfcc, axis=1),     np.std(mfcc, axis=1),
            np.mean(chroma, axis=1),   np.std(chroma, axis=1),
            np.mean(mel, axis=1)[:20],
            np.mean(contrast, axis=1),
            np.mean(rolloff),          np.std(rolloff),
            np.mean(centroid),         np.std(centroid),
            np.mean(bandwidth),        np.std(bandwidth),
            np.mean(zcr),              np.std(zcr),
            np.mean(rms),              np.std(rms),
            np.mean(tonnetz, axis=1),
            float(tempo)
        ])
        return features

    except Exception as e:
        print(f"  Warning: {e}")
        return None


def build_and_train():
    df = pd.read_csv(METADATA_FILE)
    records, X, y_labels = [], [], []

    print(f"Found {len(df)} songs in metadata\n")

    for _, row in df.iterrows():
        path = str(row["path"])
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            continue

        print(f"[{len(records)+1}/{len(df)}] Processing: {row['title']}")
        feats = extract_features(path)
        if feats is None:
            print(f"  Skipped (too short or error)")
            continue

        genre = str(row.get("genre", "Other"))
        if genre.lower() in ("unknown", "nan", "", "none"):
            genre = "Other"

        records.append({
            "track_id" : row["track_id"],
            "title"    : str(row["title"]),
            "artist"   : str(row["artist"]),
            "genre"    : genre,
            "path"     : path
        })
        X.append(feats)
        y_labels.append(genre)

    if len(records) < 20:
        print("Not enough valid songs found.")
        return

    # Merge rare genres into Other
    genre_counts = Counter(y_labels)
    print("\nRaw genre distribution:")
    for g, c in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"   {g}: {c} songs")

    y_labels = [g if genre_counts[g] >= MIN_GENRE_SIZE else "Other" for g in y_labels]
    for r, g in zip(records, y_labels):
        r["genre"] = g

    final_counts = Counter(y_labels)
    print("\nGenre distribution after merging rare genres:")
    for g, c in sorted(final_counts.items(), key=lambda x: -x[1]):
        print(f"   {g}: {c} songs")

    # Remove genres with less than 2 samples
    valid_idx = [i for i, g in enumerate(y_labels) if final_counts[g] >= 2]
    records   = [records[i] for i in valid_idx]
    X         = [X[i] for i in valid_idx]
    y_labels  = [y_labels[i] for i in valid_idx]
    print(f"\nTraining on {len(records)} songs across {len(set(y_labels))} genres")

    X  = np.array(X)
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    n_splits = min(5, min(Counter(y_labels).values()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("\nTraining models...")

    svm = SVC(kernel="rbf", C=100, gamma="scale", probability=True, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    rf  = RandomForestClassifier(n_estimators=500, random_state=42)
    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("gb", gb), ("rf", rf)],
        voting="soft"
    )

    models = {
        "SVM"         : SVC(kernel="rbf", C=100, gamma="scale", probability=True, random_state=42),
        "GradBoost"   : GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42),
        "Ensemble"    : ensemble
    }

    best_model, best_acc, best_name = None, 0, ""

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        cv_acc   = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy").mean()
        print(f"  {name}: Test={test_acc:.2%}  CV={cv_acc:.2%}")
        if test_acc > best_acc:
            best_acc, best_model, best_name = test_acc, model, name

    print(f"\nBest model: {best_name} @ {best_acc:.2%}")

    test_genres = le.classes_[np.unique(y_test)]
    print(classification_report(
        y_test,
        best_model.predict(X_test),
        target_names=test_genres,
        labels=np.unique(y_test)
    ))

    if best_acc < 0.70:
        print(f"Accuracy {best_acc:.2%} is below 70%. Try downloading more songs per genre.")
    else:
        print(f"Accuracy {best_acc:.2%} meets the 70% target!")

    for i, r in enumerate(records):
        r["features_scaled"] = X_scaled[i]
        r["features_raw"]    = X[i]

    with open("features.pkl", "wb") as f:
        pickle.dump({
            "records"         : records,
            "model"           : best_model,
            "scaler"          : scaler,
            "label_encoder"   : le,
            "accuracy"        : best_acc,
            "best_model_name" : best_name
        }, f)

    print("Saved to features.pkl")
    print(f"Final Accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    build_and_train()