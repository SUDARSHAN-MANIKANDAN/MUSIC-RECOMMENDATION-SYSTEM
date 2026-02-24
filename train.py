import os, pickle
import librosa
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

FMA_PATH = "fma_small"
MIN_DURATION = 120  

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=180, mono=True)
        if librosa.get_duration(y=y, sr=sr) < MIN_DURATION:
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)

        features = np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),     
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(mel, axis=1)[:20],                        
            np.mean(spectral_contrast, axis=1),               
            np.mean(rolloff), np.std(rolloff),
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms),
            tempo
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def build_and_train(audio_dir, metadata_csv, max_songs=200):
    df = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])
    records, X, y_labels = [], [], []
    count = 0

    for root, _, files in os.walk(audio_dir):
        for fname in sorted(files):
            if fname.endswith(".mp3") and count < max_songs:
                path = os.path.join(root, fname)
                try:
                    track_id = int(os.path.splitext(fname)[0])
                except ValueError:
                    continue

                feats = extract_features(path)
                if feats is not None:
                    try:
                        title = df.loc[track_id, ("track", "title")]
                        artist = df.loc[track_id, ("artist", "name")]
                        genre = df.loc[track_id, ("track", "genre_top")]
                    except:
                        title, artist, genre = fname, "Unknown", "Unknown"

                    if genre == "Unknown" or (isinstance(genre, float) and np.isnan(genre)):
                        continue

                    records.append({
                        "track_id": track_id,
                        "title": str(title),
                        "artist": str(artist),
                        "genre": str(genre),
                        "path": path
                    })
                    X.append(feats)
                    y_labels.append(str(genre))
                    count += 1
                    print(f"[{count}/200] {title} — {genre}")

    if len(records) < 20:
        print(" Not enough valid songs found. Check your FMA path and metadata.")
        return

    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n Training models...")

    models = {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42)
    }

    best_model, best_acc, best_name = None, 0, ""
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        cv_acc = cross_val_score(model, X_scaled, y, cv=5).mean()
        print(f"  {name}: Test={acc:.2%}  CV={cv_acc:.2%}")
        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name

    print(f"\n Best model: {best_name} @ {best_acc:.2%}")
    print(classification_report(y_test, best_model.predict(X_test),
                                 target_names=le.classes_))

    for i, r in enumerate(records):
        r["features_scaled"] = X_scaled[i]
        r["features_raw"] = X[i]

    with open("features.pkl", "wb") as f:
        pickle.dump({
            "records": records,
            "model": best_model,
            "scaler": scaler,
            "label_encoder": le,
            
            "accuracy": best_acc,
            "best_model_name": best_name
        }, f)

    print(" Saved to features.pkl")
    print(f"\n Final Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    import sys
    audio_dir = sys.argv[1] if len(sys.argv) > 1 else "fma_small"
    metadata_csv = sys.argv[2] if len(sys.argv) > 2 else "fma_metadata/tracks.csv"
    build_and_train(audio_dir, metadata_csv)
