from src.preprocess import normalize_text
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "outputs" / "sentiment_model.joblib")
vectorizer = joblib.load(ROOT / "outputs" / "tfidf_vectorizer.joblib")

samples = [
    "Ee movie chala bagundi super ga undi",
    "Movie asalu bagaledu waste"
]

for text in samples:
    prediction = model.predict(vectorizer.transform([normalize_text(text)]))[0]
    print(f"{text} -> {prediction}")
