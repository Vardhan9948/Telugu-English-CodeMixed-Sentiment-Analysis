from pathlib import Path
import joblib
from preprocess import normalize_text

ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "outputs" / "sentiment_model.joblib")
vectorizer = joblib.load(ROOT / "outputs" / "tfidf_vectorizer.joblib")

print("Telugu-English Sentiment Predictor")
print("Type 'exit' to stop.\n")

while True:
    text = input("Enter code-mixed text: ")
    if text.lower() == "exit":
        break

    clean = normalize_text(text)
    prediction = model.predict(vectorizer.transform([clean]))[0]
    print("Predicted sentiment:", prediction.upper(), "\n")
