from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from preprocess import load_and_prepare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telugu_english_sentiment.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = load_and_prepare(DATA)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"],
    test_size=0.25, random_state=42, stratify=df["sentiment"]
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, pred)

print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

(OUT / "classification_report.txt").write_text(
    f"Accuracy: {accuracy:.2%}\n\n" + classification_report(y_test, pred),
    encoding="utf-8"
)

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Telugu-English Code-Mixed Sentiment")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=200)
plt.close()

joblib.dump(model, OUT / "sentiment_model.joblib")
joblib.dump(vectorizer, OUT / "tfidf_vectorizer.joblib")

print("\nSaved model, vectorizer, report and confusion matrix in outputs/")
