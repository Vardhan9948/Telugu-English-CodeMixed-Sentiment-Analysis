import re
import pandas as pd

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_prepare(path):
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(normalize_text)
    return df
