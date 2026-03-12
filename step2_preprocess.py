"""
STEP 2: NLP Text Preprocessing & Cleaning
Pipeline:
  1. Lowercase
  2. Remove emojis
  3. Remove URLs
  4. Remove punctuation
  5. Remove stopwords (NLTK)
  6. Strip extra whitespace

Input : data/starbucks_reviews.csv + data/thirdwave_reviews.csv
Output: data/starbucks_clean.csv + data/thirdwave_clean.csv
        data/combined_clean.csv  (both brands together)
"""

import pandas as pd
import re
import os
import unicodedata
import nltk

# Download NLTK stopwords if not present
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

INPUTS = [
    ("data/starbucks_reviews.csv", "data/starbucks_clean.csv"),
    ("data/thirdwave_reviews.csv", "data/thirdwave_clean.csv"),
]
COMBINED_OUTPUT = "data/combined_clean.csv"


# ── CLEANING FUNCTIONS ────────────────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    return text.lower()


def remove_emojis(text: str) -> str:
    """Remove all emoji and non-ASCII unicode symbols."""
    return "".join(
        c for c in text
        if not unicodedata.category(c).startswith(("So", "Sm", "Sk", "Cs"))
        and ord(c) < 128
    )


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\S+", "", text)


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def full_clean_pipeline(text: str) -> str:
    """Apply all cleaning steps in order."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = to_lowercase(text)
    text = remove_emojis(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = normalize_whitespace(text)
    return text


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["word_count"]    = df["clean_text"].apply(lambda x: len(x.split()))
    df["char_count"]    = df["clean_text"].apply(len)
    df["is_long"]       = df["word_count"] >= 20
    df["star_category"] = df["rating"].apply(
        lambda r: "5★" if r == 5 else ("4★" if r == 4 else ("3★" if r == 3 else "1-2★"))
        if pd.notna(r) else "Unknown"
    )
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"\n📥 Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"   {len(df)} reviews loaded.")

    print("🧹 Cleaning text...")
    df["clean_text"] = df["review_text"].apply(full_clean_pipeline)

    # Drop reviews that became empty after cleaning
    before = len(df)
    df = df[df["clean_text"].str.len() >= 5].copy()
    print(f"   Removed {before - len(df)} reviews (too short after cleaning)")

    df = add_features(df)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} clean reviews → {output_path}")
    print(f"   Avg word count: {df['word_count'].mean():.1f}")

    return df


def main():
    all_dfs = []

    for input_path, output_path in INPUTS:
        if not os.path.exists(input_path):
            print(f"⚠️  {input_path} not found — run step1 first.")
            continue
        df = preprocess(input_path, output_path)
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(COMBINED_OUTPUT, index=False, encoding="utf-8")
        print(f"\n✅ Combined dataset → {COMBINED_OUTPUT} ({len(combined)} total reviews)")

        print(f"\n📊 Brand breakdown:")
        print(combined.groupby("brand")["rating"].agg(["count", "mean"]).round(2).to_string())

    print("\n🎉 Step 2 complete! Run step3_sentiment.py next.")


if __name__ == "__main__":
    main()