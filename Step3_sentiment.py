"""
STEP 3: VADER Sentiment Analysis
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) — 
perfect for short social/review text. No model download needed.

Outputs:
  - sentiment label: positive / neutral / negative
  - compound score : -1.0 to +1.0
  - pos / neu / neg scores

Input : data/combined_clean.csv
Output: data/combined_sentiment.csv
"""

import pandas as pd
import os
import nltk

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

INPUT_PATH  = "data/combined_clean.csv"
OUTPUT_PATH = "data/combined_sentiment.csv"

# VADER thresholds (standard)
POS_THRESHOLD =  0.05
NEG_THRESHOLD = -0.05


def classify_sentiment(compound: float) -> str:
    if compound >= POS_THRESHOLD:
        return "positive"
    elif compound <= NEG_THRESHOLD:
        return "negative"
    else:
        return "neutral"


def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    print(f"🔬 Running VADER on {len(df)} reviews...")

    scores = df["clean_text"].apply(lambda t: sia.polarity_scores(t) if t else {"compound": 0, "pos": 0, "neu": 1, "neg": 0})

    df = df.copy()
    df["compound"]  = scores.apply(lambda s: round(s["compound"], 4))
    df["pos_score"] = scores.apply(lambda s: round(s["pos"], 4))
    df["neu_score"] = scores.apply(lambda s: round(s["neu"], 4))
    df["neg_score"] = scores.apply(lambda s: round(s["neg"], 4))
    df["sentiment"] = df["compound"].apply(classify_sentiment)

    return df


def print_summary(df: pd.DataFrame):
    print(f"\n📊 Sentiment Summary by Brand:")
    print("=" * 55)
    for brand in df["brand"].unique():
        bdf = df[df["brand"] == brand]
        total = len(bdf)
        pos   = (bdf["sentiment"] == "positive").sum()
        neu   = (bdf["sentiment"] == "neutral").sum()
        neg   = (bdf["sentiment"] == "negative").sum()
        avg_c = bdf["compound"].mean()
        print(f"\n  🏷️  {brand}")
        print(f"     Total    : {total} reviews")
        print(f"     Positive : {pos} ({100*pos/total:.1f}%)")
        print(f"     Neutral  : {neu} ({100*neu/total:.1f}%)")
        print(f"     Negative : {neg} ({100*neg/total:.1f}%)")
        print(f"     Avg VADER compound: {avg_c:.3f}")


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ {INPUT_PATH} not found. Run step2_preprocess.py first.")
        return

    df = pd.read_csv(INPUT_PATH)
    df = run_vader(df)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n✅ Saved → {OUTPUT_PATH}")
    print_summary(df)
    print("\n🎉 Step 3 complete! Run step4_bertopic.py next.")


if __name__ == "__main__":
    main()