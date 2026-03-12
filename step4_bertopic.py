"""
STEP 4: BERTopic Modeling
BERT = Bidirectional Encoder Representations from Transformers
      → understands context from BOTH left AND right sides of a sentence

Pipeline:
  1. BERT embeddings (sentence-transformers) — understand sentence context
  2. UMAP — reduce embedding dimensions
  3. HDBSCAN — cluster similar reviews together
  4. c-TF-IDF — extract key words per cluster
  → Output: Human-readable topic labels per review

Input : data/combined_sentiment.csv
Output: data/combined_topics.csv
        data/topic_info.csv  (topic summary table)
"""

import pandas as pd
import os
import json
import warnings
warnings.filterwarnings("ignore")

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

INPUT_PATH        = "data/combined_sentiment.csv"
OUTPUT_PATH       = "data/combined_topics.csv"
TOPIC_INFO_PATH   = "data/topic_info.csv"
TOPIC_WORDS_PATH  = "data/topic_words.json"

# ── BERTopic CONFIG ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Fast, lightweight, great for reviews
MIN_TOPIC_SIZE  = 3                      # Min reviews per topic
N_TOPICS        = 8                      # Target number of topics per brand


def build_bertopic_model() -> BERTopic:
    """Build BERTopic with custom components for better café review topics."""

    # UMAP: dimensionality reduction
    umap_model = UMAP(
        n_neighbors    = 10,
        n_components   = 5,
        min_dist       = 0.0,
        metric         = "cosine",
        random_state   = 42,
    )

    # HDBSCAN: clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size    = MIN_TOPIC_SIZE,
        metric              = "euclidean",
        cluster_selection_method = "eom",
        prediction_data     = True,
    )

    # Vectorizer: extract topic keywords (remove common café words for cleaner topics)
    vectorizer = CountVectorizer(
        stop_words  = "english",
        ngram_range = (1, 2),   # unigrams + bigrams
        min_df      = 2,
    )

    model = BERTopic(
        embedding_model     = EMBEDDING_MODEL,
        umap_model          = umap_model,
        hdbscan_model       = hdbscan_model,
        vectorizer_model    = vectorizer,
        top_n_words         = 10,
        nr_topics           = N_TOPICS,
        calculate_probabilities = False,
        verbose             = False,
    )
    return model


def run_bertopic_per_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Run BERTopic separately per brand so topics are brand-specific."""
    all_dfs     = []
    topic_infos = []
    topic_words_all = {}

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    for brand in df["brand"].unique():
        print(f"\n🔍 Running BERTopic for: {brand}")
        bdf  = df[df["brand"] == brand].copy()
        docs = bdf["clean_text"].fillna("").tolist()

        if len(docs) < 10:
            print(f"   ⚠️  Too few reviews ({len(docs)}) — skipping BERTopic.")
            bdf["topic_id"]    = -1
            bdf["topic_label"] = "General"
            all_dfs.append(bdf)
            continue

        print(f"   📦 Generating BERT embeddings for {len(docs)} reviews...")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)

        print(f"   🧩 Fitting BERTopic model...")
        model  = build_bertopic_model()
        topics, _ = model.fit_transform(docs, embeddings)

        # Get topic info
        topic_info = model.get_topic_info()
        topic_info["brand"] = brand

        # Build human-readable topic labels from top words
        topic_label_map = {}
        brand_topic_words = {}

        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                topic_label_map[tid] = "Miscellaneous"
                brand_topic_words[str(tid)] = []
                continue

            # Get top words for this topic
            top_words = model.get_topic(tid)
            if top_words:
                # Take top 3 meaningful words as label
                words = [w for w, _ in top_words[:5]]
                label = " · ".join(words[:3]).title()
            else:
                label = f"Topic {tid}"

            topic_label_map[tid]   = label
            brand_topic_words[str(tid)] = [w for w, _ in (top_words or [])]

        bdf["topic_id"]    = topics
        bdf["topic_label"] = [topic_label_map.get(t, "Miscellaneous") for t in topics]

        topic_info["topic_label"] = topic_info["Topic"].map(topic_label_map)
        topic_infos.append(topic_info)
        topic_words_all[brand] = brand_topic_words
        all_dfs.append(bdf)

        # Print discovered topics
        print(f"\n   📌 Topics discovered for {brand}:")
        for tid, label in topic_label_map.items():
            if tid != -1:
                count = sum(1 for t in topics if t == tid)
                print(f"      Topic {tid:2d}: {label} ({count} reviews)")

    # Combine
    result_df   = pd.concat(all_dfs, ignore_index=True)
    all_topics  = pd.concat(topic_infos, ignore_index=True) if topic_infos else pd.DataFrame()

    return result_df, all_topics, topic_words_all


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ {INPUT_PATH} not found. Run step3_sentiment.py first.")
        return

    print(f"📥 Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"   {len(df)} reviews loaded across {df['brand'].nunique()} brands.")

    result_df, topic_info_df, topic_words = run_bertopic_per_brand(df)

    # Save outputs
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n✅ Topics assigned → {OUTPUT_PATH}")

    if not topic_info_df.empty:
        topic_info_df.to_csv(TOPIC_INFO_PATH, index=False, encoding="utf-8")
        print(f"✅ Topic info → {TOPIC_INFO_PATH}")

    with open(TOPIC_WORDS_PATH, "w") as f:
        json.dump(topic_words, f, indent=2)
    print(f"✅ Topic words → {TOPIC_WORDS_PATH}")

    print(f"\n📊 Final topic distribution:")
    print(result_df.groupby(["brand", "topic_label"]).size().to_string())
    print("\n🎉 Step 4 complete! Run step5_groq_insights.py next.")


if __name__ == "__main__":
    main()