"""
STEP 5: Groq LLM Insights Generation
Uses Groq API (llama3-70b-8192) — FREE & blazing fast

Generates 3 outputs per brand:
  1. Executive Summary     — overall snapshot of customer sentiment
  2. Topic Descriptions    — human-readable explanation of each BERTopic cluster
  3. Business Recommendations — actionable suggestions based on review patterns

Input : data/combined_topics.csv + data/topic_words.json
Output: data/groq_insights.json
"""

import json
import os
import pandas as pd
from groq import Groq

INPUT_PATH    = "data/combined_topics.csv"
TOPIC_WORDS   = "data/topic_words.json"
OUTPUT_PATH   = "data/groq_insights.json"

GROQ_API_KEY  = "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"
MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)


# ── GROQ CALL ─────────────────────────────────────────────────────────────────

def ask_groq(prompt: str, max_tokens: int = 600) -> str:
    """Send a prompt to Groq and return the response text."""
    response = client.chat.completions.create(
        model      = MODEL,
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = max_tokens,
        temperature= 0.4,
    )
    return response.choices[0].message.content.strip()


# ── PROMPT BUILDERS ───────────────────────────────────────────────────────────

def build_summary_prompt(brand: str, stats: dict, sample_reviews: list) -> str:
    samples = "\n".join([f"- {r[:120]}" for r in sample_reviews[:6]])
    return f"""You are a business analyst writing an executive summary for {brand}.

Customer review statistics:
- Total reviews analyzed: {stats['total']}
- Average rating: {stats['avg_rating']:.2f}/5
- Positive reviews: {stats['pos_pct']:.1f}%
- Neutral reviews: {stats['neu_pct']:.1f}%
- Negative reviews: {stats['neg_pct']:.1f}%
- Most common topics: {', '.join(stats['top_topics'][:4])}

Sample reviews:
{samples}

Write a crisp 3-4 sentence executive summary of customer sentiment for {brand}. 
Be specific, factual, and professional. No bullet points."""


def build_topic_description_prompt(brand: str, topic_label: str, topic_words: list, sample_reviews: list) -> str:
    samples = "\n".join([f"- {r[:100]}" for r in sample_reviews[:4]])
    words   = ", ".join(topic_words[:8])
    return f"""You are analyzing customer reviews for {brand}.

Topic cluster name: "{topic_label}"
Key words in this topic: {words}
Sample reviews from this topic:
{samples}

In 2 sentences, explain what customers are specifically talking about in this topic cluster.
Be concrete and human-readable. Start with "Customers in this group..."."""


def build_recommendations_prompt(brand: str, stats: dict, neg_topics: list, pos_topics: list) -> str:
    return f"""You are a senior business consultant for {brand} (a premium café in Bangalore).

Analysis:
- {stats['neg_pct']:.1f}% of reviews are negative
- Top complained topics: {', '.join(neg_topics[:4])}
- Top praised topics: {', '.join(pos_topics[:4])}
- Average rating: {stats['avg_rating']:.2f}/5

Give exactly 5 numbered, specific, actionable recommendations to improve {brand}'s customer experience.
Each recommendation should be 1-2 sentences. Be direct and practical. No fluff."""


def build_competitive_prompt(sb_stats: dict, tw_stats: dict) -> str:
    return f"""You are a market analyst comparing two premium cafés in Bangalore.

Starbucks Bangalore:
- Avg rating: {sb_stats['avg_rating']:.2f}/5
- Positive: {sb_stats['pos_pct']:.1f}%, Negative: {sb_stats['neg_pct']:.1f}%
- Strengths: {', '.join(sb_stats['top_pos_topics'][:3])}
- Weaknesses: {', '.join(sb_stats['top_neg_topics'][:3])}

Third Wave Coffee Bangalore:
- Avg rating: {tw_stats['avg_rating']:.2f}/5
- Positive: {tw_stats['pos_pct']:.1f}%, Negative: {tw_stats['neg_pct']:.1f}%
- Strengths: {', '.join(tw_stats['top_pos_topics'][:3])}
- Weaknesses: {', '.join(tw_stats['top_neg_topics'][:3])}

Write a 4-5 sentence competitive analysis comparing both cafés from a customer experience perspective.
Mention who leads on what dimension and what each can learn from the other. Be specific."""


# ── STATS HELPERS ─────────────────────────────────────────────────────────────

def compute_brand_stats(bdf: pd.DataFrame) -> dict:
    total = len(bdf)
    pos   = (bdf["sentiment"] == "positive").sum()
    neg   = (bdf["sentiment"] == "negative").sum()
    neu   = (bdf["sentiment"] == "neutral").sum()

    top_topics     = bdf["topic_label"].value_counts().index.tolist()
    pos_topics     = bdf[bdf["sentiment"] == "positive"]["topic_label"].value_counts().index.tolist()
    neg_topics_lst = bdf[bdf["sentiment"] == "negative"]["topic_label"].value_counts().index.tolist()

    return {
        "total"         : total,
        "avg_rating"    : bdf["rating"].mean(),
        "pos_pct"       : 100 * pos / total,
        "neu_pct"       : 100 * neu / total,
        "neg_pct"       : 100 * neg / total,
        "top_topics"    : [t for t in top_topics if t != "Miscellaneous"][:5],
        "top_pos_topics": [t for t in pos_topics if t != "Miscellaneous"][:4],
        "top_neg_topics": [t for t in neg_topics_lst if t != "Miscellaneous"][:4],
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ {INPUT_PATH} not found. Run step4_bertopic.py first.")
        return

    print("📥 Loading data...")
    df = pd.read_csv(INPUT_PATH)

    with open(TOPIC_WORDS, "r") as f:
        topic_words_all = json.load(f)

    brands  = df["brand"].unique().tolist()
    results = {}
    stats_by_brand = {}

    for brand in brands:
        print(f"\n{'='*55}")
        print(f"🤖 Generating Groq insights for: {brand}")
        print(f"{'='*55}")

        bdf   = df[df["brand"] == brand].copy()
        stats = compute_brand_stats(bdf)
        stats_by_brand[brand] = stats

        brand_key   = "starbucks" if "starbucks" in brand.lower() else "thirdwave"
        brand_words = topic_words_all.get(brand, topic_words_all.get(brand_key, {}))

        # 1️⃣ Executive Summary
        print("   📝 Executive summary...")
        sample_reviews = bdf["review_text"].dropna().sample(min(6, len(bdf)), random_state=42).tolist()
        summary = ask_groq(build_summary_prompt(brand, stats, sample_reviews))

        # 2️⃣ Topic Descriptions
        print("   🏷️  Topic descriptions...")
        topic_descriptions = {}
        unique_topics = bdf[bdf["topic_label"] != "Miscellaneous"]["topic_label"].unique()

        for topic_label in unique_topics[:8]:
            topic_reviews = bdf[bdf["topic_label"] == topic_label]["review_text"].dropna().head(4).tolist()
            # Find matching topic words
            words = []
            for tid, wlist in brand_words.items():
                if wlist:
                    words = wlist
                    break

            desc = ask_groq(
                build_topic_description_prompt(brand, topic_label, words, topic_reviews),
                max_tokens=150
            )
            topic_descriptions[topic_label] = desc
            print(f"      ✅ {topic_label}")

        # 3️⃣ Recommendations
        print("   💡 Business recommendations...")
        recs = ask_groq(
            build_recommendations_prompt(brand, stats, stats["top_neg_topics"], stats["top_pos_topics"])
        )

        results[brand] = {
            "stats"             : stats,
            "executive_summary" : summary,
            "topic_descriptions": topic_descriptions,
            "recommendations"   : recs,
        }

    # 4️⃣ Competitive Analysis (between both brands)
    if len(brands) >= 2:
        print(f"\n⚔️  Generating competitive analysis...")
        sb_brand = [b for b in brands if "starbucks" in b.lower()][0]
        tw_brand = [b for b in brands if "third" in b.lower() or "wave" in b.lower()][0]

        competitive = ask_groq(
            build_competitive_prompt(stats_by_brand[sb_brand], stats_by_brand[tw_brand]),
            max_tokens=400
        )
        results["competitive_analysis"] = competitive
        print("   ✅ Competitive analysis done!")

    # Save
    results["generated_at"] = pd.Timestamp.now().isoformat()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All Groq insights saved → {OUTPUT_PATH}")
    print("\n🎉 Step 5 complete! Run: streamlit run app.py")


if __name__ == "__main__":
    main()