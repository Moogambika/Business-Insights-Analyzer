"""
Streamlit Dashboard — Starbucks vs Third Wave Coffee Bangalore
Business Reputation & Insights Analyzer
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Café Insights | Starbucks vs Third Wave",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: #0c0e14;
    color: #e2e8f0;
}

.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1319 100%);
    border: 1px solid #2d3748;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 32px;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f6ad55, #ed8936);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { color: #718096; font-size: 1rem; margin-top: 6px; }

.kpi-card {
    background: #151923;
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
}
.kpi-val { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.kpi-label { font-size: 0.75rem; color: #718096; margin-top: 6px; text-transform: uppercase; letter-spacing: .08em; }

.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #f6ad55;
    text-transform: uppercase;
    letter-spacing: .12em;
    border-left: 3px solid #ed8936;
    padding-left: 10px;
    margin: 28px 0 16px 0;
}

.insight-box {
    background: #151923;
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
    line-height: 1.7;
    color: #cbd5e0;
    font-size: 0.95rem;
}

.topic-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-left: 4px solid #ed8936;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.topic-name { font-size: 0.82rem; color: #f6ad55; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
.topic-desc { color: #a0aec0; font-size: 0.9rem; margin-top: 5px; line-height: 1.5; }

.rec-card {
    background: #151923;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.rec-num { font-size: 0.7rem; color: #ed8936; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 4px; }
.rec-text { color: #cbd5e0; font-size: 0.92rem; line-height: 1.6; }

.review-card {
    background: #151923;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.review-meta { font-size: 0.78rem; color: #4a5568; margin-bottom: 6px; }
.pill-pos { background:#0d2e1a; color:#48bb78; border:1px solid #276749; border-radius:20px; padding:2px 10px; font-size:0.75rem; }
.pill-neg { background:#2d1515; color:#fc8181; border:1px solid #c53030; border-radius:20px; padding:2px 10px; font-size:0.75rem; }
.pill-neu { background:#1a1f2e; color:#90cdf4; border:1px solid #2c5282; border-radius:20px; padding:2px 10px; font-size:0.75rem; }

.vs-banner {
    background: linear-gradient(135deg, #1a1205 0%, #1a1f2e 50%, #0d1a12 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 800;
    color: #e2e8f0;
    margin-bottom: 24px;
}

section[data-testid="stSidebar"] { background: #0d0f17 !important; border-right: 1px solid #1e2535; }
</style>
""", unsafe_allow_html=True)


# ── LOADERS ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    if not os.path.exists("data/combined_topics.csv"):
        return None
    df = pd.read_csv("data/combined_topics.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data
def load_insights():
    if not os.path.exists("data/groq_insights.json"):
        return None
    with open("data/groq_insights.json") as f:
        return json.load(f)


# ── CHARTS ────────────────────────────────────────────────────────────────────

BRAND_COLORS = {
    "Starbucks Bangalore"         : "#1DB954",
    "Third Wave Coffee Bangalore" : "#F6AD55",
}

def sentiment_comparison_chart(df):
    data = df.groupby(["brand","sentiment"]).size().reset_index(name="count")
    total = df.groupby("brand").size().reset_index(name="total")
    data  = data.merge(total, on="brand")
    data["pct"] = (data["count"] / data["total"] * 100).round(1)

    colors = {"positive":"#48bb78","neutral":"#90cdf4","negative":"#fc8181"}
    fig = px.bar(data, x="brand", y="pct", color="sentiment",
                 barmode="group", color_discrete_map=colors,
                 labels={"pct":"Percentage (%)","brand":""},
                 text=data["pct"].apply(lambda x: f"{x}%"))
    fig.update_traces(textposition="outside")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#cbd5e0", legend_title="Sentiment",
                      margin=dict(t=20,b=20,l=0,r=0), height=320)
    fig.update_xaxes(gridcolor="#1e2535"); fig.update_yaxes(gridcolor="#1e2535")
    return fig


def rating_comparison_chart(df):
    avg = df.groupby("brand")["rating"].mean().reset_index()
    avg.columns = ["brand","avg_rating"]
    avg["avg_rating"] = avg["avg_rating"].round(2)
    colors = [BRAND_COLORS.get(b,"#718096") for b in avg["brand"]]
    fig = go.Figure(go.Bar(x=avg["brand"], y=avg["avg_rating"],
                           marker_color=colors,
                           text=avg["avg_rating"], textposition="outside"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#cbd5e0", yaxis=dict(range=[0,5.5],title="Avg Rating"),
                      margin=dict(t=20,b=20,l=0,r=0), height=300)
    fig.update_xaxes(gridcolor="#1e2535"); fig.update_yaxes(gridcolor="#1e2535")
    return fig


def topic_chart(df, brand):
    bdf = df[(df["brand"]==brand) & (df["topic_label"]!="Miscellaneous")]
    counts = bdf["topic_label"].value_counts().head(8).reset_index()
    counts.columns = ["topic","count"]
    color = BRAND_COLORS.get(brand,"#f6ad55")
    fig = px.bar(counts, x="count", y="topic", orientation="h",
                 color_discrete_sequence=[color])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#cbd5e0", xaxis_title="Reviews",
                      yaxis_title="", margin=dict(t=10,b=20,l=0,r=0), height=320)
    fig.update_xaxes(gridcolor="#1e2535"); fig.update_yaxes(gridcolor="#1e2535")
    return fig


def sentiment_donut(df, brand):
    bdf = df[df["brand"]==brand]
    counts = bdf["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment","count"]
    colors = {"positive":"#48bb78","neutral":"#90cdf4","negative":"#fc8181"}
    fig = px.pie(counts, values="count", names="sentiment", hole=0.6,
                 color="sentiment", color_discrete_map=colors)
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      marker=dict(line=dict(color="#0c0e14",width=2)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e0",
                      showlegend=False, margin=dict(t=10,b=10,l=0,r=0), height=260)
    return fig


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    df       = load_data()
    insights = load_insights()

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div style="font-size:0.75rem;color:#f6ad55;letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px;">
            GUVI × HCL Capstone · Business Reputation & Insights Analyzer
        </div>
        <div class="hero-title">☕ Café Review Intelligence</div>
        <div class="hero-sub">Starbucks Bangalore vs Third Wave Coffee · Powered by VADER + BERTopic + Groq LLaMA3</div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ No data found. Run the full pipeline first:")
        st.code("""python step1_fetch_reviews.py
python step2_preprocess.py
python step3_sentiment.py
python step4_bertopic.py
python step5_groq_insights.py
streamlit run app.py""")
        st.stop()

    brands = df["brand"].unique().tolist()
    sb = next((b for b in brands if "starbucks" in b.lower()), brands[0])
    tw = next((b for b in brands if "third" in b.lower() or "wave" in b.lower()), brands[-1])

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ☕ Filters")
        selected_brands = st.multiselect("Brand", brands, default=brands)
        selected_sent   = st.multiselect("Sentiment", ["positive","neutral","negative"],
                                          default=["positive","neutral","negative"])
        rating_range    = st.slider("Rating", 1, 5, (1,5))
        st.markdown("---")
        st.markdown("**Pipeline:**")
        st.caption("① SerpAPI → Raw Reviews\n② NLP Cleaning (lowercase, emojis, stopwords)\n③ VADER Sentiment\n④ BERTopic Modeling\n⑤ Groq LLaMA3 Insights")

    filtered = df[
        df["brand"].isin(selected_brands) &
        df["sentiment"].isin(selected_sent) &
        df["rating"].between(rating_range[0], rating_range[1])
    ]

    # ── VS Banner ─────────────────────────────────────────────────────────────
    sb_count = len(df[df["brand"]==sb])
    tw_count = len(df[df["brand"]==tw])
    st.markdown(f"""
    <div class="vs-banner">
        <span style="color:#1DB954">Starbucks</span>
        &nbsp;&nbsp;·&nbsp; {sb_count} reviews &nbsp;
        <span style="color:#718096; font-size:1rem;">VS</span>
        &nbsp; {tw_count} reviews &nbsp;·&nbsp;
        <span style="color:#F6AD55">Third Wave Coffee</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpis = [
        (c1, f"{len(df[df['brand']==sb])}", "Starbucks Reviews", "#1DB954"),
        (c2, f"{df[df['brand']==sb]['rating'].mean():.2f}⭐", "Starbucks Avg Rating", "#f6ad55"),
        (c3, f"{100*(df[df['brand']==sb]['sentiment']=='positive').mean():.0f}%", "Starbucks Positive", "#48bb78"),
        (c4, f"{len(df[df['brand']==tw])}", "Third Wave Reviews", "#F6AD55"),
        (c5, f"{df[df['brand']==tw]['rating'].mean():.2f}⭐", "Third Wave Avg Rating", "#f6ad55"),
        (c6, f"{100*(df[df['brand']==tw]['sentiment']=='positive').mean():.0f}%", "Third Wave Positive", "#48bb78"),
    ]
    for col, val, lbl, color in kpis:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-val" style="color:{color}">{val}</div>
                <div class="kpi-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    ca, cb, cc = st.columns([1.4, 1, 1])
    with ca:
        st.markdown('<div class="section-title">Sentiment Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(sentiment_comparison_chart(df), use_container_width=True)
    with cb:
        st.markdown('<div class="section-title">Avg Rating</div>', unsafe_allow_html=True)
        st.plotly_chart(rating_comparison_chart(df), use_container_width=True)
    with cc:
        st.markdown('<div class="section-title">Starbucks Sentiment</div>', unsafe_allow_html=True)
        st.plotly_chart(sentiment_donut(df, sb), use_container_width=True)

    # ── BERTopic themes ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""<h2 style="font-family:'Outfit';font-weight:800;color:#e2e8f0;font-size:1.5rem;">
        🧠 BERTopic — Discovered Themes
    </h2>
    <p style="color:#718096;font-size:0.9rem;">BERT understands sentence context from both left & right sides → clusters reviews into human-readable topics</p>
    """, unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f'<div class="section-title">☕ Starbucks Bangalore</div>', unsafe_allow_html=True)
        st.plotly_chart(topic_chart(df, sb), use_container_width=True)
    with t2:
        st.markdown(f'<div class="section-title">🌊 Third Wave Coffee</div>', unsafe_allow_html=True)
        st.plotly_chart(topic_chart(df, tw), use_container_width=True)

    # ── Groq LLM Insights ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""<h2 style="font-family:'Outfit';font-weight:800;color:#e2e8f0;font-size:1.5rem;">
        🤖 Groq LLaMA3 — AI Insights
    </h2>
    <p style="color:#718096;font-size:0.9rem;">Generated by llama3-70b-8192 via Groq API based on VADER sentiment + BERTopic themes</p>
    """, unsafe_allow_html=True)

    if insights:
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Executive Summaries", "🏷️ Topic Descriptions", "💡 Recommendations", "⚔️ Competitive Analysis"])

        with tab1:
            for brand in brands:
                if brand in insights:
                    st.markdown(f'<div class="section-title">{brand}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="insight-box">{insights[brand]["executive_summary"]}</div>', unsafe_allow_html=True)

        with tab2:
            col_a, col_b = st.columns(2)
            cols = {sb: col_a, tw: col_b}
            for brand, col in cols.items():
                with col:
                    st.markdown(f'<div class="section-title">{brand}</div>', unsafe_allow_html=True)
                    if brand in insights:
                        for topic, desc in insights[brand].get("topic_descriptions", {}).items():
                            st.markdown(f"""<div class="topic-card">
                                <div class="topic-name">{topic}</div>
                                <div class="topic-desc">{desc}</div>
                            </div>""", unsafe_allow_html=True)

        with tab3:
            col_a, col_b = st.columns(2)
            cols = {sb: col_a, tw: col_b}
            for brand, col in cols.items():
                with col:
                    st.markdown(f'<div class="section-title">{brand}</div>', unsafe_allow_html=True)
                    if brand in insights:
                        recs_text = insights[brand].get("recommendations","")
                        lines = [l.strip() for l in recs_text.split("\n") if l.strip()]
                        for i, line in enumerate(lines[:5], 1):
                            line = line.lstrip("0123456789.-) ").strip()
                            if line:
                                st.markdown(f"""<div class="rec-card">
                                    <div class="rec-num">Point {i:02d}</div>
                                    <div class="rec-text">{line}</div>
                                </div>""", unsafe_allow_html=True)

        with tab4:
            comp = insights.get("competitive_analysis","")
            if comp:
                st.markdown(f'<div class="insight-box" style="font-size:1rem;line-height:1.8;">{comp}</div>', unsafe_allow_html=True)
            else:
                st.info("Run step5_groq_insights.py to generate competitive analysis.")
    else:
        st.warning("Run step5_groq_insights.py to generate AI insights.")

    # ── Review Browser ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<h2 style="font-family:Outfit;font-weight:800;font-size:1.5rem;">📝 Review Browser</h2>', unsafe_allow_html=True)

    col_brand, col_sort, col_n = st.columns([1,2,1])
    with col_brand:
        brand_filter = st.selectbox("Brand", ["All"] + brands)
    with col_sort:
        sort_by = st.selectbox("Sort", ["Date (newest)","Rating ↑","Rating ↓","Most Negative"])
    with col_n:
        n_show = st.slider("Show", 5, 50, 10)

    disp = filtered.copy()
    if brand_filter != "All":
        disp = disp[disp["brand"] == brand_filter]

    sort_map = {
        "Date (newest)" : ("date", False),
        "Rating ↑"      : ("rating", True),
        "Rating ↓"      : ("rating", False),
        "Most Negative" : ("compound", True),
    }
    sc, sa = sort_map[sort_by]
    disp = disp.sort_values(sc, ascending=sa).head(n_show)

    for _, row in disp.iterrows():
        sent       = row.get("sentiment","neutral")
        pill_class = f"pill-{sent}"
        stars      = "⭐" * int(row["rating"]) if pd.notna(row.get("rating")) else ""
        brand_icon = "☕" if "starbucks" in str(row.get("brand","")).lower() else "🌊"
        topic      = row.get("topic_label","")

        st.markdown(f"""<div class="review-card">
            <div class="review-meta">
                {brand_icon} <b style="color:#a0aec0">{row.get('user_name','Anonymous')}</b>
                &nbsp;·&nbsp; {stars}
                &nbsp;·&nbsp; {str(row.get('date',''))[:10] or 'Unknown date'}
                &nbsp;&nbsp;<span class="{pill_class}">{sent}</span>
                &nbsp; <span style="font-size:0.75rem;color:#4a5568">VADER: {row.get('compound',0):.3f}</span>
            </div>
            <div style="color:#cbd5e0;font-size:0.93rem;line-height:1.6;margin:8px 0;">
                {str(row.get('review_text',''))[:400]}
            </div>
            {"<div style='font-size:0.75rem;color:#718096;margin-top:4px;'>🏷️ " + topic + "</div>" if topic and topic != "Miscellaneous" else ""}
        </div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""<div style="text-align:center;color:#2d3748;font-size:0.8rem;padding:24px 0;margin-top:40px;">
        GUVI × HCL Capstone · Business Reputation & Insights Analyzer<br/>
        Starbucks Bangalore vs Third Wave Coffee · SerpAPI · VADER · BERTopic · Groq LLaMA3-70b · Streamlit
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()