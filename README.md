# AI-Powered Business Insights Analyzer
### Starbucks vs Third Wave Coffee (Bangalore)

This project is an **end-to-end NLP pipeline** that converts raw Google Maps customer reviews into **actionable business intelligence**.

The system automatically collects reviews, performs sentiment analysis, discovers hidden themes using topic modeling, and generates **executive-level insights and recommendations using an LLM.**

Developed as part of the **GUVI × HCL Capstone Project**.

---

# Project Objective

Businesses receive thousands of customer reviews but often struggle to extract meaningful insights.

This project aims to:
- Analyze real customer feedback from Google Maps
- Identify customer sentiment patterns
- Discover hidden discussion themes automatically
- Generate executive summaries and business recommendations using AI

---

# Key Features

• Automated Google Maps Review Collection  
• Advanced NLP Text Cleaning Pipeline  
• Sentiment Analysis using VADER  
• Topic Discovery using BERTopic  
• AI-generated Business Insights using LLaMA3  
• Interactive Visual Dashboard using Streamlit

---

# Tech Stack

**Programming Language**

- Python

**Data Collection**

- SerpAPI

**Natural Language Processing**

- NLTK
- VADER Sentiment Analysis

**Topic Modeling**

- BERTopic
- Sentence Transformers
- UMAP
- HDBSCAN

**Large Language Model**

- Groq API (LLaMA3)

**Visualization & Interface**

- Streamlit
- Plotly

---

# Project Pipeline

1. **Data Collection**
   - Google Maps reviews were collected using SerpAPI.

2. **Data Cleaning**
   - Lowercasing
   - Emoji removal
   - Punctuation removal
   - Stopword removal

3. **Sentiment Analysis**
   - Each review classified as:
     - Positive
     - Neutral
     - Negative  
   using VADER sentiment analyzer.

4. **Topic Modeling**
   - BERTopic automatically grouped reviews into themes using contextual embeddings from BERT.

   Example topics discovered:
   - Staff · Service · Friendly
   - Coffee · Taste · Quality
   - Workspace · Ambience · Study

5. **AI Insight Generation**
   - Groq LLaMA3 model generates:
     - Executive summary
     - Topic explanations
     - Business recommendations
     - Competitive analysis

6. **Visualization**
   - Streamlit dashboard presents insights interactively.

---

# Key Insights

### Overall Rating Comparison

| Brand | Average Rating |
|------|------|
| Starbucks | 3.44 / 5 |
| Third Wave Coffee | 3.83 / 5 |

### Sentiment Comparison

**Starbucks**
- 63% Positive
- Strong in staff friendliness and ambience

**Third Wave Coffee**
- 80% Positive
- Strong in coffee quality and workspace vibe

### Common Improvement Areas

- Order consistency
- Wait times during peak hours

---

# Example Topics Discovered by BERTopic

| Topic | Key Words |
|------|-----------|
| Staff Experience | staff · friendly · service |
| Coffee Quality | coffee · taste · good |
| Work Environment | ambience · workspace · laptop |

---

# How to Run the Project

### 1 Install Dependencies


pip install -r requirements.txt


### 2 Set API Keys

Add your API keys:

- SerpAPI
- Groq API

### 3 Run the Application


streamlit run app.py


---

# Future Improvements

- Multi-city café comparison
- Real-time review monitoring
- Customer complaint detection
- Automated business alert system
- Review summarization dashboard

---

# Author

**Moogambika Govindaraj**

AI / Data Science Enthusiast  
Python • NLP • Machine Learning
