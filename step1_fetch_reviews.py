"""
STEP 1: Fetch Reviews via SerpAPI
Fetches Google Maps reviews for:
  - Starbucks Bangalore (MG Road)
  - Third Wave Coffee Bangalore (competitor)
Output: data/starbucks_reviews.csv
        data/thirdwave_reviews.csv
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
SERP_API_KEY = "3f0485886bb7ee26addca8c95de3197cef03044a41a0023f51e209c45bcd8cf6"

BUSINESSES = {
    "starbucks": {
        "name"    : "Starbucks Bangalore",
        "query"   : "Starbucks MG Road Bangalore",
        "output"  : "data/starbucks_reviews.csv",
    },
    "thirdwave": {
        "name"    : "Third Wave Coffee Bangalore",
        "query"   : "Third Wave Coffee Roasters Bangalore Indiranagar",
        "output"  : "data/thirdwave_reviews.csv",
    },
}

MAX_PAGES = 5   # ~50 reviews per brand


# ── HELPERS ───────────────────────────────────────────────────────────────────

def search_place_id(query: str, api_key: str) -> str | None:
    """Search Google Maps to get the place_id for a business query."""
    params = {
        "engine" : "google_maps",
        "q"      : query,
        "type"   : "search",
        "api_key": api_key,
    }
    r = requests.get("https://serpapi.com/search.json", params=params)
    r.raise_for_status()
    results = r.json().get("local_results", [])
    if results:
        place_id = results[0].get("place_id")
        title    = results[0].get("title", "")
        print(f"   📍 Found: {title} → place_id: {place_id}")
        return place_id
    return None


def fetch_reviews_page(place_id: str, api_key: str, next_page_token: str = None) -> dict:
    params = {
        "engine"   : "google_maps_reviews",
        "place_id" : place_id,
        "hl"       : "en",
        "sort_by"  : "newestFirst",
        "api_key"  : api_key,
    }
    if next_page_token:
        params["next_page_token"] = next_page_token
    r = requests.get("https://serpapi.com/search.json", params=params)
    r.raise_for_status()
    return r.json()


def fetch_all_reviews(place_id: str, api_key: str, brand_name: str, max_pages: int = 5) -> list:
    all_reviews     = []
    next_page_token = None

    for page in range(1, max_pages + 1):
        print(f"   📄 Page {page}...", end=" ")
        data    = fetch_reviews_page(place_id, api_key, next_page_token)
        reviews = data.get("reviews", [])
        if not reviews:
            print("no reviews found.")
            break
        all_reviews.extend(reviews)
        print(f"✅ {len(reviews)} reviews (total: {len(all_reviews)})")

        pagination      = data.get("serpapi_pagination", {})
        next_page_token = pagination.get("next_page_token")
        if not next_page_token:
            break
        time.sleep(1.5)

    return all_reviews


def parse_reviews(raw: list, brand_name: str) -> pd.DataFrame:
    rows = []
    for r in raw:
        rows.append({
            "brand"      : brand_name,
            "user_name"  : r.get("user", {}).get("name", "Anonymous"),
            "rating"     : r.get("rating", None),
            "date"       : r.get("date", None),
            "review_text": r.get("snippet", ""),
            "likes"      : r.get("likes", 0),
            "fetched_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    df = pd.DataFrame(rows)
    df = df[df["review_text"].str.strip().str.len() > 0]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.reset_index(drop=True, inplace=True)
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    for key, config in BUSINESSES.items():
        print(f"\n{'='*55}")
        print(f"🔍 Fetching: {config['name']}")
        print(f"{'='*55}")

        # Auto-discover place_id
        place_id = search_place_id(config["query"], SERP_API_KEY)
        if not place_id:
            print(f"❌ Could not find place_id for {config['name']}. Skipping.")
            continue

        raw = fetch_all_reviews(place_id, SERP_API_KEY, config["name"], MAX_PAGES)
        if not raw:
            print(f"❌ No reviews fetched for {config['name']}.")
            continue

        df = parse_reviews(raw, config["name"])
        df.to_csv(config["output"], index=False, encoding="utf-8")
        print(f"\n✅ Saved {len(df)} reviews → {config['output']}")
        print(f"   Avg rating: {df['rating'].mean():.2f} ⭐")
        print(f"   Rating dist:\n{df['rating'].value_counts().sort_index().to_string()}")

    print("\n\n🎉 Step 1 complete! Run step2_preprocess.py next.")


if __name__ == "__main__":
    main()