import re

import numpy as np
import pandas as pd
import requests
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "ml-latest-small"
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

# Fallback if the TMDb countries lookup is unavailable (no key / offline) — a small curated set
# so the region picker still works rather than being empty.
FALLBACK_REGIONS = {
    "US": "United States", "IN": "India", "GB": "United Kingdom", "CA": "Canada",
    "AU": "Australia", "DE": "Germany", "FR": "France", "JP": "Japan", "BR": "Brazil", "MX": "Mexico",
}

# TMDb's free API doesn't provide per-title deep links, so each provider is matched to its own
# OFFICIAL homepage instead of guessing a search URL. Anything not in this list falls back to
# TMDb's own watch page (never a third-party aggregator) so every link stays on a legitimate site.
OFFICIAL_PLATFORM_HOME = {
    "netflix": "https://www.netflix.com",
    "amazon prime video": "https://www.primevideo.com",
    "prime video": "https://www.primevideo.com",
    "disney plus": "https://www.disneyplus.com",
    "disney+": "https://www.disneyplus.com",
    "disney+ hotstar": "https://www.hotstar.com",
    "hotstar": "https://www.hotstar.com",
    "hulu": "https://www.hulu.com",
    "hbo max": "https://www.max.com",
    "max": "https://www.max.com",
    "apple tv": "https://tv.apple.com",
    "apple tv plus": "https://tv.apple.com",
    "apple tv store": "https://tv.apple.com",
    "google play movies": "https://play.google.com/store/movies",
    "youtube": "https://www.youtube.com",
    "paramount plus": "https://www.paramountplus.com",
    "peacock": "https://www.peacocktv.com",
    "jiocinema": "https://www.jiocinema.com",
    "sonyliv": "https://www.sonyliv.com",
    "zee5": "https://www.zee5.com",
    "microsoft store": "https://www.microsoft.com/en-us/store/movies-and-tv",
    "vudu": "https://www.vudu.com",
}


def platform_link(provider_name, fallback_link):
    key = provider_name.strip().lower()
    for name, home_url in OFFICIAL_PLATFORM_HOME.items():
        if name in key or key in name:
            return home_url
    return fallback_link or "https://www.themoviedb.org"

st.set_page_config(page_title="Trending Movies This Week", page_icon="🎬", layout="centered")

# MovieLens has no franchise tags, so these are matched against titles by keyword instead of genre.
FRANCHISE_KEYWORDS = {
    "Marvel": ["avengers", "iron man", "spider-man", "spiderman", "captain america", "thor", "x-men",
               "guardians of the galaxy", "ant-man", "black panther", "doctor strange", "deadpool",
               "wolverine", "fantastic four", "hulk", "captain marvel", "eternals", "shang-chi"],
    "DC": ["batman", "superman", "wonder woman", "justice league", "aquaman", "suicide squad",
           "green lantern", "watchmen", "man of steel", "shazam", "catwoman"],
}


@st.cache_data
def load_data():
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    links = pd.read_csv(DATA_DIR / "links.csv", dtype={"imdbId": "Int64", "tmdbId": "Int64"})
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    return ratings, movies, tags, links


@st.cache_data
def compute_movie_details(ratings, tags, links):
    overall = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", num_ratings="count").reset_index()

    top_tags = (
        tags.groupby("movieId")["tag"]
        .apply(lambda s: ", ".join(s.value_counts().head(3).index))
        .reset_index(name="top_tags")
    )

    links = links.copy()
    links["imdb_url"] = links["imdbId"].apply(lambda x: f"https://www.imdb.com/title/tt{x:07d}/" if pd.notna(x) else None)
    links["tmdb_url"] = links["tmdbId"].apply(lambda x: f"https://www.themoviedb.org/movie/{int(x)}" if pd.notna(x) else None)

    details = overall.merge(top_tags, on="movieId", how="left").merge(
        links[["movieId", "tmdbId", "imdb_url", "tmdb_url"]], on="movieId", how="left"
    )
    return details


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_poster_url(tmdb_id, api_key):
    if not tmdb_id or not api_key:
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": api_key},
            timeout=5,
        )
        resp.raise_for_status()
        poster_path = resp.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else None
    except requests.RequestException:
        return None


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def fetch_regions(api_key):
    if not api_key:
        return FALLBACK_REGIONS
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/configuration/countries",
            params={"api_key": api_key},
            timeout=5,
        )
        resp.raise_for_status()
        countries = {c["iso_3166_1"]: c["english_name"] for c in resp.json()}
        return countries or FALLBACK_REGIONS
    except requests.RequestException:
        return FALLBACK_REGIONS


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_watch_providers(tmdb_id, api_key, region):
    if not tmdb_id or not api_key:
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/watch/providers",
            params={"api_key": api_key},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json().get("results", {}).get(region)
        if not data:
            return None
        return {
            "link": data.get("link"),
            "flatrate": data.get("flatrate", []),
            "rent": data.get("rent", []),
            "buy": data.get("buy", []),
        }
    except requests.RequestException:
        return None


@st.cache_data
def compute_trending(ratings, movies, window_days=90, half_life_days=14):
    now = ratings["timestamp"].max()
    recent = ratings[ratings["timestamp"] >= now - pd.Timedelta(days=window_days)].copy()
    recent["age_days"] = (now - recent["timestamp"]).dt.total_seconds() / 86400
    recent["decay"] = 0.5 ** (recent["age_days"] / half_life_days)

    buzz = recent.groupby("movieId").apply(
        lambda g: pd.Series({
            "weighted_count": g["decay"].sum(),
            "weighted_avg_rating": np.average(g["rating"], weights=g["decay"]),
        }),
        include_groups=False,
    ).reset_index()

    C = buzz["weighted_count"].quantile(0.6)
    m = ratings["rating"].mean()
    buzz["trend_score"] = (
        (buzz["weighted_count"] / (buzz["weighted_count"] + C)) * buzz["weighted_avg_rating"]
        + (C / (buzz["weighted_count"] + C)) * m
    )
    return buzz.merge(movies, on="movieId").sort_values("trend_score", ascending=False).reset_index(drop=True)


def matches_category(title, genres, category):
    if category in FRANCHISE_KEYWORDS:
        return any(re.search(rf"\b{re.escape(kw)}\b", title, re.I) for kw in FRANCHISE_KEYWORDS[category])
    return category in genres.split("|")


ratings, movies, tags, links = load_data()
trending = compute_trending(ratings, movies)
details = compute_movie_details(ratings, tags, links)
trending = trending.merge(details, on="movieId", how="left")

genre_list = sorted({g for genres in movies["genres"].str.split("|") for g in genres if g != "(no genres listed)"})
categories = list(FRANCHISE_KEYWORDS.keys()) + genre_list

st.markdown(
    """
    <style>
    img { max-width: 100%; height: auto; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Trending Movies This Week")
st.caption("Worldwide trending movies, filterable by genre or franchise.")

if not TMDB_API_KEY:
    st.info(
        "No TMDb API key configured, so posters won't load. Add one to "
        "`.streamlit/secrets.toml` as `TMDB_API_KEY = \"your_key\"` to enable them.",
        icon="🖼️",
    )

regions = fetch_regions(TMDB_API_KEY)
region_order = sorted(regions, key=lambda c: regions[c])

with st.sidebar:
    st.header("Filters")
    selected = st.multiselect("Genre / Franchise", categories, placeholder="e.g. Marvel, Horror, Romance...")
    search = st.text_input("Search by title", placeholder="e.g. Batman")
    top_n = st.slider("Number of results", 5, 30, 10)
    region = st.selectbox(
        "Your country (for streaming availability)",
        region_order,
        index=region_order.index("US") if "US" in region_order else 0,
        format_func=lambda c: f"{regions[c]} ({c})",
    )

results = trending
if selected:
    mask = results.apply(lambda r: any(matches_category(r["title"], r["genres"], c) for c in selected), axis=1)
    results = results[mask]
if search:
    results = results[results["title"].str.contains(search, case=False, na=False)]
results = results.head(top_n)

if results.empty:
    st.warning("No trending movies match those filters. Try removing a filter or widening your search.")
else:
    for _, row in results.iterrows():
        with st.container(border=True):
            poster_col, info_col = st.columns([1, 3])

            with poster_col:
                poster_url = fetch_poster_url(row.get("tmdbId"), TMDB_API_KEY)
                if poster_url:
                    st.markdown(
                        f"<img src='{poster_url}' style='width:100%;max-width:200px;display:block;"
                        "margin:0 auto;border-radius:8px;'/>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div style='width:100%;max-width:200px;aspect-ratio:2/3;background:#3332;"
                        "border-radius:8px;display:flex;align-items:center;justify-content:center;"
                        "font-size:2rem;margin:0 auto;'>🎬</div>",
                        unsafe_allow_html=True,
                    )

            with info_col:
                st.markdown(f"### {row['title']}")
                st.caption(row["genres"].replace("|", " · "))
                if pd.notna(row.get("top_tags")) and row["top_tags"]:
                    st.caption(f"🏷️ {row['top_tags']}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Trending score", f"{row['trend_score']:.2f}")
                c2.metric("Avg rating", f"{row['avg_rating']:.1f} ⭐" if pd.notna(row.get("avg_rating")) else "—")
                c3.metric("Total ratings", f"{int(row['num_ratings']):,}" if pd.notna(row.get("num_ratings")) else "—")

                providers = fetch_watch_providers(row.get("tmdbId"), TMDB_API_KEY, region)
                where = providers.get("flatrate") or providers.get("rent") or providers.get("buy") if providers else None
                if where:
                    label = "Stream on" if providers.get("flatrate") else "Rent/buy on"
                    st.caption(f"📺 {label}:")
                    badges_html = "<div style='display:flex;flex-wrap:wrap;gap:10px;'>"
                    for p in where[:6]:
                        url = platform_link(p["provider_name"], providers.get("link"))
                        badges_html += (
                            f"<a href='{url}' target='_blank' rel='noopener' "
                            "style='text-decoration:none;color:inherit;display:flex;flex-direction:column;"
                            "align-items:center;width:60px;'>"
                            f"<img src='https://image.tmdb.org/t/p/w45{p['logo_path']}' width='32' height='32' "
                            "style='border-radius:6px;'/>"
                            f"<span style='text-align:center;font-size:0.7rem;line-height:1.1;margin-top:2px;'>"
                            f"{p['provider_name']}</span></a>"
                        )
                    badges_html += "</div>"
                    st.markdown(badges_html, unsafe_allow_html=True)
                    st.caption("Streaming data via [JustWatch](https://www.justwatch.com/)")
                elif TMDB_API_KEY:
                    st.caption(f"📺 Not currently available to stream in {region}")

                link_cols = st.columns(2)
                if pd.notna(row.get("imdb_url")):
                    link_cols[0].link_button("IMDb ↗", row["imdb_url"], use_container_width=True)
                if pd.notna(row.get("tmdb_url")):
                    link_cols[1].link_button("TMDb ↗", row["tmdb_url"], use_container_width=True)
