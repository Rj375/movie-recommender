# Trending This Week

A movie discovery app that surfaces worldwide trending movies (via a recency-weighted popularity model over the [MovieLens](https://grouplens.org/datasets/movielens/) dataset), filterable by genre or franchise (Marvel/DC), with posters, ratings, tags, and official streaming links pulled live from [TMDb](https://www.themoviedb.org/) / [JustWatch](https://www.justwatch.com/).

There's also `movie_recommender.ipynb`, a companion notebook that builds a personalized SVD-based recommender on the same dataset.

## Run locally

```bash
pip install -r requirements.txt
```

Add your free [TMDb API key](https://www.themoviedb.org/settings/api) to `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_key_here"
```

Then:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`.streamlit/secrets.toml` is gitignored — your key never gets committed).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
3. Pick this repo/branch and set the main file path to `app.py`.
4. In the app's **Settings → Secrets**, paste:
   ```toml
   TMDB_API_KEY = "your_key_here"
   ```
5. Deploy.
