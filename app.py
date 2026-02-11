import json
import os
import pickle

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import streamlit as st

from model import ContentRecommender

access_token = st.secrets["ACCESS_TOKEN"]
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {access_token}",
}
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


@st.cache_data(ttl=3600, show_spinner=False)
def get_show_details(show_id: int, media_type: str = "tv") -> dict | None:
    url = f"https://api.themoviedb.org/3/{media_type}/{show_id}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


def show_poster(show_id, media_type):
    data = get_show_details(int(show_id), media_type)
    if data and data.get("poster_path"):
        st.image(f"{IMAGE_BASE_URL}{data['poster_path']}")
    else:
        st.markdown("*No poster available*")


@st.cache_resource
def load_model(artifact_dir="model_artifacts"):
    df = pd.read_parquet(f"{artifact_dir}/processed_data.parquet")
    tfidf_matrix = sp.load_npz(f"{artifact_dir}/tfidf_matrix.npz")

    with open(f"{artifact_dir}/feature_weights.pkl", "rb") as f:
        feature_weights = pickle.load(f)

    feature_matrices = {}
    for name in feature_weights:
        feature_matrices[name] = sp.load_npz(f"{artifact_dir}/feature_{name}.npz")

    embeddings = None
    emb_path = f"{artifact_dir}/embeddings.npy"
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)

    with open(f"{artifact_dir}/config.json") as f:
        config = json.load(f)

    recommender = ContentRecommender(
        df=df,
        tfidf_matrix=tfidf_matrix,
        feature_matrices=feature_matrices,
        feature_weights=feature_weights,
        embeddings=embeddings,
    )
    recommender.compute_weighted_rating()

    return recommender, config


@st.cache_data
def get_all_genres(_df):
    all_genres = set()
    for g in _df["genres"].dropna():
        for part in g.split(","):
            s = part.strip()
            if s and s != "Unknown":
                all_genres.add(s)
    return sorted(all_genres)


@st.cache_data
def get_landing_page_data(_df):
    sections = {}

    recent = _df[(_df["release_year"] >= 2023) & (_df["vote_count"] > 50)]
    sections["trending"] = recent.nlargest(10, "popularity")

    quality = _df[_df["vote_count"] > 100]
    sections["top_rated"] = quality.nlargest(10, "weighted_rating")

    featured_genres = [
        "Action", "Comedy", "Animation", "Thriller",
        "Drama", "Horror", "Science Fiction", "Romance",
    ]
    genre_sections = {}
    for genre in featured_genres:
        genre_mask = _df["genres"].str.contains(genre, case=False, na=False)
        genre_df = _df[genre_mask & (_df["vote_count"] > 50)]
        genre_sections[genre] = genre_df.nlargest(10, "weighted_rating")
    sections["genres"] = genre_sections

    return sections


def render_content_row(section_df, section_title):
    st.subheader(section_title)
    if section_df.empty:
        st.caption("No items available.")
        return
    cols = st.columns(5)
    for i, (_, row) in enumerate(section_df.head(10).iterrows()):
        with cols[i % 5]:
            media_type = "movie" if row["type"] == "Movie" else "tv"
            show_poster(row["show_id"], media_type)
            st.markdown(f"**{row['title']}**")
            year = int(row["release_year"]) if pd.notna(row.get("release_year")) else ""
            st.caption(f"{row['type']} | {year}")
            st.caption(f"{row.get('genres', 'N/A')}")
            rating = row.get("weighted_rating")
            if pd.notna(rating):
                st.caption(f"Rating: {rating:.1f}")


def main():
    st.set_page_config(page_title="Netflix Recommendations", layout="wide")
    st.title("Netflix Recommendation System")

    recommender, config = load_model()
    best_approach = config.get("best_approach", "combined")
    hybrid_weights = tuple(config.get("hybrid_weights", [0.5, 0.5]))

    approach_key_map = {
        "Combined (TF-IDF Soup)": "combined",
        "Separate (TF-IDF Weighted)": "separate",
        "Embedding (SBERT Tag)": "embedding",
        "Embedding (SBERT Desc)": "embedding",
        "Hybrid (TF-IDF + SBERT Tag)": "hybrid",
        "Hybrid (TF-IDF + SBERT Desc)": "hybrid",
    }
    approach = approach_key_map.get(best_approach, best_approach)

    all_genres = get_all_genres(recommender.df)

    title = st.text_input("Search for a movie or TV show:")
    selected_genres = st.multiselect("Filter by genre:", all_genres)

    if title:
        query = title.strip()
        matches = recommender.df[recommender.df["title"].str.lower() == query.lower()]
        if matches.empty:
            matches = recommender.df[
                recommender.df["title"].str.contains(query, case=False, na=False, regex=False)
            ]

        if matches.empty:
            st.warning(f"'{title}' not found in the database. Try a different title.")
        else:
            if len(matches) > 1:
                matches = matches.sort_values("popularity", ascending=False)
                options = matches.head(20).apply(
                    lambda r: f"{r['title']} ({r['type']}, {int(r['release_year'])})", axis=1
                ).tolist()
                selected = st.selectbox("Multiple matches found. Select one:", options)
                selected_idx = options.index(selected)
                input_info = matches.iloc[selected_idx]
            else:
                input_info = matches.iloc[0]

            st.subheader(input_info["title"])
            hero_col1, hero_col2 = st.columns([1, 3])
            with hero_col1:
                media_type = "movie" if input_info["type"] == "Movie" else "tv"
                show_poster(input_info["show_id"], media_type)
            with hero_col2:
                st.markdown(f"**{input_info['type']}** | {int(input_info['release_year'])}")
                st.markdown(f"**Genres:** {input_info['genres']}")
                rating = input_info.get("weighted_rating")
                if pd.notna(rating):
                    st.markdown(f"**Rating:** {rating:.1f}")
                desc = input_info.get("description", "")
                if desc:
                    st.markdown(desc)

            fetch_k = 50 if selected_genres else 10
            results = recommender.recommend(
                input_info["title"],
                top_k=fetch_k,
                approach=approach,
                content_type=input_info["type"],
                year=int(input_info["release_year"]),
                hybrid_weights=hybrid_weights,
            )

            if results is not None:
                if selected_genres:
                    mask = results["genres"].apply(
                        lambda g: any(sg in g for sg in selected_genres)
                        if pd.notna(g) else False
                    )
                    results = results[mask]

                results = results.head(10)

                if results.empty:
                    st.info("No recommendations match the selected genres. Try broadening your filter.")
                else:
                    st.subheader(f"More Like This ({len(results)} results)")
                    cols = st.columns(5)
                    for i, (idx, row) in enumerate(results.iterrows()):
                        col = cols[i % 5]
                        media_type = "movie" if row["type"] == "Movie" else "tv"
                        show_id = recommender.df.loc[idx, "show_id"]

                        with col:
                            show_poster(show_id, media_type)

                            st.markdown(f"**{row['title']}**")
                            st.caption(f"{row['type']} | {row.get('genres', 'N/A')}")

                            score_text = ""
                            if "final_score" in row:
                                score_text += f"Score: {row['final_score']:.3f}"
                            if "weighted_rating" in row:
                                score_text += f" | Rating: {row['weighted_rating']}"
                            if score_text:
                                st.caption(score_text)
    else:
        landing_data = get_landing_page_data(recommender.df)

        def _apply_genre_filter(df):
            if not selected_genres:
                return df
            mask = df["genres"].apply(
                lambda g: any(sg in g for sg in selected_genres)
                if pd.notna(g) else False
            )
            return df[mask]

        render_content_row(
            _apply_genre_filter(landing_data["trending"]), "Trending Now"
        )
        render_content_row(
            _apply_genre_filter(landing_data["top_rated"]), "Top Rated"
        )

        for genre, genre_df in landing_data["genres"].items():
            if not genre_df.empty:
                render_content_row(
                    _apply_genre_filter(genre_df), f"Top {genre}"
                )


if __name__ == "__main__":
    main()
