import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env using absolute path
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)
os.environ["TMDB_API_KEY"] = os.getenv("TMDB_API_KEY", "")

from src.data_loader import load_data
from src.recommender import MovieRecommender
from src.api_client import get_movie_details

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide",
)


@st.cache_resource(show_spinner="Building recommendation models...")
def load_everything():
    df_movies, df_ratings = load_data(
        movies_path="movies.csv",
        ratings_path="ratings.csv",
        links_path="links.csv",
    )
    rec = MovieRecommender(df_movies)  # type: ignore
    titles = sorted(df_movies["title"].dropna().tolist())
    return rec, titles


recommender, ALL_TITLES = load_everything()

st.title("Movie Recommendation System")
st.caption("Find movies similar to ones you love — powered by TF-IDF, BERT, and a Hybrid of both.")
st.divider()

col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    typed = st.text_input(
        "Type a movie name",
        placeholder="e.g. Inception, Avatar...",
    )
    picked = st.selectbox(
        "Or pick from the list",
        options=ALL_TITLES,
        index=None,
        placeholder="Browse and select...",
    )

    # Typed input takes priority over dropdown selection
    movie_name = typed.strip() if typed.strip() else picked

with col2:
    method = st.radio(
        "Algorithm",
        options=["TF-IDF", "BERT (Semantic)", "Hybrid"],
        index=2,
        horizontal=True,
        help=(
            "TF-IDF: keyword matching · "
            "BERT: semantic/plot similarity · "
            "Hybrid: blend of both"
        ),
    )

with col3:
    num_results = st.slider("Results", min_value=5, max_value=20, value=10)

find = st.button("Find Similar Movies", type="primary", use_container_width=True)

st.divider()

if find:
    with st.spinner("Finding recommendations..."):
    # Not showing movies with no poster
        results = recommender.get_recommendations(  # type: ignore
            movie_name, method=method, top_n=num_results + 10
        )

    if results is None:
        st.error(f"No match found for **'{movie_name}'**. Try a different title.")
    else:
        matched = results[0]["matched_title"]
        st.success(f"**{len(results)} recommendations** for **{matched}** using {method}")

        with st.spinner("Fetching posters and details..."):
            enriched = []
            for rec in results:
                tmdb_id = rec.get("tmdbId", 0)
                if tmdb_id and tmdb_id > 0:
                    details: dict = get_movie_details(tmdb_id)
                else:
                    details = {"poster_url": None, "rating": None, "year": "N/A", "overview": ""}
                merged = {**rec, **details}
                merged["overview"] = details.get("overview") or rec.get("dataset_overview", "")
                enriched.append(merged)

        valid = [m for m in enriched if m.get("poster_url") and m.get("title") != "Unknown"]

        COLS = 5
        for row_start in range(0, len(valid), COLS):
            row = valid[row_start: row_start + COLS]
            cols = st.columns(COLS)
            for col, movie in zip(cols, row):
                with col:
                    st.image(movie["poster_url"], use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    year = movie.get("year", "N/A")
                    score = int(movie["score"] * 100)
                    rating = f"{movie['rating']:.1f}" if movie.get("rating") else "N/A"
                    st.caption(f"Year: {year}  | Rating: {rating}  |  Match: {score}%")
                    overview = movie.get("overview", "")
                    if overview:
                        st.markdown(
                            f"<small>{overview}</small>",
                            unsafe_allow_html = True,
                        )

else:
    st.info("Select a movie above and click **Find Similar Movies** to get started.")