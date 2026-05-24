# Movie Recommendation System

A content-based movie recommendation system that uses three different NLP algorithms to find similar movies. Built with Python and Streamlit, deployable to Streamlit Cloud.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Frontend | Streamlit |
| ML — Classic NLP | scikit-learn (TF-IDF) |
| ML — Deep Learning | sentence-transformers (SBERT) |
| Movie Metadata | TMDB API |
| Data | TMDB 5000 Movies Dataset + MovieLens |
| Secrets Management | python-dotenv |

---

## How It Works

Three algorithms are available:

**TF-IDF** — Converts each movie's genres, cast, keywords, director, tagline and plot into a bag-of-words vector. Finds movies that share rare, meaningful terms. Fast and interpretable.

**BERT (Semantic)** — Uses the `all-MiniLM-L6-v2` sentence transformer model to encode plot overviews into 384-dimensional embeddings. Understands meaning rather than just word overlap — knows that "detective" and "cop" are related concepts.

**Hybrid** — A weighted blend of TF-IDF (40%) and BERT (60%) scores. Combines genre and keyword precision with semantic plot understanding.

Similarity is measured using cosine similarity. All pairwise similarities are precomputed at startup so lookups are instant.

---

## Project Structure

```
.
├── streamlit_app.py          # Entry point — run this
├── requirements.txt          # Python dependencies
├── .env                      # Your API key (never commit this)
├── .env.example              # Template for .env
├── .gitignore
├── README.md
│
└── src/
    ├── __init__.py
    ├── data_loader.py        # Loads and cleans CSV datasets
    ├── recommender.py        # TF-IDF, BERT, and Hybrid engines
    └── api_client.py         # TMDB API calls for posters and metadata
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the datasets

**TMDB 5000 Movies:**
Download `tmdb_5000_movies.csv` from Kaggle and rename it to `movies.csv`. Place it in the project root.You will get a public URL you can share or link on your resume.
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

**MovieLens (for ratings):**
Download the small dataset from GroupLens and place `ratings.csv` and `links.csv` in the project root.
https://grouplens.org/datasets/movielens/latest/

### 5. Get a TMDB API key

1. Create a free account at https://www.themoviedb.org
2. Go to Settings > API
3. Copy the API Key 

### 6. Configure the API key

Copy `.env.example` to `.env` and paste your key:

```
TMDB_API_KEY=your_key_here
```

No quotes. No spaces around the equals sign.

### 7. Run the app

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## Deploying to Streamlit Cloud

1. Push the project to a public GitHub repository. Make sure `.env` is in `.gitignore`.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click New app, select your repository, and set the main file to `streamlit_app.py`.
4. Go to Settings > Secrets and add:

```toml
TMDB_API_KEY = "your_key_here"
```

5. Deploy.

Note: The BERT model (~80MB) downloads automatically on first run. Streamlit Cloud handles this but the first cold start takes a few minutes.

---

## Usage

1. Type a movie name in the text box or browse the dropdown list.
2. Select an algorithm — Hybrid is recommended as the default.
3. Adjust the results slider if needed.
4. Click Find Similar Movies.
5. Each result shows the poster, year, TMDB rating, match percentage, and plot summary.

---

## Dataset Limitations

The TMDB 5000 dataset covers mainstream films up to around 2017. Some movies will not be found — particularly recent releases, independent films, and non-English titles with limited metadata. If a search returns no match, try using the dropdown list to find the exact title.