#getting data ready for recommendation Engine

import os
import pandas as pd

#checking the existence of the files
def _check_for_files(*paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n\n Missing File: '{path}'\n"
                f"Please provide the csv files. Use readme.md for instructions"
            )

#Loading the data from the csv files        
def load_data(
        movies_path = "movies.csv",
        ratings_path = "ratings.csv",
        links_path = "links.csv",
):
    _check_for_files(movies_path, links_path, ratings_path)

    #loading the csv 
    df_movies = pd.read_csv(movies_path)
    df_ratings = pd.read_csv(ratings_path)
    df_links = pd.read_csv(links_path)

    print(f"Movies loaded: {len(df_movies):,} rows")
    print(f"Ratings loaded: {len(df_ratings):,} rows")
    print(f"Links loaded: {len(df_links):,} rows")

    df_movies = df_movies.rename(columns = {"id": "tmdbId"})

    #cleaning links.csv so that no sparsity is less
    df_links = df_links.dropna(subset=["tmdbId"])
    df_links["tmdbId"] = df_links["tmdbId"].astype(int)

    #Creating the mapping from movies.csv to ratings.csv using linking.csv
    df_ratings_mapped = pd.merge(
        df_ratings,
        df_links[["movieId", "tmdbId"]],
        on = "movieId",
        how = "inner",
    )

    print(f"Ratings with TMDB IDs: {len(df_ratings_mapped):,} rows")

    #handling missing values

    content_cols = ["generes", "keywords", "tagline", "cast", "director", "overview"]
    for col in content_cols:
        if col in df_movies.columns:
            df_movies[col] = df_movies[col].fillna("")


    #Creating whole string for TF-IDF vectorizer
    df_movies["tfidf_features"] = (
        df_movies["genres"] + " " 
        + df_movies["keywords"] + " " + df_movies["keywords"] + " "
        + df_movies["overview"] + " " + df_movies["overview"] + " "
        + df_movies["cast"] + " "
        + df_movies["director"]+ " "
        + df_movies["tagline"] 
    ).fillna("").str.lower()

    #creating natural language string for BERT
    df_movies["bert_features"] = (
        df_movies["overview"] + " " + df_movies["genres"]
    ).fillna("")

    #dropping empty titles
    df_movies = df_movies.dropna(subset=["title"]).reset_index(drop = True)

    print(f"Final clean movies: {len(df_movies):,}")
    print("Data loading complete.\n")

    return df_movies, df_ratings_mapped