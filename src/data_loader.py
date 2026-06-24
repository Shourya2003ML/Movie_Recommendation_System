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
def load_data(movies_path = "movies.csv"):
    _check_for_files(movies_path)

    #loading the csv 
    df_movies = pd.read_csv(movies_path)

    print(f"Movies loaded: {len(df_movies):,} rows")

    # 'id' column in movies.csv is the TMDB ID
    df_movies = df_movies.rename(columns = {"id": "tmdbId"})

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

    return df_movies