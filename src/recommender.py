#core engine for recommendation 
# We are using hybrid approach for recommendation 
# We are combining Content-Based filtering and Collaborative Filtering

# Three main models are implemented
# TF-IDF (Content Based)
# BERT (Semantic/Deep Learning)
# Hybrid

import difflib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class MovieRecommender:

    def __init__(self, df):
        print("Initializing the Recommendation Engine")
        self.df = df
        
        #Stroing all the titles in a list
        self.titles = df["title"].tolist()

        #building the tfidf matrix with cosine similarity calculated
        self._build_tfidf_engine()
        
        #loading the bert model and encode all movie description into vectors
        self._build_bert_engine()

        print("Both engine initialized and ready!\n")

    def _build_tfidf_engine(self):
        print("Building the TF-IDF model ...")
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words = "english",
                                                ngram_range=(1, 2),
                                                min_df = 2,
                                                max_df = 0.8,
                                                sublinear_tf = True,)
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.df["tfidf_features"]
        )
        
        #Pre computing Cosine similarity for faster retrieval
        self.tfidf_similarity = cosine_similarity(tfidf_matrix)
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        print(f"TF-IDF matrix: {tfidf_matrix.shape} | Vocab: {vocab_size:,} words")


    def _build_bert_engine(self):
        #we will use all-MiniLM-L6-v2 as our embedding model
        print("Loading the BERT model")
        print("Will take some time if running for the first time")

        self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Encoding movie overviews")

        bert_embeddings = self.bert_model.encode(
            self.df["bert_features"].tolist(),
            show_progress_bar= True,
            #process 64 films at a time for faster search
            batch_size = 64,
        )

        self.bert_similarity = cosine_similarity(bert_embeddings)

        print(f"BERT embeddings: {bert_embeddings.shape}")


    #dealing with misspelled words

    def _find_movie_index(self, movie_name):
        #We use difflib.get_close_matches() to get the closest match of misspelled words
        # First try exact case-insensitive match
        lower_input = movie_name.lower().strip()
        exact = [t for t in self.titles if t.lower() == lower_input]
        if exact:
            matches = [exact[0]]
        else:
            # Fall back to fuzzy matching only if no exact match
            matches = difflib.get_close_matches(
                movie_name,
                self.titles,
                n=1,
                cutoff=0.6
            )

        if not matches:
            return None, None
        
        matched_title = matches[0]

        row_index = self.df[self.df["title"] == matched_title].index[0]

        return matched_title, row_index

    def get_recommendations(self, movie_name, method="Hybrid", top_n=10):
        #Give movie name , algorithm choice and the number of top results  
        matched_title, movie_idx = self._find_movie_index(movie_name)

        if movie_idx is None:
            return None
        
        if method == "TF-IDF":
            scores = self.tfidf_similarity[movie_idx]

        elif method == "BERT (Semantic)":
            scores = self.bert_similarity[movie_idx]

        else:
            #Hybrid
            scores = (
                0.4 * self.tfidf_similarity[movie_idx]
                + 0.6 * self.bert_similarity[movie_idx]
            )

        scores_list = list(enumerate(scores))
        sorted_movies = sorted(scores_list, key = lambda x : x[1], reverse = True)

        top_movies = sorted_movies[1: top_n + 1]

        #structuring the dictonaries
        results = []
        for row_idx, score in top_movies:
            movie_row = self.df.iloc[row_idx]
            results.append({
                "title":         movie_row["title"],
                "genres":        movie_row.get("genres", ""),
                "dataset_overview": movie_row.get("overview", ""),
                "tmdbId":        int(movie_row.get("tmdbId", 0)),
                "score":         round(float(score), 3),
                "matched_title": matched_title,
            })

        return results