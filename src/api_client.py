#handles movie poster and urls for additional details
#Uses REST API for communicating with TMDB

#Change this file if TMDB api or database of movies changes

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Try Streamlit secrets first (for Streamlit Cloud), then .env (for local dev)
try:
    _TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except (FileNotFoundError, KeyError):
    _TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

_TMDB_BASE_URL = "https://api.themoviedb.org/3"
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

_PLACEHOLDER_URL = "https://via.placeholder.com/500x750/1a1a2e/ffffff?text=No+Poster"


def get_movie_details(tmdb_id):
    #Get movie details from TMDB API by TMDB ID
    
    if not _TMDB_API_KEY or _TMDB_API_KEY == "paste_your_key_here":
        return _fallback_details("No API key provided")

    if not tmdb_id or tmdb_id <= 0:
        return _fallback_details()

    try:
        url = f"{_TMDB_BASE_URL}/movie/{tmdb_id}"
        params = {
            "api_key": _TMDB_API_KEY,
            "language": "en-US",
        }

        response = requests.get(url , params = params, timeout = 5, verify = False)

        response.raise_for_status()

        data = response.json()

        poster_path = data.get("poster_path")
        poster_url = (
            f"{_TMDB_IMAGE_BASE}{poster_path}"
            if poster_path
            else _PLACEHOLDER_URL
        )

        release_date = data.get("release_date", "")
        year = release_date[:4] if release_date else "N/A"

        return {
            "poster_url" : poster_url,
            "title": data.get("title", "Unknown"),
            "rating": data.get("vote_average", 0.0),
            "year": year,
            "overview": data.get("overview", ""),
        }
    
    except requests.exceptions.Timeout:
        return _fallback_details("Timeout")
    
    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            return _fallback_details(f"HTTP {e.response.status_code}")
        else:
            return "No Response"
    
    except requests.exceptions.RequestException:
        return _fallback_details("Network error")
    
    except Exception:
        return _fallback_details("Unknown Error")
    

def _fallback_details(reason = ""):
    return {
        "poster_url" : _PLACEHOLDER_URL,
        "title" : "Unknown",
        "rating": 0.0,
        "year": "N/A",
        "overview" : "",
    }