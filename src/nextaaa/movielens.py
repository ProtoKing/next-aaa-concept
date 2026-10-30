"""Load MovieLens 20M (optional input for Part A).

Download ml-20m.zip from https://grouplens.org/datasets/movielens/20m/ (or the
Kaggle mirror) and unzip it so the folder contains:

    data/ml-20m/ratings.csv   20M rows: userId, movieId, rating (0.5-5), timestamp
    data/ml-20m/movies.csv    movieId, title "Toy Story (1995)", genres "A|B|C"
    data/ml-20m/links.csv     movieId -> imdbId (lets us join to IMDb)

In Part A, MovieLens powers two things:
  * the "trends in user preferences over the years" analysis (1995-2015), and
  * the check that the "Popular but Divisive" cluster really is divisive
    (IMDb only publishes average ratings; MovieLens has every individual rating).

Everything in Part A still runs without MovieLens; those two sections are
skipped with a clear message. `load_movielens` returns None in that case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ._cache import cached_frame
from .config import MOVIELENS_DOWNLOAD_URL, Paths

logger = logging.getLogger(__name__)

REQUIRED_FILES = ("ratings.csv", "movies.csv", "links.csv")


@dataclass
class MovieLens:
    """The three MovieLens tables we use."""

    ratings: pd.DataFrame  # userId, movieId, rating, timestamp, rated_year
    movies: pd.DataFrame   # movieId, title, genres, release_year
    links: pd.DataFrame    # movieId, tconst (IMDb id, e.g. "tt0114709")


def movielens_available(paths: Paths) -> bool:
    return all((paths.movielens_dir / f).exists() for f in REQUIRED_FILES)


def imdb_id_to_tconst(imdb_id: pd.Series) -> pd.Series:
    """MovieLens stores IMDb ids as bare numbers (114709); IMDb uses "tt0114709"."""
    numeric = pd.to_numeric(imdb_id, errors="coerce").astype("Int64")
    return ("tt" + numeric.astype("string").str.zfill(7)).where(numeric.notna())


def load_movielens(paths: Paths, refresh: bool = False) -> MovieLens | None:
    """Load ratings, movies and links, or return None if the files are missing."""
    if not movielens_available(paths):
        logger.warning(
            "MovieLens 20M not found in %s (need %s). Skipping MovieLens sections. "
            "Download: %s", paths.movielens_dir, ", ".join(REQUIRED_FILES), MOVIELENS_DOWNLOAD_URL)
        return None

    d = paths.movielens_dir

    def build_ratings() -> pd.DataFrame:
        df = pd.read_csv(
            d / "ratings.csv",
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
        )
        df["rated_year"] = pd.to_datetime(df["timestamp"], unit="s").dt.year.astype("int16")
        return df

    ratings = cached_frame("ml_ratings", build_ratings, paths.cache_dir,
                           [d / "ratings.csv"], refresh=refresh)

    movies = pd.read_csv(d / "movies.csv", dtype={"movieId": "int32", "title": str, "genres": str})
    # The release year sits in brackets at the end of the title: "Heat (1995)".
    movies["release_year"] = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False), errors="coerce"
    ).astype("Int64")
    movies["genres"] = movies["genres"].replace("(no genres listed)", "Unknown")

    links = pd.read_csv(d / "links.csv", dtype={"movieId": "int32", "imdbId": str, "tmdbId": str})
    links["tconst"] = imdb_id_to_tconst(links["imdbId"])
    links = links[["movieId", "tconst"]].dropna()

    logger.info("MovieLens: %s ratings, %s movies", f"{len(ratings):,}", f"{len(movies):,}")
    return MovieLens(ratings=ratings, movies=movies, links=links)
