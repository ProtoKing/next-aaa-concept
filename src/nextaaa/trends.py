"""Trends in audience preferences over the years.

Two lenses, because each dataset sees a different thing:

IMDb (1970 to last year): what got made and what got *attention*.
    film_share  share of that year's films in each genre (supply)
    vote_share  share of that year's IMDb votes going to each genre (demand)
    The gap between the two is the story: a genre with 10% of films but 25% of
    votes is over-performing.

MovieLens 20M (1995-2015): what real users watched and how they rated it.
    rating_share  share of that year's ratings going to each genre
    mean_rating   average rating given to each genre that year
    plus a yearly summary: activity, average rating, how old the films people
    rate are ("nostalgia gap"), and how concentrated attention is on the top 1%
    of films.

Multi-genre films: "Action,Adventure,Sci-Fi" is split evenly, 1/3 to each
genre, so shares add up to 100% instead of double-counting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .movielens import MovieLens


def explode_genres(df: pd.DataFrame, col: str = "genres", sep: str = ",") -> pd.DataFrame:
    """One row per (film, genre) with `genre_weight` = 1 / number of genres."""
    out = df.assign(genre=df[col].fillna("Unknown").str.split(sep))
    out["genre_weight"] = 1.0 / out["genre"].str.len()
    out = out.explode("genre")
    out["genre"] = out["genre"].str.strip()
    return out


def fold_genres(genre: pd.Series, keep: list[str]) -> pd.Series:
    """Keep the listed genres, rename everything else to "Other"."""
    return genre.where(genre.isin(keep), "Other")


def top_genres(exploded: pd.DataFrame, weight_col: str, n: int) -> list[str]:
    """The n genres with the most total weight (votes or ratings)."""
    totals = (exploded[weight_col] * exploded["genre_weight"]).groupby(exploded["genre"]).sum()
    return totals.drop(labels=["Unknown"], errors="ignore").nlargest(n).index.tolist()


def imdb_genre_trends(movies: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Per (year, genre): film_share, vote_share and vote-weighted mean rating."""
    df = movies[movies["startYear"].between(cfg.trend_min_year, cfg.max_year)]
    ex = explode_genres(df)
    keep = top_genres(ex, "numVotes", cfg.trend_top_genres)
    ex["genre"] = fold_genres(ex["genre"], keep)

    ex["film_w"] = ex["genre_weight"]
    ex["vote_w"] = ex["genre_weight"] * ex["numVotes"]
    ex["rating_x_vote_w"] = ex["vote_w"] * ex["averageRating"]
    g = ex.groupby(["startYear", "genre"], observed=True)[["film_w", "vote_w", "rating_x_vote_w"]].sum()
    g = g.reset_index()
    year_tot = g.groupby("startYear")[["film_w", "vote_w"]].transform("sum")
    g["film_share"] = 100 * g["film_w"] / year_tot["film_w"]
    g["vote_share"] = 100 * g["vote_w"] / year_tot["vote_w"]
    g["wavg_rating"] = g["rating_x_vote_w"] / g["vote_w"]
    g = g.rename(columns={"startYear": "year"})
    return g[["year", "genre", "film_share", "vote_share", "wavg_rating"]]


def movielens_genre_trends(ml: MovieLens, cfg: AnalysisConfig,
                           min_ratings_per_year: int = 10_000) -> pd.DataFrame:
    """Per (rating year, genre): rating_share and mean_rating.

    Memory-friendly: we first collapse 20M ratings to one row per
    (movie, year), and only then split genres.
    """
    per = (ml.ratings.groupby(["movieId", "rated_year"], observed=True)["rating"]
           .agg(n="size", total="sum").reset_index())
    per = per.merge(ml.movies[["movieId", "genres"]], on="movieId")
    ex = explode_genres(per, sep="|")
    keep = top_genres(ex, "n", cfg.trend_top_genres)
    ex["genre"] = fold_genres(ex["genre"], keep)
    ex["n_w"] = ex["n"] * ex["genre_weight"]
    ex["total_w"] = ex["total"] * ex["genre_weight"]
    g = ex.groupby(["rated_year", "genre"], observed=True)[["n_w", "total_w"]].sum().reset_index()
    g["year_n"] = g.groupby("rated_year")["n_w"].transform("sum")
    g = g[g["year_n"] >= min_ratings_per_year].copy()  # drop sparse early years
    g["rating_share"] = 100 * g["n_w"] / g["year_n"]
    g["mean_rating"] = g["total_w"] / g["n_w"]
    return g.rename(columns={"rated_year": "year"})[["year", "genre", "rating_share", "mean_rating"]]


def movielens_yearly_summary(ml: MovieLens, min_ratings_per_year: int = 10_000) -> pd.DataFrame:
    """Per rating year: activity, mean rating, film age, attention concentration."""
    r = ml.ratings.merge(ml.movies[["movieId", "release_year"]], on="movieId", how="left")
    r["film_age"] = r["rated_year"] - r["release_year"].astype("float")
    yearly = r.groupby("rated_year").agg(
        n_ratings=("rating", "size"),
        n_users=("userId", "nunique"),
        mean_rating=("rating", "mean"),
        median_film_age=("film_age", "median"),
    )

    # Concentration: what share of the year's ratings go to its top 1% of films?
    per_movie = r.groupby(["rated_year", "movieId"]).size().rename("n").reset_index()

    def top1_share(n: pd.Series) -> float:
        k = max(1, int(np.ceil(0.01 * len(n))))
        return 100 * n.nlargest(k).sum() / n.sum()

    yearly["top1pct_share"] = per_movie.groupby("rated_year")["n"].agg(top1_share)
    yearly = yearly[yearly["n_ratings"] >= min_ratings_per_year]
    # MovieLens 20M stops in March 2015: a partial final year would fake a collapse.
    last = pd.to_datetime(ml.ratings["timestamp"].max(), unit="s")
    if last.month < 12:
        yearly = yearly.drop(index=last.year, errors="ignore")
    return yearly.reset_index().rename(columns={"rated_year": "year"})


def rating_spread_by_cluster(ml: MovieLens, cluster_frame: pd.DataFrame,
                             label_col: str, min_ratings: int) -> pd.DataFrame:
    """Per film: the spread (std) of its MovieLens ratings, with its cluster label.

    This tests the hypothesis behind "Popular but Divisive": if those films are
    really divisive, their individual ratings should be more spread out than
    those of Hits. IMDb only gives averages, so only MovieLens can test this.
    Coverage: films released up to 2015 that appear in both datasets.
    """
    stats = ml.ratings.groupby("movieId")["rating"].agg(n="size", rating_std="std").reset_index()
    stats = stats[stats["n"] >= min_ratings].merge(ml.links, on="movieId")
    out = stats.merge(cluster_frame[["tconst", "primaryTitle", label_col]], on="tconst")
    return out[["tconst", "primaryTitle", label_col, "n", "rating_std"]]
