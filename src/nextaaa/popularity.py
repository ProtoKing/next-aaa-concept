"""Who was the most popular actor of 2023? Two definitions, side by side.

"Popular" is ambiguous, and showing two reasonable definitions next to each
other is more honest (and more interesting to stakeholders) than silently
picking one.

Both sum over every 2023 film the actor appears in, with a billing weight so a
lead role counts more than a cameo:

    billing_weight = 1 / billing_rank ** power      (power = 0.5 by default)
                     -> lead 1.00, 2nd 0.71, 3rd 0.58, 4th 0.50

Definition A: Reach
    reach = sum( billing_weight * numVotes )
    "How many people engaged with this actor's 2023 work?"

Definition B: Net goodwill (reach x quality)
    goodwill = sum( billing_weight * numVotes * (shrunk_rating - C) )
    "How many people engaged, and did they like it?"
    C is the average 2023 rating, so a film rated above average adds goodwill
    and a film rated below average subtracts it.

Why not simply numVotes * rating? Ratings only vary from about 5 to 8.5 while
votes vary 1,000-fold, so multiplying by the raw rating barely changes the
ranking: it is almost Definition A again. Using the rating *above average*
lets quality genuinely move the result.
"""

from __future__ import annotations

import pandas as pd

from .ratings import shrinkage_params, weighted_rating


def actor_popularity(movies: pd.DataFrame, credits: pd.DataFrame, names: pd.DataFrame,
                     year: int, quantile: float, billing_power: float = 0.5,
                     min_votes: int = 0) -> pd.DataFrame:
    """One row per actor with both popularity scores and their ranks.

    Args:
        movies: the movie table (needs tconst, primaryTitle, startYear,
            averageRating, numVotes).
        credits: actor credits (tconst, nconst, billing_rank).
        names: nconst -> primaryName.
        year: the release year to analyse (2023).
        quantile: shrinkage quantile for m (same as the rest of the analysis).
        billing_power: exponent in 1 / rank ** power.
        min_votes: ignore films with fewer votes (same pool as the 2023 leaderboard,
            so C and m match the "highest rated film" slide).
    """
    films = movies[(movies["startYear"] == year) & (movies["numVotes"] >= min_votes)].copy()
    if films.empty:
        raise ValueError(f"No rated films for {year}.")
    m, C = shrinkage_params(films, quantile)
    films["shrunk_rating"] = weighted_rating(films["averageRating"], films["numVotes"], m, C)

    df = credits.merge(films[["tconst", "primaryTitle", "numVotes", "shrunk_rating"]], on="tconst")
    df["billing_weight"] = 1.0 / df["billing_rank"].astype(float) ** billing_power
    df["reach_part"] = df["billing_weight"] * df["numVotes"]
    df["goodwill_part"] = df["reach_part"] * (df["shrunk_rating"] - C)

    # The actor's single most-voted 2023 film, to make the table readable.
    top_film = (df.sort_values("numVotes", ascending=False)
                  .drop_duplicates("nconst")[["nconst", "primaryTitle"]]
                  .rename(columns={"primaryTitle": "top_film"}))

    agg = df.groupby("nconst").agg(
        films_2023=("tconst", "nunique"),
        reach=("reach_part", "sum"),
        net_goodwill=("goodwill_part", "sum"),
    ).reset_index()
    agg = agg.merge(top_film, on="nconst").merge(names, on="nconst", how="left")
    agg["primaryName"] = agg["primaryName"].fillna(agg["nconst"])
    agg["rank_reach"] = agg["reach"].rank(ascending=False, method="min").astype(int)
    agg["rank_goodwill"] = agg["net_goodwill"].rank(ascending=False, method="min").astype(int)
    agg.attrs["C"], agg.attrs["m"] = C, m
    cols = ["primaryName", "films_2023", "top_film", "reach", "rank_reach",
            "net_goodwill", "rank_goodwill", "nconst"]
    return agg[cols].sort_values("rank_reach").reset_index(drop=True)


def side_by_side(popularity: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top-N under each definition, next to each other, for the slide."""
    a = popularity.nsmallest(top_n, "rank_reach")[["primaryName", "reach", "top_film"]]
    b = popularity.nsmallest(top_n, "rank_goodwill")[["primaryName", "net_goodwill", "top_film"]]
    a.columns = ["Reach: actor", "Reach: weighted votes", "Reach: top film"]
    b.columns = ["Goodwill: actor", "Goodwill: score", "Goodwill: top film"]
    out = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)
    out.index = range(1, len(out) + 1)
    out.index.name = "rank"
    return out


# ---------------------------------------------------------------------------
# v2: a third definition now that TMDB gives us revenue
# ---------------------------------------------------------------------------

def actor_box_office(films: pd.DataFrame, credits: pd.DataFrame, names: pd.DataFrame,
                     year: int, billing_power: float = 0.5) -> pd.DataFrame:
    """Definition C: Box office = sum( billing_weight * worldwide revenue ).

    "Whose films sold the most tickets?" Only films with known revenue count,
    so actors in films TMDB has no revenue for are under-counted; the slide
    should say so. Revenue is nominal dollars of that year (all 2023 here).

    films: needs tconst, primaryTitle, startYear, revenue (NaN where unknown).
    """
    f = films[(films["startYear"] == year) & films["revenue"].notna()]
    df = credits.merge(f[["tconst", "primaryTitle", "revenue"]], on="tconst")
    df["box_office_part"] = df["revenue"] / df["billing_rank"].astype(float) ** billing_power
    top_film = (df.sort_values("revenue", ascending=False).drop_duplicates("nconst")
                  [["nconst", "primaryTitle"]].rename(columns={"primaryTitle": "top_film_by_revenue"}))
    agg = (df.groupby("nconst")
             .agg(films_with_revenue=("tconst", "nunique"), box_office=("box_office_part", "sum"))
             .reset_index().merge(top_film, on="nconst").merge(names, on="nconst", how="left"))
    agg["primaryName"] = agg["primaryName"].fillna(agg["nconst"])
    agg["rank_box_office"] = agg["box_office"].rank(ascending=False, method="min").astype(int)
    return agg.sort_values("rank_box_office").reset_index(drop=True)


def three_way(popularity: pd.DataFrame, box_office: pd.DataFrame | None, top_n: int = 10) -> pd.DataFrame:
    """Top-N actors under Reach, Net goodwill and (if available) Box office."""
    out = side_by_side(popularity, top_n)
    if box_office is not None and len(box_office):
        c = box_office.nsmallest(top_n, "rank_box_office")[["primaryName", "box_office", "top_film_by_revenue"]]
        c.columns = ["Box office: actor", "Box office: weighted US$", "Box office: top film"]
        c.index = range(1, len(c) + 1)
        out = out.join(c)
    return out
