"""The "balanced" rating: Bayesian shrinkage of averageRating by numVotes.

The problem
-----------
IMDb's averageRating is already *weighted* (it down-weights suspicious voters
to resist ballot-stuffing), but it is not *shrunk*: a film with 12 genuine
votes can still show 9.4. Ranking by raw averageRating rewards obscurity.

The fix (IMDb's own Top 250 formula)
------------------------------------
    shrunk = (v / (v + m)) * R  +  (m / (v + m)) * C

    R = the film's averageRating        v = the film's numVotes
    C = the average rating of the pool  m = "prior strength", in votes

Read it as a weighted average of two opinions: the film's own rating R, and a
sensible default C. The film's own rating gets weight v / (v + m):

    * v = 0      -> shrunk = C       (no evidence: assume an average film)
    * v = m      -> halfway between R and C
    * v >> m     -> shrunk ~= R      (lots of evidence: trust the film's rating)

Why it is principled: this is the posterior mean of a Normal-Normal Bayesian
model, and the same "credibility weighting" actuaries have used since Bühlmann
(1967) to price insurance from limited claims history. m is the ratio of
per-vote noise to the real spread of film quality; we set it from data as a
high quantile of numVotes, the common practical choice, and then show how much
the answer depends on it (`sensitivity_to_m`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_rating(R, v, m: float, C: float):
    """IMDb's Bayesian weighted rating. Works on scalars, arrays or Series."""
    v = np.asarray(v, dtype=float)
    R = np.asarray(R, dtype=float)
    return (v / (v + m)) * R + (m / (v + m)) * C


def shrinkage_params(pool: pd.DataFrame, quantile: float) -> tuple[float, float]:
    """Choose (m, C) from a pool of films.

    m = the given quantile of numVotes (e.g. 0.90 -> a film needs as many votes
        as a top-10% film before its own rating carries half the weight).
    C = the mean averageRating of the pool (IMDb's "mean vote across the report").
    """
    if pool.empty:
        raise ValueError("Cannot compute shrinkage parameters on an empty pool.")
    m = float(np.quantile(pool["numVotes"], quantile))
    C = float(pool["averageRating"].mean())
    return max(m, 1.0), C


def add_shrunk_rating(df: pd.DataFrame, quantile: float, by: str | None = None) -> pd.DataFrame:
    """Return a copy of df with `shrunk_rating`, plus the m and C used.

    by: optional column (e.g. "startYear") to compute m and C *within* each
        group. Comparing a 2023 film with other 2023 films is fairer than
        comparing it with 1994 films that have had 30 years to collect votes.
    """
    out = df.copy()
    if by is None:
        m, C = shrinkage_params(out, quantile)
        out["shrink_m"], out["shrink_C"] = m, C
    else:
        grouped = out.groupby(by, observed=True)
        out["shrink_m"] = grouped["numVotes"].transform(lambda s: max(float(s.quantile(quantile)), 1.0))
        out["shrink_C"] = grouped["averageRating"].transform("mean")
    out["shrunk_rating"] = weighted_rating(out["averageRating"], out["numVotes"],
                                           out["shrink_m"].to_numpy(), out["shrink_C"].to_numpy())
    return out


def top_rated(movies: pd.DataFrame, year: int, quantile: float,
              min_votes: int = 0, top_n: int = 10) -> tuple[pd.DataFrame, dict]:
    """Highest-rated films of one release year, by shrunk rating.

    Returns (leaderboard, info) where info holds m, C and the pool size so the
    deck can state exactly how the ranking was made.
    """
    pool = movies[(movies["startYear"] == year) & (movies["numVotes"] >= min_votes)]
    if pool.empty:
        raise ValueError(f"No films found for {year} with at least {min_votes} votes.")
    m, C = shrinkage_params(pool, quantile)
    ranked = pool.assign(shrunk_rating=weighted_rating(pool["averageRating"], pool["numVotes"], m, C))
    ranked = ranked.sort_values("shrunk_rating", ascending=False).reset_index(drop=True)
    ranked["rank_shrunk"] = np.arange(1, len(ranked) + 1)
    # For contrast: where would each film rank on the raw average alone?
    ranked["rank_raw"] = ranked["averageRating"].rank(ascending=False, method="min").astype(int)
    cols = ["rank_shrunk", "primaryTitle", "averageRating", "numVotes", "shrunk_rating",
            "rank_raw", "genres", "tconst"]
    info = {"year": year, "m": m, "C": C, "quantile": quantile,
            "pool_size": int(len(pool)), "min_votes": min_votes}
    return ranked[cols].head(top_n), info


def sensitivity_to_m(movies: pd.DataFrame, year: int, quantiles: tuple[float, ...],
                     min_votes: int = 0, top_n: int = 10) -> pd.DataFrame:
    """How does the leaderboard change as we vary the prior strength m?

    Small m trusts each film's own rating (favours small, beloved films);
    large m demands lots of votes (favours big, well-liked films). If the #1
    film is stable across a wide range of m, the answer is robust.

    Returns a long table: one row per (quantile, film) for films that reach the
    top_n at any m, with their rank at every m (for a "bump chart").
    """
    pool = movies[(movies["startYear"] == year) & (movies["numVotes"] >= min_votes)]
    frames = []
    for q in quantiles:
        m, C = shrinkage_params(pool, q)
        score = weighted_rating(pool["averageRating"], pool["numVotes"], m, C)
        ranks = pd.Series(score, index=pool.index).rank(ascending=False, method="first")
        frames.append(pd.DataFrame({
            "quantile": q, "m": m, "tconst": pool["tconst"].to_numpy(),
            "primaryTitle": pool["primaryTitle"].to_numpy(),
            "shrunk_rating": score, "rank": ranks.to_numpy().astype(int),
        }))
    long = pd.concat(frames, ignore_index=True)
    ever_top = long.loc[long["rank"] <= top_n, "tconst"].unique()
    return long[long["tconst"].isin(ever_top)].reset_index(drop=True)
