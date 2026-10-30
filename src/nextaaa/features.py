"""Build the two axes we cluster on: quality and year-adjusted reach.

Axis 1, quality = averageRating (raw, not shrunk)
    We deliberately do NOT cluster on the shrunk rating. Shrinkage pulls
    low-vote films toward the average by design, which would erase the very
    "Hidden Gem" quadrant (high rating, low votes) we want to find. Instead we
    make the raw rating trustworthy by only clustering films with at least
    `min_votes_for_clustering` votes. The shrunk rating is still computed here,
    but it is used for *ranking*, not for clustering.

Axis 2, reach = numVotes, adjusted for release year
    Raw vote counts confuse popularity with age and with IMDb's growth: a 2005
    film has had 20 years to collect votes, and IMDb had fewer users in 1995.
    So we ask: "how many more votes did this film get than a typical film
    released the same year?"

        log_votes = log10(numVotes)          # 3 = 1k votes, 6 = 1M votes
        reach_z   = (log_votes - median of its year) / (robust spread of its year)

    Median and IQR (rather than mean and std) keep a few mega-blockbusters from
    distorting "typical". The IQR is divided by 1.349 so that, for normally
    distributed data, reach_z is on the same scale as an ordinary z-score.
    reach_z = 0 means "a typical film of its year"; reach_z = 2 means "far more
    votes than typical". We keep the long right tail on purpose: true hits ARE
    outliers, and DBSCAN needs that tail to spot them.

Both axes are finally standardised (mean 0, std 1) so KMeans and DBSCAN, which
measure plain Euclidean distance, weigh them equally.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import AnalysisConfig
from .ratings import add_shrunk_rating

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["x_quality", "x_reach"]  # the standardised clustering inputs

IQR_TO_SIGMA = 1.349  # IQR of a standard normal distribution


def _robust_z_within(values: pd.Series, groups: pd.Series) -> pd.Series:
    """(x - group median) / (group IQR / 1.349), computed separately per group."""
    grouped = values.groupby(groups, observed=True)
    median = grouped.transform("median")
    q75 = grouped.transform(lambda s: s.quantile(0.75))
    q25 = grouped.transform(lambda s: s.quantile(0.25))
    scale = (q75 - q25) / IQR_TO_SIGMA
    # A group where every film has identical votes has IQR 0; avoid dividing by 0.
    scale = scale.where(scale > 0, values.std())
    return (values - median) / scale


def build_cluster_frame(movies: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Filter to the clustering universe and add the feature columns.

    Added columns:
        log_votes      log10(numVotes)
        reach_z        year-adjusted reach (see module docstring)
        reach_pct      percentile of votes within the release year (0-100), for display
        shrunk_rating  Bayesian rating, shrunk within each release year (for ranking)
        x_quality      standardised averageRating   } the two inputs
        x_reach        standardised reach_z         } to KMeans / DBSCAN
    """
    df = movies[
        (movies["numVotes"] >= cfg.min_votes_for_clustering)
        & movies["startYear"].between(cfg.min_year, cfg.max_year)
    ].copy()

    # Drop years with too few films to define "typical" reliably.
    films_per_year = df.groupby("startYear")["tconst"].transform("size")
    thin_years = sorted(df.loc[films_per_year < cfg.min_films_per_year, "startYear"].unique())
    if thin_years:
        logger.info("Dropping %d thin years (< %d films): %s",
                    len(thin_years), cfg.min_films_per_year, thin_years[:10])
    df = df[films_per_year >= cfg.min_films_per_year]
    if df.empty:
        raise ValueError("No films left after filtering; check min_votes / year settings.")

    df["log_votes"] = np.log10(df["numVotes"].astype(float))
    df["reach_z"] = _robust_z_within(df["log_votes"], df["startYear"])
    df["reach_pct"] = df.groupby("startYear")["numVotes"].rank(pct=True) * 100

    df = add_shrunk_rating(df, cfg.shrinkage_quantile, by="startYear")

    scaled = StandardScaler().fit_transform(df[["averageRating", "reach_z"]])
    df["x_quality"], df["x_reach"] = scaled[:, 0], scaled[:, 1]

    logger.info("Clustering universe: %s films, %d-%d", f"{len(df):,}",
                df["startYear"].min(), df["startYear"].max())
    return df.reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The (n_films x 2) array that KMeans and DBSCAN consume."""
    return frame[FEATURE_COLUMNS].to_numpy(dtype=float)
