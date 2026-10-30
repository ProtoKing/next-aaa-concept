"""v2: from films to verdicts, by-year hit lists and the all-time list.

The two-stage gate
------------------
    Audience hit    = at least 3 of 5 methods say "Hit" on quality x reach
    Commercial hit  = at least 3 of 5 methods say "Blockbuster" on ROI x revenue
                      AND the film actually cleared break-even (ROI >= 2.5x).
                      (The absolute gate matters: in a weak year, "better than
                      the year's median" can still mean "lost money".)

    AAA hit         = audience hit AND commercial hit

For films with budget and revenue, every film lands in one cell of a 2x2:

                         commercial hit            not a commercial hit
    audience hit       | AAA hit                 | Loved, unprofitable     |
    not audience hit   | Profitable, not loved   | Neither                 |

Films without money data still get the audience verdict
("Audience hit (no $ data)"), so nothing is silently dropped.

Ranking within the hits: the hit score
---------------------------------------
    hit_score = mean of the film's four within-year percentiles
                (quality, reach, ROI, revenue), 0-100

"Avatar was in the 99th percentile of its year on reach and revenue..." is
something a studio head can repeat. Because every percentile is relative to the
film's own year, the all-time list compares eras fairly: it ranks films by how
completely they dominated their own year.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import finance
from .config import AnalysisConfig
from .normalize import year_relative
from .ratings import add_shrunk_rating

logger = logging.getLogger(__name__)

VERDICTS = ["AAA hit", "Loved, unprofitable", "Profitable, not loved", "Neither"]
NO_MONEY_HIT = "Audience hit (no $ data)"
NO_MONEY = "No $ data"


# ---------------------------------------------------------------------------
# 1. The feature table
# ---------------------------------------------------------------------------

def build_v2_frame(movies: pd.DataFrame, tmdb: pd.DataFrame | None, cfg: AnalysisConfig,
                   cpi: pd.Series | None = None) -> pd.DataFrame:
    """Films in scope, joined to TMDB money, with year-relative z and percentiles.

    Scope: feature films with >= min_votes_for_clustering IMDb votes, released
    min_year..max_year (same as v1). Money columns are NaN where TMDB lacks them.
    """
    df = movies[(movies["numVotes"] >= cfg.min_votes_for_clustering)
                & movies["startYear"].between(cfg.min_year, cfg.max_year)].copy()

    money_cols = ["tconst", "budget", "revenue"]
    extra = ["tmdb_title", "tmdb_release_year", "tmdb_popularity", "original_language",
             "production_companies", "overview", "tagline", "keywords"]
    if tmdb is not None:
        keep = money_cols + [c for c in extra if c in tmdb.columns]
        df = df.merge(tmdb[keep], on="tconst", how="left")
    else:
        df["budget"] = np.nan
        df["revenue"] = np.nan

    df = finance.add_finance(df, cfg, cpi)
    df["log_votes"] = np.log10(df["numVotes"].astype(float))

    # Audience axes: every film in scope.
    for name, col in (("quality", "averageRating"), ("reach", "log_votes")):
        rel = year_relative(df[col], df["startYear"], cfg.min_ref_films_audience, cfg.max_year_window)
        df[f"z_{name}"], df[f"pct_{name}"] = rel["z"], rel["pct"]
    dropped = df["z_quality"].isna() | df["z_reach"].isna()
    if dropped.any():
        logger.info("Dropping %d films from years too thin even with a +/-%d-year window.",
                    int(dropped.sum()), cfg.max_year_window)
    df = df[~dropped].copy()

    # Money axes: only films with budget and revenue. Real dollars when available,
    # so a +/- few-year reference window does not mix in inflation.
    fin = df["has_finance"]
    revenue_for_rank = df["revenue_real"].fillna(df["revenue"])
    for name, values in (("roi", df["log_roi"]), ("revenue", np.log10(revenue_for_rank))):
        rel = year_relative(values.where(fin), df["startYear"], cfg.min_ref_films_finance,
                            cfg.max_year_window)
        df[f"z_{name}"], df[f"pct_{name}"] = rel["z"], rel["pct"]
    thin = fin & (df["z_roi"].isna() | df["z_revenue"].isna())
    if thin.any():
        logger.info("%d films with money data are in years too thin for the money stage.", int(thin.sum()))
    df["has_finance"] = fin & ~thin

    df["audience_score"] = df[["pct_quality", "pct_reach"]].mean(axis=1)
    df["commercial_score"] = df[["pct_roi", "pct_revenue"]].mean(axis=1).where(df["has_finance"])
    df["hit_score"] = df[["pct_quality", "pct_reach", "pct_roi", "pct_revenue"]].mean(axis=1) \
        .where(df["has_finance"])
    df = add_shrunk_rating(df, cfg.shrinkage_quantile, by="startYear")
    logger.info("v2 frame: %s films (%s with budget + revenue)", f"{len(df):,}",
                f"{int(df['has_finance'].sum()):,}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Combining the two stages
# ---------------------------------------------------------------------------

def combine_stages(frame: pd.DataFrame, audience: pd.DataFrame,
                   commercial: pd.DataFrame | None, cfg: AnalysisConfig) -> pd.DataFrame:
    """Attach both stages' columns to the frame and assign the final verdict."""
    out = frame.copy()
    aud_cols = [c for c in audience.columns if c.startswith("aud_")]
    out = out.join(audience[aud_cols])

    if commercial is not None and len(commercial):
        com_cols = [c for c in commercial.columns if c.startswith("com_")]
        out = out.join(commercial[com_cols])
        # Absolute gate on top of the relative vote.
        out["com_hit_relative"] = out["com_hit"].astype("boolean")
        out["com_hit"] = out["com_hit_relative"] & (out["roi_multiple"] >= cfg.breakeven_multiple)
        out["com_mega"] = out["com_mega"].astype("boolean") & out["com_hit"]
    else:
        out["com_hit"] = pd.array([pd.NA] * len(out), dtype="boolean")
        out["com_mega"] = pd.array([pd.NA] * len(out), dtype="boolean")

    fin = out["has_finance"]
    aud = out["aud_hit"].astype(bool)
    com = out["com_hit"].fillna(False).astype(bool)
    out["verdict"] = np.select(
        [fin & aud & com, fin & aud & ~com, fin & ~aud & com, fin, aud],
        [VERDICTS[0], VERDICTS[1], VERDICTS[2], VERDICTS[3], NO_MONEY_HIT],
        default=NO_MONEY,
    )
    out["is_aaa_hit"] = out["verdict"] == "AAA hit"
    out["aaa_tier"] = np.select(
        [out["is_aaa_hit"] & out["aud_mega"].astype(bool) & out["com_mega"].fillna(False).astype(bool),
         out["is_aaa_hit"]],
        ["AAA mega-hit", "AAA hit"], default="")
    return out


def verdict_grid(df: pd.DataFrame) -> pd.DataFrame:
    """The 2 x 2 counts (films with money data only)."""
    fin = df[df["has_finance"]]
    grid = pd.crosstab(fin["aud_hit"].map({True: "Audience hit", False: "Not an audience hit"}),
                       fin["com_hit"].fillna(False).map({True: "Commercial hit", False: "Not a commercial hit"}))
    return grid.reindex(index=["Audience hit", "Not an audience hit"],
                        columns=["Commercial hit", "Not a commercial hit"], fill_value=0)


# ---------------------------------------------------------------------------
# 3. By year and all time
# ---------------------------------------------------------------------------

DISPLAY_COLS = ["startYear", "primaryTitle", "genres", "hit_score", "pct_quality", "pct_reach",
                "pct_roi", "pct_revenue", "averageRating", "numVotes", "budget", "revenue",
                "roi_multiple", "budget_real", "revenue_real", "aaa_tier", "tconst"]


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per year: how many films, how many hits, and the hit of the year."""
    g = df.groupby("startYear")
    fin = df[df["has_finance"]].groupby("startYear")
    out = pd.DataFrame({
        "films": g.size(),
        "audience_hits": g["aud_hit"].sum(),
        "films_with_money": fin.size(),
        "profitable_pct": 100 * fin["profitable"].mean(),
        "aaa_hits": fin["is_aaa_hit"].sum(),
    }).fillna({"films_with_money": 0, "aaa_hits": 0})
    out["aaa_hit_pct_of_money_films"] = 100 * out["aaa_hits"] / out["films_with_money"].replace(0, np.nan)
    best = (df[df["is_aaa_hit"]].sort_values("hit_score", ascending=False)
            .drop_duplicates("startYear").set_index("startYear"))
    out["hit_of_the_year"] = best["primaryTitle"]
    out["hit_of_the_year_score"] = best["hit_score"]
    # Years with no AAA hit (or no money data): fall back to the top audience hit.
    fallback = (df[df["aud_hit"]].sort_values("audience_score", ascending=False)
                .drop_duplicates("startYear").set_index("startYear")["primaryTitle"])
    out["top_audience_hit"] = fallback
    return out.reset_index().rename(columns={"startYear": "year"})


def hits_by_year(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """The top-N AAA hits of every year, ranked by hit score."""
    hits = df[df["is_aaa_hit"]].sort_values(["startYear", "hit_score"], ascending=[True, False])
    hits = hits.groupby("startYear").head(top_n).copy()
    hits["rank_in_year"] = hits.groupby("startYear").cumcount() + 1
    cols = ["rank_in_year"] + [c for c in DISPLAY_COLS if c in hits.columns]
    return hits[cols].reset_index(drop=True)


def all_time_hits(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """The all-time AAA hits: films that dominated their own year the most."""
    hits = df[df["is_aaa_hit"]].nlargest(top_n, "hit_score").copy()
    hits.insert(0, "rank_all_time", np.arange(1, len(hits) + 1))
    return hits[["rank_all_time"] + [c for c in DISPLAY_COLS if c in hits.columns]].reset_index(drop=True)


def all_time_by_real_revenue(df: pd.DataFrame, top_n: int) -> pd.DataFrame | None:
    """Biggest box office in base-year dollars (needs CPI). A familiar cross-check."""
    if df["revenue_real"].notna().sum() == 0:
        return None
    top = df[df["has_finance"]].nlargest(top_n, "revenue_real")
    return top[[c for c in DISPLAY_COLS + ["verdict"] if c in top.columns]].reset_index(drop=True)


def audience_all_time(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """All-time audience hits (all films, including those without money data)."""
    hits = df[df["aud_hit"]].nlargest(top_n, "audience_score")
    cols = ["startYear", "primaryTitle", "genres", "audience_score", "pct_quality", "pct_reach",
            "averageRating", "numVotes", "verdict"]
    return hits[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Genre economics (what to make next)
# ---------------------------------------------------------------------------

def genre_economics(df: pd.DataFrame, since_year: int, min_films: int = 30) -> pd.DataFrame:
    """Per genre, for films released since `since_year` with money data.

    A film tagged "Action,Sci-Fi" counts once in each genre (these are rates,
    so no splitting is needed). Genres with fewer than min_films are dropped.
    """
    fin = df[df["has_finance"] & (df["startYear"] >= since_year)].copy()
    fin["genre"] = fin["genres"].fillna("Unknown").str.split(",")
    ex = fin.explode("genre")
    g = ex.groupby("genre")
    out = pd.DataFrame({
        "films": g.size(),
        "median_budget": g["budget_for_tier"].median(),
        "median_roi": g["roi_multiple"].median(),
        "profitable_pct": 100 * g["profitable"].mean(),
        "audience_hit_pct": 100 * g["aud_hit"].mean(),
        "aaa_hit_pct": 100 * g["is_aaa_hit"].mean(),
    })
    out = out[out["films"] >= min_films].drop(index="Unknown", errors="ignore")
    return out.sort_values("aaa_hit_pct", ascending=False)
