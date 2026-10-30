"""Director and lead-actor track records, as of each film's release.

Why "as of"?
    To predict whether a 2015 film becomes a hit, we may only use what was
    knowable in 2015: the director's and leads' results on films released
    BEFORE 2015. Using their full career would leak the future (a director who
    later made three hits would look great in 2015 too). Every track-record
    feature here is computed from strictly earlier release years.

What a track record contains (per person, per role)
    prior_films        films in scope (>= 1,000 IMDb votes) released earlier
    prior_aud_hits     of those, audience hits (Part A v2 audience stage)
    prior_money_films  of those, films with budget and revenue
    prior_aaa_hits     of those, AAA hits
    aud_rate, aaa_rate smoothed hit rates: (hits + a * base_rate) / (films + a),
                       so 1 hit from 1 film is not "100%" (a = prior strength)
    mean_audience_score  average within-year audience percentile of earlier films

Shortlists for the generator
    The generator may only cast real people from these tables (by short id,
    D01.. for directors, A01.. for actors). Filters: alive (no death year in
    IMDb), active recently, and enough films to have a record. Ranked by
    smoothed AAA rate, then audience rate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .._cache import cached_frame
from ..config import AnalysisConfig, PartBConfig, Paths
from ..io_imdb import CHUNK_ROWS, _read_tsv, _to_int, find_imdb_file

logger = logging.getLogger(__name__)

OUTCOMES = ["aud_hit", "has_finance", "is_aaa_hit", "audience_score"]


# ---------------------------------------------------------------------------
# Loading credits and people
# ---------------------------------------------------------------------------

def load_credits(paths: Paths, cfg: AnalysisConfig, movie_ids: set[str], categories: tuple[str, ...],
                 refresh: bool = False) -> pd.DataFrame:
    """Credits of the given categories (e.g. director, actor, actress) for the given films.

    Returns tconst, nconst, category, billing_rank (1 = first credit within the
    category group: directors and performers are ranked separately).
    """
    path = find_imdb_file(paths.imdb_dir, "principals")
    cats = set(categories)

    def build() -> pd.DataFrame:
        parts = []
        for chunk in _read_tsv(path, ["tconst", "ordering", "nconst", "category"], chunksize=CHUNK_ROWS):
            parts.append(chunk[chunk["category"].isin(cats) & chunk["tconst"].isin(movie_ids)])
        df = pd.concat(parts, ignore_index=True)
        df["ordering"] = _to_int(df["ordering"])
        df["group"] = np.where(df["category"] == "director", "director", "performer")
        df = df.dropna(subset=["ordering"]).sort_values(["tconst", "group", "ordering"])
        df = df.drop_duplicates(["tconst", "nconst", "group"])
        df["billing_rank"] = df.groupby(["tconst", "group"]).cumcount() + 1
        return df[["tconst", "nconst", "category", "billing_rank"]].reset_index(drop=True)

    basics = find_imdb_file(paths.imdb_dir, "basics")
    return cached_frame("credits_partb", build, paths.cache_dir, [path, basics],
                        params={"cats": sorted(cats), "n": len(movie_ids)}, refresh=refresh)


def load_people(paths: Paths, person_ids: set[str], refresh: bool = False) -> pd.DataFrame:
    """nconst, primaryName, birthYear, deathYear for the given people."""
    path = find_imdb_file(paths.imdb_dir, "names")

    def build() -> pd.DataFrame:
        parts = []
        cols = ["nconst", "primaryName", "birthYear", "deathYear"]
        for chunk in _read_tsv(path, cols, chunksize=CHUNK_ROWS):
            parts.append(chunk[chunk["nconst"].isin(person_ids)])
        df = pd.concat(parts, ignore_index=True)
        df["birthYear"], df["deathYear"] = _to_int(df["birthYear"]), _to_int(df["deathYear"])
        return df

    return cached_frame("people_partb", build, paths.cache_dir, [path],
                        params={"n": len(person_ids)}, refresh=refresh)


# ---------------------------------------------------------------------------
# Person x film table and as-of records
# ---------------------------------------------------------------------------

def person_film_table(labels: pd.DataFrame, credits: pd.DataFrame, leads_per_film: int) -> pd.DataFrame:
    """One row per (person, role, film): directors (all credited) and the top-billed leads."""
    directors = credits[credits["category"] == "director"].assign(role="director")
    leads = credits[(credits["category"] != "director")
                    & (credits["billing_rank"] <= leads_per_film)].assign(role="lead")
    pf = pd.concat([directors, leads], ignore_index=True)
    film_cols = ["tconst", "startYear", "genres"] + OUTCOMES
    pf = pf.merge(labels[film_cols], on="tconst", how="inner")
    for c in ("aud_hit", "has_finance", "is_aaa_hit"):
        pf[c] = pf[c].fillna(False).astype(bool)
    pf["aaa_hit_money"] = pf["is_aaa_hit"] & pf["has_finance"]
    return pf.rename(columns={"startYear": "year"})


def _aggregate(pf: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return pf.groupby(keys).agg(
        films=("tconst", "nunique"), aud_hits=("aud_hit", "sum"),
        money_films=("has_finance", "sum"), aaa_hits=("aaa_hit_money", "sum"),
        sum_audience=("audience_score", "sum"),
    ).reset_index()


def _add_rates(df: pd.DataFrame, prefix: str, base_aud: float, base_aaa: float, a: float) -> pd.DataFrame:
    films, money = df[f"{prefix}films"], df[f"{prefix}money_films"]
    df[f"{prefix}aud_rate"] = (df[f"{prefix}aud_hits"] + a * base_aud) / (films + a)
    df[f"{prefix}aaa_rate"] = (df[f"{prefix}aaa_hits"] + a * base_aaa) / (money + a)
    df[f"{prefix}mean_audience_score"] = (df[f"{prefix}sum_audience"] / films.replace(0, np.nan))
    return df.drop(columns=[f"{prefix}sum_audience"])


def base_rates(labels: pd.DataFrame) -> tuple[float, float]:
    fin = labels[labels["has_finance"]]
    return float(labels["aud_hit"].mean()), float(fin["is_aaa_hit"].mean()) if len(fin) else 0.0


def asof_records(pf: pd.DataFrame, labels: pd.DataFrame, a: float) -> pd.DataFrame:
    """Track record of each (role, person) at the start of each year they worked.

    Per (role, nconst, year): cumulative totals over strictly earlier years.
    """
    base_aud, base_aaa = base_rates(labels)
    per_year = _aggregate(pf, ["role", "nconst", "year"]).sort_values(["role", "nconst", "year"])
    totals = ["films", "aud_hits", "money_films", "aaa_hits", "sum_audience"]
    cum = per_year.groupby(["role", "nconst"])[totals].cumsum() - per_year[totals]  # exclude own year
    rec = per_year[["role", "nconst", "year"]].join(cum.add_prefix("prior_"))
    return _add_rates(rec, "prior_", base_aud, base_aaa, a)


def records_as_of(pf: pd.DataFrame, labels: pd.DataFrame, year: int, a: float) -> pd.DataFrame:
    """Track record per (role, person) using films released before `year` (for new concepts)."""
    base_aud, base_aaa = base_rates(labels)
    rec = _aggregate(pf[pf["year"] < year], ["role", "nconst"]).add_prefix("prior_")
    rec = rec.rename(columns={"prior_role": "role", "prior_nconst": "nconst"})
    last = pf[pf["year"] < year].groupby(["role", "nconst"])["year"].max().rename("last_year")
    return _add_rates(rec, "prior_", base_aud, base_aaa, a).merge(last.reset_index(), on=["role", "nconst"])


# ---------------------------------------------------------------------------
# Film-level features
# ---------------------------------------------------------------------------

DIRECTOR_FEATURES = ["dir_prior_films", "dir_prior_aud_hits", "dir_aud_rate", "dir_prior_money_films",
                     "dir_prior_aaa_hits", "dir_aaa_rate", "dir_mean_audience_score"]
LEAD_FEATURES = ["lead_prior_films", "lead_aud_rate", "lead_aaa_rate", "lead_aaa_rate_max",
                 "lead_prior_aaa_hits", "lead_mean_audience_score"]
TRACK_FEATURES = DIRECTOR_FEATURES + LEAD_FEATURES


def _director_cols(rec: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dir_prior_films": rec["prior_films"], "dir_prior_aud_hits": rec["prior_aud_hits"],
        "dir_aud_rate": rec["prior_aud_rate"], "dir_prior_money_films": rec["prior_money_films"],
        "dir_prior_aaa_hits": rec["prior_aaa_hits"], "dir_aaa_rate": rec["prior_aaa_rate"],
        "dir_mean_audience_score": rec["prior_mean_audience_score"],
    })


def _lead_agg(rec: pd.DataFrame, key) -> pd.DataFrame:
    g = rec.groupby(key)
    return pd.DataFrame({
        "lead_prior_films": g["prior_films"].mean(), "lead_aud_rate": g["prior_aud_rate"].mean(),
        "lead_aaa_rate": g["prior_aaa_rate"].mean(), "lead_aaa_rate_max": g["prior_aaa_rate"].max(),
        "lead_prior_aaa_hits": g["prior_aaa_hits"].sum(),
        "lead_mean_audience_score": g["prior_mean_audience_score"].mean(),
    })


def film_track_features(pf: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Per film: its first-credited director's and its leads' as-of track records."""
    rec = pf.merge(records, on=["role", "nconst", "year"], how="left")
    d = rec[(rec["role"] == "director") & (rec["billing_rank"] == 1)].drop_duplicates("tconst")
    dir_feats = _director_cols(d).set_index(d["tconst"].to_numpy())
    lead_feats = _lead_agg(rec[rec["role"] == "lead"], "tconst")
    out = dir_feats.join(lead_feats, how="outer")
    out.index.name = "tconst"
    return out.reindex(columns=TRACK_FEATURES)


def concept_track_features(director: pd.Series, leads: pd.DataFrame) -> dict:
    """The same features for a concept, from shortlist rows (records as of now)."""
    feats = _director_cols(director.to_frame().T).iloc[0].to_dict()
    lead = _lead_agg(leads.assign(_k=0), "_k").iloc[0].to_dict()
    return {**feats, **lead}


# ---------------------------------------------------------------------------
# Shortlists for the generator
# ---------------------------------------------------------------------------

def shortlist(pf: pd.DataFrame, labels: pd.DataFrame, people: pd.DataFrame, role: str,
              now_year: int, bcfg: PartBConfig, top_n: int, id_prefix: str) -> pd.DataFrame:
    """Alive, recently active people with a record, ranked by smoothed AAA rate.

    Returns one row per person with a short id (D01.. / A01..) the LLM must use.
    """
    rec = records_as_of(pf, labels, now_year + 1, bcfg.track_record_prior_strength)
    rec = rec[(rec["role"] == role)
              & (rec["prior_films"] >= bcfg.shortlist_min_prior_films)
              & (rec["last_year"] >= now_year - bcfg.shortlist_active_within_years)]
    rec = rec.merge(people, on="nconst", how="left")
    rec = rec[rec["deathYear"].isna()]
    alive_age = (now_year - rec["birthYear"]).fillna(0)
    rec = rec[alive_age <= 90]
    rec = rec.sort_values(["prior_aaa_rate", "prior_aud_rate", "prior_films"], ascending=False).head(top_n)

    # Main genres, for the prompt ("known for sci-fi, action").
    films = pf[(pf["role"] == role) & pf["nconst"].isin(rec["nconst"])]
    genre_counts = (films.assign(genre=films["genres"].fillna("").str.split(",")).explode("genre")
                    .query("genre != ''").groupby(["nconst", "genre"]).size())
    top_genres = (genre_counts.sort_values(ascending=False).groupby(level=0)
                  .apply(lambda s: ", ".join(s.index.get_level_values(1)[:3])))
    rec["top_genres"] = rec["nconst"].map(top_genres).fillna("")
    rec = rec.reset_index(drop=True)
    rec.insert(0, "id", [f"{id_prefix}{i + 1:02d}" for i in range(len(rec))])
    rec["primaryName"] = rec["primaryName"].fillna(rec["nconst"])
    return rec
