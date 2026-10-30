"""Load the TMDB movies dataset (budget, revenue, plot text) and join it to IMDb.

Download "TMDB Movies Dataset 2023 (930K+ movies)" by asaniczka from Kaggle:
    https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
and put EITHER the downloaded zip OR the extracted CSV in data/tmdb/:

    data/tmdb/archive.zip                     (the Kaggle download, untouched), or
    data/tmdb/TMDB_movie_dataset_v11.csv      (any *.csv name works)

The join key is `imdb_id` ("tt0114709"), which matches IMDb's `tconst`.

What we use from TMDB
    budget, revenue   production budget and worldwide box office, in nominal US$
    release_date      used only as a cross-check
    status            only "Released" films get money figures
    popularity, vote_count, original_language, production_companies
    overview, tagline, keywords   plot text: not needed here, but kept for Part B
                                  (the LLM concept generator), so we do not need
                                  the separate Wikipedia plots dataset.

Data-quality traps handled here (each counted in the returned report):
  1. "Unknown" is stored as 0. Most films have budget = 0 -> treated as missing.
  2. Some figures are typed in millions ("5" meaning $5M). Anything below
     `min_budget_usd` / `min_revenue_usd` (default $10k) is treated as missing.
     The floor is low on purpose: real micro-budget hits exist.
  3. revenue exactly equal to budget is almost always a copy-paste placeholder;
     the revenue is treated as missing.
  4. The same imdb_id can appear on several TMDB rows. We keep the row with the
     most information (money present, then most TMDB votes).
  5. Malformed or missing imdb_id rows cannot be joined and are dropped.
  6. Plot text contains commas, quotes and line breaks; pandas' CSV parser
     handles these when the file is properly quoted, which it is.

If no TMDB file is present, `load_tmdb` returns (None, report): the v2 pipeline
then runs the audience analysis only and says clearly what was skipped.
"""

from __future__ import annotations

import json
import logging
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from ._cache import _params_hash, cached_frame
from .config import TMDB_DOWNLOAD_URL, AnalysisConfig, Paths

logger = logging.getLogger(__name__)

# Columns we read when present. Only the first three are required.
REQUIRED = ("imdb_id", "budget", "revenue")
OPTIONAL = ("id", "title", "status", "release_date", "popularity", "vote_count",
            "original_language", "production_companies", "overview", "tagline", "keywords")
CHUNK_ROWS = 200_000
IMDB_ID_PATTERN = r"^tt\d{7,}$"


def find_tmdb_source(tmdb_dir: Path) -> Path | None:
    """The TMDB file to read: a CSV if one exists, else a zip, else None.

    If several CSVs are present, the largest wins (the main dataset is by far
    the biggest file).
    """
    if not tmdb_dir.exists():
        return None
    csvs = sorted(tmdb_dir.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if csvs:
        return csvs[0]
    zips = sorted(tmdb_dir.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    return zips[0] if zips else None


@contextmanager
def _open_csv(source: Path):
    """Yield a readable handle to the CSV, opening it inside a zip if needed."""
    if source.suffix.lower() != ".zip":
        yield source
        return
    with zipfile.ZipFile(source) as zf:
        members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV inside {source.name}.")
        member = max(members, key=lambda m: m.file_size)
        logger.info("Reading %s from inside %s", member.filename, source.name)
        with zf.open(member) as fh:
            yield fh


def _read_columns(source: Path) -> list[str]:
    with _open_csv(source) as fh:
        return list(pd.read_csv(fh, nrows=0).columns)


def clean_tmdb(raw: pd.DataFrame, cfg: AnalysisConfig, report: dict) -> pd.DataFrame:
    """Apply the data-quality rules above. Pure function, so it is unit-tested.

    Expects TMDB column names; returns one row per IMDb id with tidy columns:
    tconst, budget, revenue, tmdb_release_year, plus the optional columns.
    """
    df = raw.copy()
    df["imdb_id"] = df["imdb_id"].astype("string").str.strip()
    valid_id = df["imdb_id"].str.match(IMDB_ID_PATTERN, na=False)
    report["rows_without_valid_imdb_id"] = int((~valid_id).sum())
    df = df[valid_id]

    for col in ("budget", "revenue"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "status" in df:
        unreleased = df["status"].fillna("") != "Released"
        df.loc[unreleased, ["budget", "revenue"]] = np.nan

    # Trap 1 and 2: zeros and implausibly small figures -> unknown.
    report["budget_zero_or_missing"] = int((df["budget"].fillna(0) <= 0).sum())
    report["revenue_zero_or_missing"] = int((df["revenue"].fillna(0) <= 0).sum())
    tiny_b = df["budget"].between(0, cfg.min_budget_usd, inclusive="neither")
    tiny_r = df["revenue"].between(0, cfg.min_revenue_usd, inclusive="neither")
    report["budget_below_floor"] = int(tiny_b.sum())
    report["revenue_below_floor"] = int(tiny_r.sum())
    df.loc[df["budget"] < cfg.min_budget_usd, "budget"] = np.nan
    df.loc[df["revenue"] < cfg.min_revenue_usd, "revenue"] = np.nan

    # Trap 3: revenue copied from budget.
    same = df["budget"].notna() & (df["revenue"] == df["budget"])
    report["revenue_equal_to_budget"] = int(same.sum())
    df.loc[same, "revenue"] = np.nan

    # Trap 4: duplicates. Prefer rows with money, then the most TMDB votes.
    df["_info"] = df["budget"].notna().astype(int) + df["revenue"].notna().astype(int)
    votes = pd.to_numeric(df["vote_count"], errors="coerce") if "vote_count" in df else 0
    df["_votes"] = votes
    n_before = len(df)
    df = (df.sort_values(["_info", "_votes"], ascending=False)
            .drop_duplicates("imdb_id", keep="first"))
    report["duplicate_imdb_ids_removed"] = int(n_before - len(df))

    if "release_date" in df:
        df["tmdb_release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype("Int64")
    rename = {"imdb_id": "tconst", "id": "tmdb_id", "title": "tmdb_title",
              "popularity": "tmdb_popularity", "vote_count": "tmdb_vote_count"}
    df = df.rename(columns=rename).drop(columns=["_info", "_votes", "release_date", "status"],
                                        errors="ignore")
    df["tconst"] = df["tconst"].astype(str)
    for col in ("tmdb_popularity", "tmdb_vote_count"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_tmdb(paths: Paths, cfg: AnalysisConfig, movie_ids: set[str] | None = None,
              refresh: bool = False) -> tuple[pd.DataFrame | None, dict]:
    """Load, clean and (optionally) restrict TMDB to the given IMDb films.

    Returns (tmdb, report). tmdb is None when no TMDB file is found.
    The report is a dict of data-quality counts, worth one appendix slide.
    """
    source = find_tmdb_source(paths.tmdb_dir)
    if source is None:
        logger.warning("TMDB file not found in %s. Money analysis (budget, revenue, ROI) "
                       "will be skipped. Download: %s", paths.tmdb_dir, TMDB_DOWNLOAD_URL)
        return None, {"available": False}

    columns = _read_columns(source)
    missing = [c for c in REQUIRED if c not in columns]
    if missing:
        raise ValueError(f"{source.name} lacks required columns {missing}. "
                         f"Is this the asaniczka TMDB dataset? Columns found: {columns[:12]}...")
    usecols = [c for c in (*REQUIRED, *OPTIONAL) if c in columns]
    params = {"floors": [cfg.min_budget_usd, cfg.min_revenue_usd],
              "n_movies": len(movie_ids) if movie_ids else 0, "cols": usecols}
    report_path = paths.cache_dir / f"tmdb_report__{_params_hash(params)}.json"

    def build() -> pd.DataFrame:
        report: dict = {"available": True, "source": source.name, "rows_read": 0}
        parts = []
        with _open_csv(source) as fh:
            reader = pd.read_csv(fh, usecols=usecols, dtype=str, chunksize=CHUNK_ROWS,
                                 on_bad_lines="warn", keep_default_na=True)
            for chunk in reader:
                report["rows_read"] += len(chunk)
                if movie_ids is not None:  # keep memory small: only films we analyse
                    chunk = chunk[chunk["imdb_id"].isin(movie_ids)]
                parts.append(chunk)
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
        report["rows_matching_imdb_films"] = int(len(raw)) if movie_ids is not None else None
        df = clean_tmdb(raw, cfg, report)
        report["films_with_budget"] = int(df["budget"].notna().sum())
        report["films_with_revenue"] = int(df["revenue"].notna().sum())
        report["films_with_both"] = int((df["budget"].notna() & df["revenue"].notna()).sum())
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        return df

    df = cached_frame("tmdb_clean", build, paths.cache_dir, [source], params=params, refresh=refresh)
    report = json.loads(report_path.read_text()) if report_path.exists() else {"available": True}
    logger.info("TMDB: %s films joined, %s with both budget and revenue",
                f"{len(df):,}", f"{int((df['budget'].notna() & df['revenue'].notna()).sum()):,}")
    return df, report
