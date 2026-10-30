"""Load the IMDb non-commercial datasets safely and efficiently.

Download the files from https://datasets.imdbws.com/ (field definitions at
https://developer.imdb.com/non-commercial-datasets/) and put them, still
gzipped, in one folder:

    data/imdb/title.basics.tsv.gz       one row per title (movie, series, episode...)
    data/imdb/title.ratings.tsv.gz      averageRating and numVotes per title
    data/imdb/title.principals.tsv.gz   top-billed cast and crew per title
    data/imdb/name.basics.tsv.gz        names of people

Three traps in these files, all handled here:
  1. Missing values are written as the two characters "\\N", not as blanks.
  2. Titles contain quote characters, which break pandas' default CSV quoting,
     so we read with quoting disabled (csv.QUOTE_NONE).
  3. Some titles are literally named "NA" or "None"; pandas would silently turn
     those into missing values unless we switch off its default NA list.

Memory: title.basics (~12M rows) and title.principals (~100M rows) are read in
chunks and filtered as they stream in, so only the rows we need are ever held
in memory. Each result is cached (see _cache.py), so this happens once.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from ._cache import cached_frame
from .config import IMDB_DOWNLOAD_URL, AnalysisConfig, Paths

logger = logging.getLogger(__name__)

IMDB_FILES = {
    "basics": "title.basics.tsv.gz",
    "ratings": "title.ratings.tsv.gz",
    "principals": "title.principals.tsv.gz",
    "names": "name.basics.tsv.gz",
}

CHUNK_ROWS = 2_000_000  # rows per chunk when streaming the big files


class MissingDataError(FileNotFoundError):
    """Raised when a required data file is not where we expect it."""


def find_imdb_file(imdb_dir: Path, key: str) -> Path:
    """Locate an IMDb file, accepting either the .tsv.gz or an unzipped .tsv."""
    gz_name = IMDB_FILES[key]
    for candidate in (imdb_dir / gz_name, imdb_dir / gz_name.removesuffix(".gz")):
        if candidate.exists():
            return candidate
    raise MissingDataError(
        f"Could not find '{gz_name}' in {imdb_dir.resolve()}.\n"
        f"  - Download it from {IMDB_DOWNLOAD_URL} and place it in that folder, or\n"
        f"  - point the code at your folder with --imdb-dir or the NEXTAAA_IMDB_DIR env var."
    )


def _read_tsv(path: Path, usecols: list[str], chunksize: int | None = None):
    """Read an IMDb TSV with the three traps above handled.

    Every column is read as text first; numeric conversion happens afterwards
    with errors="coerce", so one malformed value can never crash the load.
    """
    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype=str,
        na_values=["\\N"],
        keep_default_na=False,  # trap 3: keep films literally titled "NA"
        quoting=csv.QUOTE_NONE,  # trap 2: quotes inside titles
        on_bad_lines="warn",
        encoding="utf-8",
        chunksize=chunksize,
    )


def _to_int(series: pd.Series) -> pd.Series:
    """Text -> nullable integer. Anything unparseable becomes <NA>."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


# ---------------------------------------------------------------------------
# title.basics
# ---------------------------------------------------------------------------

def load_title_basics(paths: Paths, cfg: AnalysisConfig, refresh: bool = False) -> pd.DataFrame:
    """Titles of the configured types (default: feature films only).

    Columns: tconst, titleType, primaryTitle, isAdult, startYear,
    runtimeMinutes, genres (comma-separated text, e.g. "Action,Sci-Fi").
    """
    path = find_imdb_file(paths.imdb_dir, "basics")
    keep_types = set(cfg.title_types)

    def build() -> pd.DataFrame:
        cols = ["tconst", "titleType", "primaryTitle", "isAdult",
                "startYear", "runtimeMinutes", "genres"]
        parts = []
        for chunk in _read_tsv(path, cols, chunksize=CHUNK_ROWS):
            parts.append(chunk[chunk["titleType"].isin(keep_types)])
        df = pd.concat(parts, ignore_index=True)
        df["isAdult"] = _to_int(df["isAdult"]).fillna(0).astype("int8")
        df["startYear"] = _to_int(df["startYear"])
        df["runtimeMinutes"] = _to_int(df["runtimeMinutes"])
        return df

    return cached_frame("title_basics", build, paths.cache_dir, [path],
                        params={"types": sorted(keep_types)}, refresh=refresh)


# ---------------------------------------------------------------------------
# title.ratings
# ---------------------------------------------------------------------------

def load_title_ratings(paths: Paths, refresh: bool = False) -> pd.DataFrame:
    """averageRating (IMDb's weighted average, 1-10) and numVotes per title."""
    path = find_imdb_file(paths.imdb_dir, "ratings")

    def build() -> pd.DataFrame:
        df = _read_tsv(path, ["tconst", "averageRating", "numVotes"])
        df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")
        df["numVotes"] = _to_int(df["numVotes"])
        return df.dropna(subset=["averageRating", "numVotes"]).astype({"numVotes": "int64"})

    return cached_frame("title_ratings", build, paths.cache_dir, [path], refresh=refresh)


# ---------------------------------------------------------------------------
# The analysis table: one row per rated feature film
# ---------------------------------------------------------------------------

def build_movie_table(basics: pd.DataFrame, ratings: pd.DataFrame,
                      cfg: AnalysisConfig) -> pd.DataFrame:
    """Join titles to ratings and apply the "what counts as a movie" filters.

    This is a pure function (no file access), so it is easy to unit-test.
    Films with no rating are dropped: they cannot be scored.
    """
    df = basics.merge(ratings, on="tconst", how="inner")

    if cfg.exclude_adult:
        df = df[df["isAdult"] == 0]

    runtime_ok = df["runtimeMinutes"] >= cfg.min_runtime_minutes
    if cfg.keep_missing_runtime:
        runtime_ok = runtime_ok | df["runtimeMinutes"].isna()
    df = df[runtime_ok.fillna(False).astype(bool)]

    # A film with no release year cannot be placed in time; drop it.
    df = df[df["startYear"].notna()].copy()
    df["startYear"] = df["startYear"].astype("int64")
    df["genres"] = df["genres"].fillna("Unknown")

    logger.info("Movie table: %s rated feature films", f"{len(df):,}")
    return df.reset_index(drop=True)


def load_movies(paths: Paths, cfg: AnalysisConfig, refresh: bool = False) -> pd.DataFrame:
    """Convenience wrapper: load basics + ratings and build the movie table."""
    basics = load_title_basics(paths, cfg, refresh=refresh)
    ratings = load_title_ratings(paths, refresh=refresh)
    return build_movie_table(basics, ratings, cfg)


# ---------------------------------------------------------------------------
# Cast: title.principals + name.basics
# ---------------------------------------------------------------------------

def load_actor_credits(paths: Paths, cfg: AnalysisConfig, movie_ids: set[str],
                       refresh: bool = False) -> pd.DataFrame:
    """Actor/actress credits for the given films, with billing rank.

    title.principals lists the top-billed people per title in `ordering`
    (1 = first credit), mixing actors with directors, writers, composers...
    We keep only actors/actresses and re-number them 1, 2, 3... within each
    film, so `billing_rank` 1 is always the top-billed performer.

    Columns: tconst, nconst, billing_rank.
    """
    path = find_imdb_file(paths.imdb_dir, "principals")
    categories = set(cfg.actor_categories)

    def build() -> pd.DataFrame:
        cols = ["tconst", "ordering", "nconst", "category"]
        parts = []
        for chunk in _read_tsv(path, cols, chunksize=CHUNK_ROWS):
            mask = chunk["category"].isin(categories) & chunk["tconst"].isin(movie_ids)
            parts.append(chunk.loc[mask, ["tconst", "ordering", "nconst"]])
        df = pd.concat(parts, ignore_index=True)
        df["ordering"] = _to_int(df["ordering"])
        df = df.dropna(subset=["ordering"]).sort_values(["tconst", "ordering"])
        # The same person can appear twice for one film (e.g. two roles); keep the top credit.
        df = df.drop_duplicates(["tconst", "nconst"], keep="first")
        df["billing_rank"] = df.groupby("tconst").cumcount() + 1
        return df[["tconst", "nconst", "billing_rank"]].reset_index(drop=True)

    # The cache depends on which films we filtered to; the movie set is derived
    # deterministically from title.basics, so fingerprint by its size and types.
    params = {"cats": sorted(categories), "types": sorted(cfg.title_types), "n_movies": len(movie_ids)}
    basics_path = find_imdb_file(paths.imdb_dir, "basics")
    return cached_frame("actor_credits", build, paths.cache_dir, [path, basics_path],
                        params=params, refresh=refresh)


def load_names(paths: Paths, person_ids: set[str], refresh: bool = False) -> pd.DataFrame:
    """primaryName for the given people. Columns: nconst, primaryName."""
    path = find_imdb_file(paths.imdb_dir, "names")

    def build() -> pd.DataFrame:
        parts = []
        for chunk in _read_tsv(path, ["nconst", "primaryName"], chunksize=CHUNK_ROWS):
            parts.append(chunk[chunk["nconst"].isin(person_ids)])
        return pd.concat(parts, ignore_index=True)

    return cached_frame("names", build, paths.cache_dir, [path],
                        params={"n_people": len(person_ids)}, refresh=refresh)
