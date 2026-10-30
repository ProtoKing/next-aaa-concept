"""Shared test fixtures: a small synthetic dataset in the real IMDb/MovieLens formats."""

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextaaa.config import AnalysisConfig, Paths  # noqa: E402
from nextaaa.synthetic import (make_synthetic_cpi, make_synthetic_imdb,  # noqa: E402
                               make_synthetic_movielens, make_synthetic_tmdb)


@pytest.fixture(scope="session")
def synthetic_paths(tmp_path_factory) -> Paths:
    root = tmp_path_factory.mktemp("data")
    make_synthetic_imdb(root / "imdb", n_movies=4_000)
    make_synthetic_movielens(root / "imdb", root / "ml-20m", n_users=400, n_films=300)
    return Paths.resolve(root / "imdb", root / "ml-20m", root / "cache", root / "outputs")


@pytest.fixture(scope="session")
def small_cfg() -> AnalysisConfig:
    """Thresholds relaxed for the small fake dataset."""
    return replace(AnalysisConfig(), min_votes_for_clustering=200, min_films_per_year=10,
                   movielens_min_ratings_per_year=200, movielens_min_ratings_for_std=10,
                   dbscan_min_samples=10, k_range=(2, 3, 4, 5), silhouette_sample_size=3_000)


@pytest.fixture(scope="session")
def v2_paths(synthetic_paths, tmp_path_factory) -> Paths:
    """Synthetic data plus TMDB (as a Kaggle-style zip, to exercise that path) and CPI."""
    root = tmp_path_factory.mktemp("v2data")
    make_synthetic_tmdb(synthetic_paths.imdb_dir, root / "tmdb", as_zip=True)
    make_synthetic_cpi(root / "cpi")
    return replace(synthetic_paths, tmdb_dir=root / "tmdb", cpi_dir=root / "cpi",
                   cache_dir=root / "cache", output_dir=root / "out")


@pytest.fixture(scope="session")
def v2_cfg(small_cfg) -> AnalysisConfig:
    return replace(small_cfg, min_ref_films_audience=20, min_ref_films_finance=15)
