import pytest

from nextaaa import io_imdb
from nextaaa.config import Paths
from nextaaa.movielens import imdb_id_to_tconst, load_movielens


def test_missing_file_gives_a_helpful_error(tmp_path):
    paths = Paths.resolve(tmp_path / "nope", tmp_path, tmp_path / "cache", tmp_path / "out")
    with pytest.raises(io_imdb.MissingDataError, match="datasets.imdbws.com"):
        io_imdb.find_imdb_file(paths.imdb_dir, "basics")


def test_loader_handles_imdb_traps(synthetic_paths, small_cfg):
    basics = io_imdb.load_title_basics(synthetic_paths, small_cfg)
    titles = set(basics["primaryTitle"])
    assert 'The "Quoted" Picture' in titles   # quotes inside titles survive
    assert "NA" in titles                      # a film called "NA" is not a missing value
    assert set(basics["titleType"]) == {"movie"}  # non-movies filtered while streaming


def test_movie_table_filters(synthetic_paths, small_cfg):
    movies = io_imdb.load_movies(synthetic_paths, small_cfg)
    assert (movies["isAdult"] == 0).all()
    rt = movies["runtimeMinutes"].dropna()
    assert (rt >= small_cfg.min_runtime_minutes).all()
    assert movies["runtimeMinutes"].isna().any()  # missing runtimes kept by default
    assert movies["startYear"].notna().all()
    assert movies["tconst"].is_unique


def test_cache_is_reused(synthetic_paths, small_cfg):
    a = io_imdb.load_title_ratings(synthetic_paths)
    b = io_imdb.load_title_ratings(synthetic_paths)  # second call reads the cache
    assert a.equals(b)


def test_actor_credits_billing(synthetic_paths, small_cfg):
    movies = io_imdb.load_movies(synthetic_paths, small_cfg)
    credits = io_imdb.load_actor_credits(synthetic_paths, small_cfg, set(movies["tconst"]))
    first = credits.groupby("tconst")["billing_rank"].min()
    assert (first == 1).all()  # directors/writers removed, actors renumbered from 1
    assert not credits.duplicated(["tconst", "nconst"]).any()


def test_imdb_id_conversion():
    import pandas as pd
    out = imdb_id_to_tconst(pd.Series(["114709", "0113497", "x"]))
    assert out.tolist()[:2] == ["tt0114709", "tt0113497"]
    assert pd.isna(out.iloc[2])


def test_movielens_optional(tmp_path):
    paths = Paths.resolve(tmp_path, tmp_path / "no-movielens", tmp_path / "cache", tmp_path / "out")
    assert load_movielens(paths) is None  # missing data -> None, not a crash
