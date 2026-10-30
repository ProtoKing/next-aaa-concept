"""Tests for v2: TMDB loading, CPI, finance, year normalisation, five-method stages, verdicts."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nextaaa import ensemble, finance, hits, io_imdb, io_tmdb
from nextaaa.config import AnalysisConfig, Paths
from nextaaa.normalize import year_relative
from nextaaa.pipeline_v2 import run_part_a_v2


# ---------------------------------------------------------------------------
# Year normalisation
# ---------------------------------------------------------------------------

def test_year_relative_median_is_zero_and_pct_is_mid_rank():
    years = pd.Series([2000] * 5)
    out = year_relative(pd.Series([1.0, 2, 3, 4, 5]), years, min_ref=3, max_window=0)
    assert out.loc[2, "z"] == pytest.approx(0.0)
    assert out.loc[2, "pct"] == pytest.approx(50.0)
    assert out.loc[4, "pct"] == pytest.approx(90.0)  # mid-rank: (4 + 0.5) / 5


def test_year_relative_removes_era_shifts():
    # 1990 films are all 10x lower than 2020 films (e.g. inflation): same z after normalising.
    v = pd.Series(np.r_[np.arange(30.0), np.arange(30.0) + 10])
    y = pd.Series([1990] * 30 + [2020] * 30)
    out = year_relative(v, y, min_ref=10, max_window=0)
    assert np.allclose(out["z"].iloc[:30], out["z"].iloc[30:])


def test_year_relative_widens_window_for_thin_years():
    v = pd.Series(np.arange(25.0))
    y = pd.Series([1980] * 3 + [1981] * 22)  # 1980 alone is too thin
    out = year_relative(v, y, min_ref=20, max_window=2)
    assert out.loc[0, "ref_window"] == 1 and out.loc[0, "ref_n"] == 25
    assert out.loc[10, "ref_window"] == 0
    # Too thin even with the widest window -> NaN, not a made-up score.
    out2 = year_relative(v.iloc[:3], y.iloc[:3], min_ref=20, max_window=2)
    assert out2["z"].isna().all()


# ---------------------------------------------------------------------------
# TMDB cleaning and loading
# ---------------------------------------------------------------------------

def test_clean_tmdb_rules():
    cfg = AnalysisConfig()
    raw = pd.DataFrame({
        "imdb_id": ["tt0000001", "tt0000002", "tt0000003", "tt0000004", "tt0000004", "bad", None],
        "budget": ["0", "5", "20000000", "1000000", "0", "1", "1"],
        "revenue": ["100", "50000000", "20000000", "9000000", "0", "1", "1"],
        "status": ["Released"] * 7, "vote_count": ["1", "1", "1", "50", "999", "1", "1"],
    })
    report: dict = {}
    out = io_tmdb.clean_tmdb(raw, cfg, report).set_index("tconst")
    assert pd.isna(out.loc["tt0000001", "budget"])        # 0 = unknown
    assert pd.isna(out.loc["tt0000001", "revenue"])       # below floor
    assert pd.isna(out.loc["tt0000002", "budget"])        # "5" (millions) -> unknown
    assert pd.isna(out.loc["tt0000003", "revenue"])       # revenue copied from budget
    assert out.loc["tt0000004", "budget"] == 1_000_000    # duplicate: richer row kept
    assert "bad" not in out.index and report["rows_without_valid_imdb_id"] == 2
    assert report["duplicate_imdb_ids_removed"] == 1


def test_load_tmdb_from_zip_and_missing(v2_paths, v2_cfg, tmp_path):
    movies = io_imdb.load_movies(v2_paths, v2_cfg)
    tmdb, report = io_tmdb.load_tmdb(v2_paths, v2_cfg, set(movies["tconst"]))
    assert tmdb["tconst"].is_unique
    assert set(tmdb["tconst"]) <= set(movies["tconst"])   # TMDB-only films dropped
    assert report["films_with_both"] > 100
    assert (tmdb["budget"].dropna() >= v2_cfg.min_budget_usd).all()
    none, rep = io_tmdb.load_tmdb(replace(v2_paths, tmdb_dir=tmp_path / "nothing"), v2_cfg)
    assert none is None and rep == {"available": False}


# ---------------------------------------------------------------------------
# CPI and finance
# ---------------------------------------------------------------------------

def test_parse_cpi_layouts():
    fred_new = pd.DataFrame({"observation_date": ["2022-01-01", "2022-07-01", "2023-01-01"],
                             "CPIAUCSL": [280.0, 300.0, 300.0]})
    fred_old = fred_new.rename(columns={"observation_date": "DATE"})
    simple = pd.DataFrame({"year": [2022, 2023], "cpi": [290.0, 300.0]})
    for df in (fred_new, fred_old, simple):
        cpi = finance.parse_cpi(df)
        assert cpi.loc[2022] == pytest.approx(290.0)


def test_deflator_falls_back_to_latest_year():
    cpi = pd.Series({2020: 100.0, 2021: 110.0})
    factor, base = finance.deflator(cpi, 2023)
    assert base == 2021 and factor.loc[2020] == pytest.approx(1.1)


def test_add_finance_roi_and_gate():
    cfg = AnalysisConfig()
    df = pd.DataFrame({"startYear": [2000, 2000, 1960, 2000],
                       "budget": [10e6, 10e6, 10e6, np.nan], "revenue": [30e6, 20e6, 50e6, 5e6]})
    out = finance.add_finance(df, cfg, cpi=None)
    assert out["roi_multiple"].iloc[0] == pytest.approx(3.0)
    assert out["profitable"].iloc[0] == 1.0 and out["profitable"].iloc[1] == 0.0
    assert out["above_breakeven_usd"].iloc[0] == pytest.approx(5e6)
    assert not out["has_finance"].iloc[2]   # before finance_min_year
    assert not out["has_finance"].iloc[3]   # budget unknown
    assert out.attrs["dollar_basis"] == "nominal US$"
    cpi = pd.Series({2000: 50.0, 2023: 100.0})
    real = finance.add_finance(df, cfg, cpi)
    assert real["budget_real"].iloc[0] == pytest.approx(20e6)


# ---------------------------------------------------------------------------
# Stages and verdicts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def staged(v2_paths, v2_cfg):
    movies = io_imdb.load_movies(v2_paths, v2_cfg)
    tmdb, _ = io_tmdb.load_tmdb(v2_paths, v2_cfg, set(movies["tconst"]))
    frame = hits.build_v2_frame(movies, tmdb, v2_cfg, finance.load_cpi(v2_paths))
    aud = ensemble.run_stage(frame, ensemble.AUDIENCE, v2_cfg)
    com = ensemble.run_stage(frame[frame["has_finance"]], ensemble.COMMERCIAL, v2_cfg)
    return frame, aud, com, hits.combine_stages(frame, aud.frame, com.frame, v2_cfg)


def test_stage_votes_are_consistent(staged, v2_cfg):
    _, aud, com, _ = staged
    for r in (aud, com):
        p, df = r.stage.key, r.frame
        votes = df[[f"{p}_vote_{m}" for m in ensemble.METHODS]].sum(axis=1)
        assert (votes == df[f"{p}_votes"]).all()
        assert (df[f"{p}_hit"] == (votes >= v2_cfg.hit_min_votes)).all()
        # Outlier methods only vote for films in the hit corner.
        corner = df[f"{p}_corner"] == r.stage.hit_name
        for m in ("dbscan", "hdbscan", "iforest"):
            assert not (df[f"{p}_vote_{m}"] & ~corner).any()
        assert r.agreement.shape == (5, 5)
        assert np.allclose(np.diag(r.agreement), 1.0)


def test_commercial_hits_cleared_breakeven(staged, v2_cfg):
    *_, out = staged
    com_hits = out[out["com_hit"].fillna(False).astype(bool)]
    assert len(com_hits) > 0
    assert (com_hits["roi_multiple"] >= v2_cfg.breakeven_multiple).all()


def test_verdict_logic(staged):
    *_, out = staged
    fin = out[out["has_finance"]]
    assert set(fin["verdict"]) <= set(hits.VERDICTS)
    assert set(out.loc[~out["has_finance"], "verdict"]) <= {hits.NO_MONEY, hits.NO_MONEY_HIT}
    aaa = out[out["is_aaa_hit"]]
    assert aaa["aud_hit"].all() and aaa["com_hit"].astype(bool).all()
    grid = hits.verdict_grid(out)
    assert grid.to_numpy().sum() == len(fin)
    assert grid.loc["Audience hit", "Commercial hit"] == len(aaa)
    # AAA hits should beat the rest on every axis.
    rest = fin[~fin["is_aaa_hit"]]
    for c in ("pct_quality", "pct_reach", "pct_roi", "pct_revenue"):
        assert aaa[c].median() > rest[c].median()


def test_by_year_and_all_time(staged, v2_cfg):
    *_, out = staged
    by_year = hits.hits_by_year(out, top_n=3)
    assert by_year.groupby("startYear").size().max() <= 3
    assert by_year.groupby("startYear")["hit_score"].apply(lambda s: s.is_monotonic_decreasing).all()
    top = hits.all_time_hits(out, 10)
    assert top["hit_score"].is_monotonic_decreasing and top["rank_all_time"].tolist()[:2] == [1, 2]
    yearly = hits.yearly_summary(out)
    assert {"hit_of_the_year", "aaa_hit_pct_of_money_films"} <= set(yearly.columns)


# ---------------------------------------------------------------------------
# End to end, with and without the optional inputs
# ---------------------------------------------------------------------------

def test_pipeline_v2_full(v2_paths, v2_cfg):
    summary = run_part_a_v2(v2_paths, v2_cfg)
    out = v2_paths.output_dir / "v2"
    assert json.loads((out / "summary.json").read_text())["cpi_available"] is True
    assert summary["aaa_hits"] > 0
    for name in ("03_actors_2023", "04_budget_vs_revenue", "08_aud_method_maps",
                 "11_com_method_maps", "12_method_agreement", "15_all_time_aaa_hits"):
        assert (out / "figures" / f"{name}.png").exists(), name
    assert (out / "tables" / "aaa_hits_by_year.csv").exists()


def test_pipeline_v2_without_tmdb_or_cpi(v2_paths, v2_cfg, tmp_path):
    paths = replace(v2_paths, tmdb_dir=tmp_path / "no_tmdb", cpi_dir=tmp_path / "no_cpi",
                    output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    summary = run_part_a_v2(paths, v2_cfg, make_plots=False)
    assert summary["tmdb"] == {"available": False}
    assert "aaa_hits" not in summary and summary["audience_hits"] > 0


def test_glosh_fallback_without_hdbscan(monkeypatch):
    monkeypatch.setattr(ensemble, "_hdbscan_lib", None)
    X = np.random.default_rng(0).normal(size=(500, 2))
    X[0] = [8, 8]  # an obvious outlier
    scores, labels, how = ensemble.glosh_scores(X, 50, 10, 10_000, 0)
    assert "approximation" in how and scores.argmax() == 0
