import numpy as np
import pandas as pd
import pytest

from nextaaa.ratings import add_shrunk_rating, shrinkage_params, top_rated, weighted_rating


def test_weighted_rating_limits():
    # No votes -> the prior mean C; huge votes -> the film's own rating R.
    assert weighted_rating(9.0, 0, m=1000, C=6.0) == pytest.approx(6.0)
    assert weighted_rating(9.0, 1e9, m=1000, C=6.0) == pytest.approx(9.0, abs=1e-4)
    # v == m -> exactly halfway.
    assert weighted_rating(9.0, 1000, m=1000, C=6.0) == pytest.approx(7.5)


def test_worked_example_from_the_conversation():
    # 12 votes at 9.4 is pulled to ~C; 900k votes at 8.6 barely moves.
    small = weighted_rating(9.4, 12, m=25_000, C=6.2)
    big = weighted_rating(8.6, 900_000, m=25_000, C=6.2)
    assert small == pytest.approx(6.2, abs=0.01)
    assert big == pytest.approx(8.54, abs=0.01)
    assert big > small


def test_shrinkage_params():
    pool = pd.DataFrame({"numVotes": np.arange(1, 101), "averageRating": [6.0] * 100})
    m, C = shrinkage_params(pool, 0.9)
    assert m == pytest.approx(90.1)
    assert C == pytest.approx(6.0)


def test_shrunk_by_group_uses_each_groups_params():
    df = pd.DataFrame({
        "startYear": [2000, 2000, 2020, 2020],
        "averageRating": [5.0, 7.0, 8.0, 6.0],
        "numVotes": [100, 100, 10, 10],
    })
    out = add_shrunk_rating(df, 0.5, by="startYear")
    assert out.loc[out.startYear == 2000, "shrink_C"].iloc[0] == pytest.approx(6.0)
    assert out.loc[out.startYear == 2020, "shrink_C"].iloc[0] == pytest.approx(7.0)


def test_top_rated_prefers_well_voted_film():
    movies = pd.DataFrame({
        "tconst": ["a", "b", "c"], "primaryTitle": ["Obscure", "Blockbuster", "Meh"],
        "startYear": [2023] * 3, "averageRating": [9.8, 8.6, 6.0],
        "numVotes": [15, 900_000, 50_000], "genres": ["Drama"] * 3,
    })
    board, info = top_rated(movies, 2023, quantile=0.9)
    assert board.iloc[0]["primaryTitle"] == "Blockbuster"
    # ...even though the obscure film is #1 on the raw average.
    assert board.set_index("primaryTitle").loc["Obscure", "rank_raw"] == 1
    assert info["pool_size"] == 3
