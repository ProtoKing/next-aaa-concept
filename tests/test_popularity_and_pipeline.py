import json

import pandas as pd
import pytest

from nextaaa.pipeline import run_part_a
from nextaaa.popularity import actor_popularity


def test_two_definitions_can_disagree():
    """Actor A: two huge but poorly rated films. Actor B: one big, acclaimed film."""
    movies = pd.DataFrame({
        "tconst": ["f1", "f2", "f3", "f4"], "primaryTitle": ["Flop1", "Flop2", "Gem", "Filler"],
        "startYear": [2023] * 4, "averageRating": [5.0, 5.2, 8.8, 6.5],
        "numVotes": [400_000, 350_000, 500_000, 20_000],
    })
    credits = pd.DataFrame({"tconst": ["f1", "f2", "f3", "f4"],
                            "nconst": ["A", "A", "B", "C"], "billing_rank": [1, 1, 1, 1]})
    names = pd.DataFrame({"nconst": ["A", "B", "C"], "primaryName": ["Actor A", "Actor B", "Actor C"]})
    pop = actor_popularity(movies, credits, names, 2023, quantile=0.5).set_index("primaryName")
    assert pop.loc["Actor A", "rank_reach"] == 1      # most total votes
    assert pop.loc["Actor B", "rank_goodwill"] == 1   # most liked
    assert pop.loc["Actor A", "net_goodwill"] < 0     # below-average films subtract goodwill


def test_billing_weight_rewards_leads():
    movies = pd.DataFrame({"tconst": ["f1"], "primaryTitle": ["X"], "startYear": [2023],
                           "averageRating": [7.0], "numVotes": [10_000]})
    credits = pd.DataFrame({"tconst": ["f1", "f1"], "nconst": ["lead", "fourth"], "billing_rank": [1, 4]})
    names = pd.DataFrame({"nconst": ["lead", "fourth"], "primaryName": ["Lead", "Fourth"]})
    pop = actor_popularity(movies, credits, names, 2023, 0.9).set_index("primaryName")
    assert pop.loc["Fourth", "reach"] == pytest.approx(0.5 * pop.loc["Lead", "reach"])


def test_full_pipeline_writes_everything(synthetic_paths, small_cfg):
    summary = run_part_a(synthetic_paths, small_cfg)
    out = synthetic_paths.output_dir
    assert json.loads((out / "summary.json").read_text())["movielens_used"] is True
    for name in ["01_rating_vs_votes_2023", "04_elbow_silhouette", "06_cluster_maps",
                 "07_hit_tiers", "08_imdb_genre_trends", "10_movielens_yearly"]:
        assert (out / "figures" / f"{name}.png").exists(), name
    labels = pd.read_csv(out / "tables" / "hit_labels.csv")
    assert {"is_hit", "hit_tier", "cluster_kmeans"} <= set(labels.columns)
    assert summary["clustering"]["k_used"] == small_cfg.k_hypothesis
