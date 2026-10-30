import numpy as np
import pandas as pd

from nextaaa import clustering, features, io_imdb


def test_find_knee_on_a_textbook_elbow():
    k = np.arange(1, 9)
    inertia = np.array([100, 40, 20, 17, 15, 14, 13, 12])
    assert clustering.find_knee(k, inertia) == 3


def test_kmeans_naming_matches_quadrants():
    centres = np.array([[1.0, 1.2], [1.1, -0.9], [-1.0, -1.0], [-0.8, 1.1]])
    names, hit_id = clustering.name_kmeans_clusters(centres)
    assert names == {0: "Hit", 1: "Hidden Gem", 2: "Loser", 3: "Popular but Divisive"}
    assert hit_id == 0


def test_kmeans_naming_other_k_still_finds_a_hit():
    centres = np.array([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    names, hit_id = clustering.name_kmeans_clusters(centres)
    assert hit_id == 0 and names[0].endswith("(Hit)")


def test_quadrant_of():
    q = pd.Series([1.0, 1.0, -1.0, -1.0])
    r = pd.Series([1.0, -1.0, -1.0, 1.0])
    assert clustering.quadrant_of(q, r).tolist() == ["Hit", "Hidden Gem", "Loser", "Popular but Divisive"]


def test_reach_is_year_adjusted(synthetic_paths, small_cfg):
    movies = io_imdb.load_movies(synthetic_paths, small_cfg)
    frame = features.build_cluster_frame(movies, small_cfg)
    # Within every year, the median film has reach_z ~ 0 by construction.
    med = frame.groupby("startYear")["reach_z"].median()
    assert np.allclose(med, 0, atol=0.15)
    assert (frame["numVotes"] >= small_cfg.min_votes_for_clustering).all()
    # Standardised clustering inputs.
    assert abs(frame["x_quality"].mean()) < 1e-9 and abs(frame["x_reach"].std(ddof=0) - 1) < 1e-9


def test_run_clustering_end_to_end(synthetic_paths, small_cfg):
    movies = io_imdb.load_movies(synthetic_paths, small_cfg)
    frame = features.build_cluster_frame(movies, small_cfg)
    res = clustering.run_clustering(frame, features.feature_matrix(frame), small_cfg)
    out = res.frame
    # Majority vote is consistent with its parts.
    assert (out["is_hit"] == (out["hit_votes"] >= 2)).all()
    assert set(out["hit_tier"]) <= {"Mega-hit", "Hit", "Not a hit"}
    assert (out.loc[out["hit_tier"] == "Mega-hit", "hit_dbscan"]).all()
    # Hits should be better rated and better voted than non-hits.
    hits, rest = out[out["is_hit"]], out[~out["is_hit"]]
    assert hits["averageRating"].median() > rest["averageRating"].median()
    assert hits["reach_z"].median() > rest["reach_z"].median()
    assert 0 < out["is_hit"].mean() < 0.5
    assert set(res.kmeans.table["k"]) == set(small_cfg.k_range)


def test_dbscan_sampling_path_matches_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 2))
    labels = clustering.fit_dbscan(X, eps=0.3, min_samples=10, max_points=1000, random_state=0)
    assert labels.shape == (3000,)
    assert (labels == -1).mean() < 0.5
