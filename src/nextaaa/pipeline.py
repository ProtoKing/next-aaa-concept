"""Run the whole of Part A in order, saving every table and figure.

    from nextaaa.config import Paths, AnalysisConfig
    from nextaaa.pipeline import run_part_a
    summary = run_part_a(Paths.resolve(), AnalysisConfig())

Outputs (under outputs/ by default):
    tables/  top_rated_<year>.csv, m_sensitivity_<year>.csv, actor_popularity_<year>.csv,
             actors_side_by_side_<year>.csv, kmeans_diagnostics.csv, dbscan_eps_sweep.csv,
             cluster_profile_*.csv, label_agreement.csv, hit_labels.csv,
             imdb_genre_trends.csv, movielens_*.csv (if MovieLens is present)
    figures/ 01_... to 12_... PNGs, numbered in deck order
    summary.json  the headline numbers, for the deck
The labelled film table is also cached as data/cache/hit_labels.parquet for Part B.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import pandas as pd

from . import clustering, features, io_imdb, movielens, plots, popularity, ratings, trends
from .config import AnalysisConfig, Paths

logger = logging.getLogger(__name__)


def _write(df: pd.DataFrame, paths: Paths, name: str, index: bool = False) -> None:
    df.to_csv(paths.tables_dir / f"{name}.csv", index=index)


def run_part_a(paths: Paths, cfg: AnalysisConfig, k: int | None = None,
               refresh: bool = False, make_plots: bool = True) -> dict:
    """Run every Part A step. Returns the headline summary (also saved as JSON)."""
    paths.make_output_dirs()
    fig_dir = paths.figures_dir
    summary: dict = {"config": {k_: (list(v) if isinstance(v, tuple) else v)
                                for k_, v in asdict(cfg).items()}}

    # -- Step 1: data ---------------------------------------------------------
    logger.info("Step 1/5: loading IMDb")
    movies = io_imdb.load_movies(paths, cfg, refresh=refresh)
    summary["n_rated_movies"] = int(len(movies))

    # -- Step 2: highest-rated film of the target year -----------------------
    logger.info("Step 2/5: highest-rated film of %d", cfg.target_year)
    year = cfg.target_year
    board, info = ratings.top_rated(movies, year, cfg.shrinkage_quantile,
                                    cfg.min_votes_target_year, top_n=25)
    sens = ratings.sensitivity_to_m(movies, year, cfg.sensitivity_quantiles,
                                    cfg.min_votes_target_year)
    _write(board, paths, f"top_rated_{year}")
    _write(sens, paths, f"m_sensitivity_{year}")
    top1_by_q = sens[sens["rank"] == 1].set_index("quantile")["primaryTitle"].to_dict()
    summary["top_rated"] = {**info, "winner": board.iloc[0]["primaryTitle"],
                            "winner_rating": float(board.iloc[0]["averageRating"]),
                            "winner_votes": int(board.iloc[0]["numVotes"]),
                            "winner_by_quantile": {str(q): t for q, t in top1_by_q.items()}}
    if make_plots:
        plots.save(plots.plot_rating_vs_votes(movies, info, board), fig_dir / f"01_rating_vs_votes_{year}.png")
        plots.save(plots.plot_m_sensitivity(sens, cfg.shrinkage_quantile), fig_dir / f"02_m_sensitivity_{year}.png")

    # -- Step 3: most popular actor of the target year -------------------------
    logger.info("Step 3/5: most popular actor of %d", year)
    credits = io_imdb.load_actor_credits(paths, cfg, set(movies["tconst"]), refresh=refresh)
    year_ids = set(movies.loc[movies["startYear"] == year, "tconst"])
    people = set(credits.loc[credits["tconst"].isin(year_ids), "nconst"])
    names = io_imdb.load_names(paths, people, refresh=refresh)
    pop = popularity.actor_popularity(movies, credits, names, year, cfg.shrinkage_quantile,
                                      cfg.billing_weight_power, cfg.min_votes_target_year)
    sbs = popularity.side_by_side(pop, cfg.top_n_actors)
    _write(pop, paths, f"actor_popularity_{year}")
    _write(sbs, paths, f"actors_side_by_side_{year}", index=True)
    summary["actors"] = {"top_by_reach": pop.nsmallest(1, "rank_reach")["primaryName"].iloc[0],
                         "top_by_goodwill": pop.nsmallest(1, "rank_goodwill")["primaryName"].iloc[0]}
    if make_plots:
        plots.save(plots.plot_actor_side_by_side(pop, year), fig_dir / f"03_actors_{year}.png")

    # -- Step 4: hit metric via clustering ---------------------------------------
    logger.info("Step 4/5: clustering")
    frame = features.build_cluster_frame(movies, cfg)
    X = features.feature_matrix(frame)
    res = clustering.run_clustering(frame, X, cfg, k=k)
    labelled = res.frame

    _write(res.kmeans.table, paths, "kmeans_diagnostics")
    _write(res.eps_sweep, paths, "dbscan_eps_sweep")
    _write(res.agreement, paths, "label_agreement")
    for col in ("cluster_kmeans", "cluster_dbscan", "quadrant_median", "hit_tier"):
        _write(clustering.cluster_profile(labelled, col), paths, f"cluster_profile_{col}", index=True)
    keep = ["tconst", "primaryTitle", "startYear", "genres", "runtimeMinutes", "averageRating",
            "numVotes", "reach_z", "reach_pct", "shrunk_rating", "cluster_kmeans", "cluster_dbscan",
            "quadrant_median", "hit_kmeans", "hit_dbscan", "hit_quadrant", "hit_votes",
            "is_hit", "hit_tier"]
    _write(labelled[keep], paths, "hit_labels")
    labelled[keep].to_pickle(paths.cache_dir / "hit_labels.pkl")  # handoff to Part B

    tier_counts = labelled["hit_tier"].value_counts().to_dict()
    summary["clustering"] = {
        "n_films": int(len(labelled)),
        "years": [int(labelled["startYear"].min()), int(labelled["startYear"].max())],
        "k_elbow": res.kmeans.k_elbow, "k_silhouette": res.kmeans.k_silhouette, "k_used": res.k_used,
        "silhouette_k_used": float(res.kmeans.table.set_index("k").loc[res.k_used, "silhouette"]),
        "dbscan_eps": res.eps,
        "dbscan_noise_share": float((labelled["dbscan_id"] == -1).mean()),
        "dbscan_silhouette": {k_: (None if pd.isna(v) else float(v)) for k_, v in res.dbscan_silhouette.items()},
        "hit_tiers": {k_: int(v) for k_, v in tier_counts.items()},
    }
    if make_plots:
        plots.save(plots.plot_elbow_silhouette(res.kmeans.table, res.kmeans.k_elbow,
                                               res.kmeans.k_silhouette, res.k_used),
                   fig_dir / "04_elbow_silhouette.png")
        plots.save(plots.plot_k_distance(res.kdist, res.eps, cfg.dbscan_min_samples),
                   fig_dir / "05_k_distance.png")
        plots.save(plots.plot_cluster_maps(labelled, res.k_used), fig_dir / "06_cluster_maps.png")
        plots.save(plots.plot_hit_tiers(labelled), fig_dir / "07_hit_tiers.png")

    # -- Step 5: trends -----------------------------------------------------------
    logger.info("Step 5/5: trends")
    imdb_tr = trends.imdb_genre_trends(movies, cfg)
    _write(imdb_tr, paths, "imdb_genre_trends")
    if make_plots:
        plots.save(plots.plot_imdb_genre_trends(imdb_tr), fig_dir / "08_imdb_genre_trends.png")

    ml = movielens.load_movielens(paths, refresh=refresh)
    summary["movielens_used"] = ml is not None
    if ml is not None:
        ml_tr = trends.movielens_genre_trends(ml, cfg, cfg.movielens_min_ratings_per_year)
        yearly = trends.movielens_yearly_summary(ml, cfg.movielens_min_ratings_per_year)
        spread = trends.rating_spread_by_cluster(ml, labelled, "cluster_kmeans",
                                                 cfg.movielens_min_ratings_for_std)
        _write(ml_tr, paths, "movielens_genre_trends")
        _write(yearly, paths, "movielens_yearly_summary")
        _write(spread, paths, "movielens_rating_spread")
        summary["divisiveness_median_std"] = (
            spread.groupby("cluster_kmeans")["rating_std"].median().round(3).to_dict())
        if make_plots:
            plots.save(plots.plot_movielens_genre_trends(ml_tr), fig_dir / "09_movielens_genre_trends.png")
            plots.save(plots.plot_movielens_yearly(yearly), fig_dir / "10_movielens_yearly.png")
            if len(spread):
                plots.save(plots.plot_divisiveness(spread, "cluster_kmeans"),
                           fig_dir / "11_divisiveness_check.png")

    with open(paths.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Done. Outputs in %s", paths.output_dir.resolve())
    return summary
