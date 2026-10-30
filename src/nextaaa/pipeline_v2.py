"""Run all of Part A v2 in order and save every table and figure.

    from nextaaa.config import Paths, AnalysisConfig
    from nextaaa.pipeline_v2 import run_part_a_v2
    summary = run_part_a_v2(Paths.resolve(), AnalysisConfig())

Outputs go to outputs/v2/ (v1 outputs in outputs/ are left untouched):
    figures/01_... to 19_...   numbered in deck order
    tables/*.csv               every number behind every chart
    summary.json               headline numbers
The labelled films are also saved as data/cache/hit_labels_v2.pkl for Part B.

Degrades gracefully:
    no TMDB file   -> audience stage only (no money, no AAA verdicts), clearly logged
    no CPI file    -> nominal dollars on slides; clustering is unaffected
    no MovieLens   -> MovieLens trend charts skipped
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import pandas as pd

from . import ensemble, finance, hits, io_imdb, io_tmdb, movielens, plots, plots_v2, popularity, ratings, trends
from .config import AnalysisConfig, Paths

logger = logging.getLogger(__name__)


def v2_paths(paths: Paths) -> Paths:
    """Same inputs, outputs in a v2 subfolder."""
    return replace(paths, output_dir=paths.output_dir / "v2")


def run_part_a_v2(paths: Paths, cfg: AnalysisConfig, k_audience: int | None = None,
                  k_commercial: int | None = None, refresh: bool = False,
                  make_plots: bool = True) -> dict:
    out_paths = v2_paths(paths)
    out_paths.make_output_dirs()
    fig_dir, tab_dir = out_paths.figures_dir, out_paths.tables_dir

    def save_fig(fig, name):
        if make_plots:
            plots.save(fig, fig_dir / f"{name}.png")
        else:
            plt.close(fig)

    def save_tab(df, name, index=False):
        if df is not None:
            df.to_csv(tab_dir / f"{name}.csv", index=index)

    summary: dict = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}}

    # -- 1. Data ---------------------------------------------------------------------
    logger.info("Step 1/6: loading IMDb, TMDB and CPI")
    movies = io_imdb.load_movies(paths, cfg, refresh=refresh)
    tmdb, tmdb_report = io_tmdb.load_tmdb(paths, cfg, set(movies["tconst"]), refresh=refresh)
    cpi = finance.load_cpi(paths)
    summary["tmdb"] = tmdb_report
    summary["cpi_available"] = cpi is not None

    # -- 2. 2023: highest rated film, most popular actor ------------------------------
    logger.info("Step 2/6: the %d questions", cfg.target_year)
    year = cfg.target_year
    board, info = ratings.top_rated(movies, year, cfg.shrinkage_quantile, cfg.min_votes_target_year, top_n=25)
    if tmdb is not None:  # add money columns to the leaderboard for context
        board = board.merge(tmdb[["tconst", "budget", "revenue"]], on="tconst", how="left")
        board["roi_multiple"] = board["revenue"] / board["budget"]
    sens = ratings.sensitivity_to_m(movies, year, cfg.sensitivity_quantiles, cfg.min_votes_target_year)
    save_tab(board, f"top_rated_{year}")
    save_tab(sens, f"m_sensitivity_{year}")
    summary["top_rated"] = {**info, "winner": board.iloc[0]["primaryTitle"]}
    save_fig(plots.plot_rating_vs_votes(movies, info, board), f"01_rating_vs_votes_{year}")
    save_fig(plots.plot_m_sensitivity(sens, cfg.shrinkage_quantile), f"02_m_sensitivity_{year}")

    credits = io_imdb.load_actor_credits(paths, cfg, set(movies["tconst"]), refresh=refresh)
    year_ids = set(movies.loc[movies["startYear"] == year, "tconst"])
    names = io_imdb.load_names(paths, set(credits.loc[credits["tconst"].isin(year_ids), "nconst"]),
                               refresh=refresh)
    pop = popularity.actor_popularity(movies, credits, names, year, cfg.shrinkage_quantile,
                                      cfg.billing_weight_power, cfg.min_votes_target_year)
    box = None
    if tmdb is not None:
        films_money = movies.merge(tmdb[["tconst", "revenue"]], on="tconst", how="left")
        box = popularity.actor_box_office(films_money, credits, names, year, cfg.billing_weight_power)
        save_tab(box, f"actor_box_office_{year}")
    save_tab(pop, f"actor_popularity_{year}")
    save_tab(popularity.three_way(pop, box, cfg.top_n_actors), f"actors_three_way_{year}", index=True)
    summary["actors"] = {
        "top_by_reach": pop.nsmallest(1, "rank_reach")["primaryName"].iloc[0],
        "top_by_goodwill": pop.nsmallest(1, "rank_goodwill")["primaryName"].iloc[0],
        "top_by_box_office": None if box is None or box.empty else box["primaryName"].iloc[0],
    }
    save_fig(plots_v2.plot_actors_three_way(pop, box, year), f"03_actors_{year}")

    # -- 3. Features --------------------------------------------------------------------
    logger.info("Step 3/6: year-relative features")
    frame = hits.build_v2_frame(movies, tmdb, cfg, cpi)
    has_money = bool(frame["has_finance"].sum() >= max(cfg.min_ref_films_finance, 50))
    if not has_money:
        logger.warning("Too few films with budget and revenue: running the audience stage only.")

    # -- 4. The two stages ----------------------------------------------------------------
    logger.info("Step 4/6: audience stage (5 methods)")
    aud = ensemble.run_stage(frame, ensemble.AUDIENCE, cfg, k=k_audience)
    com = None
    if has_money:
        logger.info("Step 4/6: commercial stage (5 methods)")
        com = ensemble.run_stage(frame[frame["has_finance"]], ensemble.COMMERCIAL, cfg, k=k_commercial)
    labelled = hits.combine_stages(frame, aud.frame, com.frame if com else None, cfg)

    for r in (aud, com):
        if r is None:
            continue
        key = r.stage.key
        save_tab(r.kmeans.table, f"{key}_kmeans_diagnostics")
        save_tab(r.method_summary, f"{key}_method_summary")
        save_tab(r.agreement, f"{key}_agreement_kappa", index=True)
        save_tab(r.vote_counts.rename_axis("votes").reset_index(name="films"), f"{key}_vote_counts")
    num = {"aud": ("06", "07", "08"), "com": ("09", "10", "11")}
    for r in (aud, com):
        if r is not None:
            a, b, c = num[r.stage.key]
            save_fig(plots_v2.plot_stage_k(r), f"{a}_{r.stage.key}_elbow_silhouette")
            save_fig(plots_v2.plot_stage_kdist(r, cfg.dbscan_min_samples), f"{b}_{r.stage.key}_k_distance")
            save_fig(plots_v2.plot_method_maps(r, cfg.hit_min_votes, cfg.mega_min_votes),
                     f"{c}_{r.stage.key}_method_maps")
    save_fig(plots_v2.plot_agreement(aud, com), "12_method_agreement")

    # -- 5. Verdicts, by year, all time ------------------------------------------------------
    logger.info("Step 5/6: verdicts, by-year and all-time hits")
    keep = [c for c in labelled.columns if c not in ("overview", "tagline", "keywords")]
    save_tab(labelled[keep], "hit_labels_v2")
    labelled.to_pickle(paths.cache_dir / "hit_labels_v2.pkl")  # handoff to Part B (includes plot text)

    summary["audience_hits"] = int(labelled["aud_hit"].sum())
    summary["verdicts"] = labelled["verdict"].value_counts().to_dict()
    summary["dollar_basis"] = frame.attrs.get("dollar_basis", "nominal US$")
    yearly = hits.yearly_summary(labelled)
    save_tab(yearly, "yearly_summary")
    save_tab(hits.audience_all_time(labelled, cfg.top_n_all_time), "all_time_audience_hits")

    if has_money:
        grid = hits.verdict_grid(labelled)
        tiers = finance.budget_tier_economics(labelled, "verdict")
        by_year = hits.hits_by_year(labelled, cfg.top_n_per_year)
        all_time = hits.all_time_hits(labelled, cfg.top_n_all_time)
        real_rev = hits.all_time_by_real_revenue(labelled, cfg.top_n_all_time)
        since = cfg.max_year - cfg.genre_recent_years + 1
        genres = hits.genre_economics(labelled, since)
        for df, name, idx in ((grid, "verdict_grid", True), (tiers, "budget_tiers", True),
                              (by_year, "aaa_hits_by_year", False), (all_time, "aaa_hits_all_time", False),
                              (real_rev, "all_time_by_real_revenue", False), (genres, "genre_economics", True)):
            save_tab(df, name, index=idx)
        summary["aaa_hits"] = int(labelled["is_aaa_hit"].sum())
        summary["aaa_mega_hits"] = int((labelled["aaa_tier"] == "AAA mega-hit").sum())
        summary["all_time_top5"] = all_time.head(5)[["startYear", "primaryTitle", "hit_score"]].to_dict("records")
        summary["verdict_grid"] = grid.to_dict()

        save_fig(plots_v2.plot_budget_vs_revenue(labelled, cfg.breakeven_multiple), "04_budget_vs_revenue")
        save_fig(plots_v2.plot_budget_tiers(tiers), "05_budget_tiers")
        save_fig(plots_v2.plot_verdict_map(labelled), "13_verdict_map")
        save_fig(plots_v2.plot_hits_per_year(yearly), "14_hit_rates_by_year")
        if len(all_time):
            save_fig(plots_v2.plot_all_time_heatmap(all_time), "15_all_time_aaa_hits")
        if len(genres):
            save_fig(plots_v2.plot_genre_economics(genres, cfg.breakeven_multiple, since), "16_genre_economics")

    # -- 6. Trends (as in v1) ------------------------------------------------------------------
    logger.info("Step 6/6: trends")
    imdb_tr = trends.imdb_genre_trends(movies, cfg)
    save_tab(imdb_tr, "imdb_genre_trends")
    save_fig(plots.plot_imdb_genre_trends(imdb_tr), "17_imdb_genre_trends")
    ml = movielens.load_movielens(paths, refresh=refresh)
    summary["movielens_used"] = ml is not None
    if ml is not None:
        yearly_ml = trends.movielens_yearly_summary(ml, cfg.movielens_min_ratings_per_year)
        save_tab(yearly_ml, "movielens_yearly_summary")
        save_fig(plots.plot_movielens_yearly(yearly_ml), "18_movielens_yearly")
        spread = trends.rating_spread_by_cluster(ml, labelled, "aud_kmeans", cfg.movielens_min_ratings_for_std)
        save_tab(spread, "movielens_rating_spread")
        if len(spread):
            save_fig(plots.plot_divisiveness(spread, "aud_kmeans"), "19_divisiveness_check")

    with open(out_paths.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Done. Outputs in %s", out_paths.output_dir.resolve())
    return summary
