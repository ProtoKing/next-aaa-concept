"""Run all of Part B and save every output to outputs/partb/.

    from nextaaa.config import Paths, AnalysisConfig, PartBConfig
    from nextaaa.partb.pipeline import run_part_b
    summary = run_part_b(Paths.resolve(), AnalysisConfig(), PartBConfig())

Needs Part A v2's labels (data/cache/hit_labels_v2.pkl, with TMDB money data).
If they are missing, Part A v2 is run first.

Outputs
    pitch_cards.md            the final portfolio, ready for the deck
    final_portfolio.json      the same, machine-readable
    tables/                   backtest, model selection, feature importance, every concept,
                              loop progress, talent shortlists, LLM baseline
    figures/01_... 06_...     backtest PR curves, calibration, importance, loop progress,
                              concept map, LLM vs judge
    llm_log.jsonl             every prompt and reply (audit trail; no keys)
    summary.json              headline numbers, models used, token usage
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import hits, plots as plots_a
from ..config import PROJECT_ROOT, AnalysisConfig, PartBConfig, Paths
from . import dataset, llm, llm_baseline, loop, plots, prompts, scorer, talent

logger = logging.getLogger(__name__)


def load_labels(paths: Paths, cfg: AnalysisConfig) -> pd.DataFrame:
    path = paths.cache_dir / "hit_labels_v2.pkl"
    if not path.exists():
        logger.info("Part A v2 labels not found: running Part A v2 first (tables only).")
        from ..pipeline_v2 import run_part_a_v2
        run_part_a_v2(paths, cfg, make_plots=False)
    labels = pd.read_pickle(path)
    if labels["has_finance"].sum() < 50 or labels["is_aaa_hit"].sum() < 10:
        raise ValueError("Part B needs TMDB budget/revenue data and AAA-hit labels from Part A v2. "
                         "Put the TMDB file in data/tmdb/ and re-run scripts/run_part_a_v2.py.")
    return labels


def _record(p: pd.Series) -> str:
    hits_, films = int(p["prior_aaa_hits"]), int(p["prior_films"])
    return f"{hits_} AAA hit{'' if hits_ == 1 else 's'} in {films} film{'' if films == 1 else 's'}"


def pitch_card(i: int, r: pd.Series, directors: pd.DataFrame, actors: pd.DataFrame) -> str:
    d = directors.set_index("id").loc[r["director_id"]]
    leads = actors.set_index("id").loc[r["lead_ids"]]
    lead_txt = "; ".join(f"{a['primaryName']} ({_record(a)})" for _, a in leads.iterrows())
    drivers = "\n".join(f"  - {'raises' if v >= 0 else 'lowers'} the score: {g} ({100 * v:+.1f} percentage points)"
                        for g, v in r["drivers"])
    return (f"## {i}. {r['title']}\n\n"
            f"**{r['logline']}**\n\n{r['synopsis']}\n\n"
            f"| | |\n|---|---|\n"
            f"| Genres | {', '.join(r['genres'])} |\n"
            f"| Budget / runtime | ${r['budget_usd'] / 1e6:,.0f}M / {r['runtime_min']} min |\n"
            f"| Director | {d['primaryName']} ({_record(d)}) |\n"
            f"| Leads | {lead_txt} |\n"
            f"| Pitched against | {', '.join(r['comparables'])} |\n"
            f"| Why now | {r['why_now']} |\n"
            f"| **P(AAA hit)** | **{prompts._pct(r['p_aaa_hit'])}** (bootstrap range {prompts._pct(r['p_lo'])} "
            f"to {prompts._pct(r['p_hi'])}) |\n"
            f"| Closest existing film | {r['nearest_film']} (similarity {r['nearest_sim']:.2f}) |\n"
            f"| Familiarity | more unusual than {r['ood_pct']:.0f}% of past films |\n\n"
            f"What drives the score:\n{drivers}\n")


def run_part_b(paths: Paths, cfg: AnalysisConfig, bcfg: PartBConfig, mode: str | None = None,
               refresh: bool = False, make_plots: bool = True) -> dict:
    out = paths.output_dir / "partb"
    fig_dir, tab_dir = out / "figures", out / "tables"
    for d in (fig_dir, tab_dir, paths.cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    log_path = out / "llm_log.jsonl"
    log_path.unlink(missing_ok=True)

    def save_fig(fig, name):
        if make_plots:
            plots_a.save(fig, fig_dir / f"{name}.png")
        else:
            plt.close(fig)

    clients = llm.get_clients(PROJECT_ROOT, paths.cache_dir, log_path, mode)
    summary: dict = {"clients": clients.describe(),
                     "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(bcfg).items()}}

    # -- 1. Labels, talent, track records ------------------------------------------------
    logger.info("Step 1/6: labels and talent track records")
    labels = load_labels(paths, cfg)
    credits = talent.load_credits(paths, cfg, set(labels["tconst"]), ("director", "actor", "actress"),
                                  refresh=refresh)
    people = talent.load_people(paths, set(credits["nconst"]), refresh=refresh)
    pf = talent.person_film_table(labels, credits, bcfg.leads_per_film)
    records = talent.asof_records(pf, labels, bcfg.track_record_prior_strength)
    track = talent.film_track_features(pf, records)
    now = cfg.max_year
    directors = talent.shortlist(pf, labels, people, "director", now, bcfg, bcfg.shortlist_directors, "D")
    actors = talent.shortlist(pf, labels, people, "lead", now, bcfg, bcfg.shortlist_actors, "A")
    directors.to_csv(tab_dir / "shortlist_directors.csv", index=False)
    actors.to_csv(tab_dir / "shortlist_actors.csv", index=False)
    if len(directors) < 3 or len(actors) < 4:
        raise ValueError("Talent shortlists are too small; relax shortlist_* settings in PartBConfig.")

    # -- 2. Plot embeddings and features --------------------------------------------------
    logger.info("Step 2/6: plot embeddings (%s) and features", clients.embeddings.name)
    has_text = labels["overview"].fillna("").str.split().str.len().fillna(0).ge(5).to_numpy() \
        if "overview" in labels else np.zeros(len(labels), bool)
    texts = labels.loc[has_text, "overview"].tolist()
    if getattr(clients.embeddings, "is_local", False):
        clients.embeddings.fit(texts)
    emb_text = clients.embeddings.embed(texts)
    emb = np.full((len(labels), emb_text.shape[1]), np.nan, dtype=np.float32)
    emb[has_text] = emb_text
    space = dataset.build_feature_space(labels, emb_text, np.ones(len(emb_text), bool),
                                        bcfg.text_pca_components, bcfg.random_state)
    fin_mask = labels["has_finance"].to_numpy()
    fin = labels[fin_mask].reset_index(drop=True)
    X = dataset.film_features(fin, track, emb[fin_mask], space)
    y = fin["is_aaa_hit"].astype(int)
    years = fin["startYear"]
    X.to_csv(tab_dir / "features_films.csv")
    summary["training_films"] = {"films": len(X), "aaa_hits": int(y.sum()), "features": X.shape[1]}

    # -- 3. The judge: backtest, then refit -------------------------------------------------
    logger.info("Step 3/6: backtest and judge")
    groups = space.groups()
    bt = scorer.run_backtest(X.reset_index(drop=True), y, years, now, bcfg, groups)
    bt.selection.to_csv(tab_dir / "model_selection_validation.csv", index=False)
    bt.test.to_csv(tab_dir / "backtest_test_metrics.csv", index=False)
    bt.predictions.assign(tconst=fin["tconst"].to_numpy()[bt.predictions["tconst"].to_numpy()]) \
        .merge(fin[["tconst", "primaryTitle"]], on="tconst").to_csv(tab_dir / "backtest_predictions.csv", index=False)
    bt.group_importance.to_csv(tab_dir / "feature_group_importance.csv", index=False)
    aaa_scale = (X["log_budget"] >= np.log10(bcfg.min_budget_usd)).to_numpy()
    judge = scorer.fit_judge(X, y, bt.chosen, bcfg, groups, ref_mask=aaa_scale)
    summary["backtest"] = {"years": bt.years, "chosen": bt.chosen,
                           "test": bt.test.round(4).to_dict("records")}
    save_fig(plots.plot_backtest(bt), "01_backtest_pr_curves")
    save_fig(plots.plot_calibration(bt), "02_calibration")
    save_fig(plots.plot_group_importance(bt.group_importance), "03_what_the_judge_uses")

    # -- 4. The loop --------------------------------------------------------------------------
    logger.info("Step 4/6: generate -> score -> refine (%s)", clients.generator.name)
    since = now - cfg.genre_recent_years + 1
    genre_econ = hits.genre_economics(labels, since)
    evidence = prompts.evidence_block(prompts.genre_table(genre_econ),
                                      prompts.recent_hits_table(labels, now - 9),
                                      prompts.talent_table(directors), prompts.talent_table(actors), since)
    allowed = space.genres
    film_emb, film_titles = emb_text, labels.loc[has_text, "primaryTitle"].to_numpy()
    threshold = loop.nn_similarity_threshold(film_emb, bcfg.novelty_percentile)
    ctx = loop.LoopContext(generator=clients.generator, embeddings=clients.embeddings, judge=judge,
                           space=space, directors=directors, actors=actors, film_emb=film_emb,
                           film_titles=film_titles, novelty_threshold=threshold, evidence=evidence,
                           constraints=prompts.constraints_block(allowed, bcfg), allowed_genres=allowed,
                           bcfg=bcfg)
    islands = loop.lead_genres(genre_econ, bcfg.n_islands)
    recs = loop.run_arm(ctx, "loop", islands)
    if bcfg.run_ablation:
        recs += loop.run_arm(ctx, "no_feedback", islands)
    concepts = pd.DataFrame(recs)
    for col in ("in_distribution", "p_aaa_hit", "nearest_sim"):
        if col not in concepts:
            concepts[col] = np.nan
    concepts.to_csv(tab_dir / "concepts_all.csv", index=False)
    prog = loop.progress(concepts)
    prog.to_csv(tab_dir / "loop_progress.csv", index=False)
    final = loop.portfolio(concepts, bcfg.portfolio_size)
    summary["loop"] = {"islands": islands, "novelty_threshold": threshold,
                       "status_counts": concepts["status"].value_counts().to_dict(),
                       "progress": prog.round(4).to_dict("records")}
    save_fig(plots.plot_loop_progress(prog), "04_loop_progress")
    save_fig(plots.plot_concept_map(concepts, final, threshold), "05_concept_map")

    # -- 5. The pitch --------------------------------------------------------------------------
    logger.info("Step 5/6: the pitch")
    header = ("# The next AAA title: concept portfolio\n\n"
              f"Generator: `{clients.generator.name}` | judge: {scorer.MODEL_LABELS.get(bt.chosen, bt.chosen)} "
              f"(test PR-AUC {bt.test.set_index('model').loc[bt.chosen, 'pr_auc']:.2f}, "
              f"{bt.test.set_index('model').loc[bt.chosen, 'lift']:.1f}x the base rate) | "
              f"embeddings: `{clients.embeddings.name}`\n\n"
              "P(AAA hit) is a calibrated estimate of how often concepts with these pre-release features "
              "became AAA hits (loved by audiences AND at least 2.5x budget at the box office) in the past. "
              "It cannot see script quality, marketing, competition or release timing.\n\n")
    if clients.mode == "mock":
        header += "> **MOCK MODE**: concepts are random placeholders from the offline stand-in. Configure an LLM.\n\n"
    cards = [pitch_card(i, r, directors, actors) for i, (_, r) in enumerate(final.iterrows(), start=1)]
    (out / "pitch_cards.md").write_text(header + "\n".join(cards), encoding="utf-8")
    final_json = final.drop(columns=["drivers"], errors="ignore").to_dict("records")
    (out / "final_portfolio.json").write_text(json.dumps(final_json, indent=2, default=str), encoding="utf-8")
    summary["portfolio"] = [{k: r.get(k) for k in ("title", "genres", "p_aaa_hit", "p_lo", "p_hi",
                                                     "director_name", "lead_names", "nearest_film")}
                            for r in final_json]

    # -- 6. LLM baseline --------------------------------------------------------------------------
    if bcfg.run_llm_baseline:
        logger.info("Step 6/6: LLM-as-predictor baseline (%s)", clients.judge.name)
        preds, table = llm_baseline.run_llm_baseline(clients.judge, fin, X, emb[fin_mask], has_text[fin_mask],
                                                     bt.chosen, now, bcfg)
        if len(table):
            preds.to_csv(tab_dir / "llm_baseline_predictions.csv", index=False)
            table.to_csv(tab_dir / "llm_baseline_metrics.csv", index=False)
            summary["llm_baseline"] = table.round(4).to_dict("records")
            save_fig(plots.plot_llm_vs_judge(preds, table), "06_llm_vs_judge")

    summary["usage"] = clients.usage()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Done. Pitch cards: %s", (out / "pitch_cards.md").resolve())
    return summary
