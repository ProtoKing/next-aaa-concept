"""Backtest: can an LLM, given few-shot examples, predict AAA hits before release?

This is the original "few-shot LLM" idea, run as a BASELINE that has to earn
trust against the trained judge, on a test it cannot have memorised:

    * Only films released AFTER the LLM's knowledge cutoff are tested
      (BASELINE_KNOWLEDGE_CUTOFF, e.g. "2024-05" for gpt-5-mini; we test
      from the following calendar year, to be safe).
    * Titles are hidden. Each prompt shows the most similar earlier films
      (by plot embedding, released before the cutoff year) with their
      pre-release facts and outcome, then the test film without its outcome.
    * Temperature 0; the reply is {"p_aaa_hit": 0-1, "reason": "..."}.
    * The judge is refit on the SAME pre-cutoff films and scored on the SAME
      test films, so the comparison is like for like.

Retrieval keeps 5 of the shots for the most similar past HITS so that the LLM
sees what a hit looks like (hits are rare). That shifts the base rate it
sees, so compare the two on ranking (PR-AUC) first, calibration second.

Reference: Predicting Movie Hits Before They Happen with LLMs (UMAP 2025),
which found LLMs given metadata beat simple baselines at ranking new films.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..config import PartBConfig
from .scorer import calibrated, metrics

logger = logging.getLogger(__name__)

PREDICT_SYSTEM = ("You estimate the chance that a feature film becomes a AAA hit: in the top tier of its "
                  "release year for BOTH audience reception and box-office return (at least 2.5x its "
                  "budget). Use only the information given. Titles are hidden on purpose. Reply with a "
                  "single JSON object and nothing else.")


def cutoff_year() -> int:
    raw = os.environ.get("BASELINE_KNOWLEDGE_CUTOFF")
    if not raw:
        logger.warning("BASELINE_KNOWLEDGE_CUTOFF not set: assuming 2024-05 (gpt-5-mini). "
                       "Set it to your baseline model's knowledge cutoff.")
        raw = "2024-05"
    return int(str(raw)[:4])


def _money(v) -> str:
    return "unknown" if pd.isna(v) else f"${v / 1e6:,.0f}M"


def describe(film: pd.Series, feats: pd.Series) -> str:
    plot = " ".join(str(film.get("overview") or "").split()[:45])
    src = ("sequel/franchise" if feats.get("src_sequel") == 1 else
           "adaptation" if feats.get("src_adaptation") == 1 else "original")
    d_hits, d_films = feats.get("dir_prior_aaa_hits"), feats.get("dir_prior_money_films")
    return (f"Genres: {film['genres']} | budget {_money(film['budget'])} | runtime {film['runtimeMinutes']} min | "
            f"{src} | director's earlier AAA hits: {0 if pd.isna(d_hits) else int(d_hits)} of "
            f"{0 if pd.isna(d_films) else int(d_films)} | leads' earlier AAA hits: "
            f"{0 if pd.isna(feats.get('lead_prior_aaa_hits')) else int(feats.get('lead_prior_aaa_hits'))} | "
            f"premise: {plot}")


def predict_prompt(shots: list[tuple[str, bool]], target: str, year: int) -> list[dict]:
    lines = "\n".join(f"{i}. {d} -> AAA hit: {'YES' if hit else 'NO'}" for i, (d, hit) in enumerate(shots, 1))
    user = (f"Here are {len(shots)} earlier films similar to the new one, with their outcomes:\n{lines}\n\n"
            f"New film (released {year}): {target}\n\n"
            'Return {"p_aaa_hit": a number from 0 to 1, "reason": "at most 30 words"}')
    return [{"role": "system", "content": PREDICT_SYSTEM}, {"role": "user", "content": user}]


def run_llm_baseline(llm, films: pd.DataFrame, X: pd.DataFrame, emb: np.ndarray, has_emb: np.ndarray,
                     judge_name: str, max_year: int, bcfg: PartBConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """films/X/emb aligned row by row (films with money data). Returns (predictions, metrics)."""
    cut = cutoff_year()
    year = films["startYear"].to_numpy()
    y = films["is_aaa_hit"].astype(int).to_numpy()
    pool = np.where((year < cut) & has_emb)[0]
    test = np.where((year > cut) & (year <= max_year) & has_emb)[0]
    if len(test) == 0 or y[test].sum() == 0 or y[pool].sum() < 5:
        logger.warning("LLM baseline skipped: no post-cutoff test films with hits (cutoff %d).", cut)
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(bcfg.random_state)
    hits, misses = test[y[test] == 1], test[y[test] == 0]
    n_miss = max(0, min(len(misses), bcfg.llm_baseline_max_films - len(hits)))
    test = np.concatenate([hits, rng.choice(misses, n_miss, replace=False)])
    pool_hits = pool[y[pool] == 1]
    n_hit_shots = min(5, bcfg.llm_baseline_shots // 4)

    def one(i: int) -> float:
        sims = emb[pool] @ emb[i]
        near = pool[np.argsort(-sims)[: bcfg.llm_baseline_shots - n_hit_shots]]
        near_hits = pool_hits[np.argsort(-(emb[pool_hits] @ emb[i]))[:n_hit_shots]]
        idx = list(dict.fromkeys(list(near_hits) + list(near)))
        rng_i = np.random.default_rng(i)
        rng_i.shuffle(idx)
        shots = [(describe(films.iloc[j], X.iloc[j]), bool(y[j])) for j in idx]
        msgs = predict_prompt(shots, describe(films.iloc[i], X.iloc[i]), int(year[i]))
        try:
            p = llm.complete_json(msgs, temperature=0.0, purpose="predict:backtest").get("p_aaa_hit")
            return float(np.clip(float(p), 0, 1))
        except Exception as exc:
            logger.warning("LLM baseline call failed (%s).", exc)
            return np.nan

    logger.info("LLM baseline: %d test films released after %d (%d hits), %d shots each.",
                len(test), cut, y[test].sum(), bcfg.llm_baseline_shots)
    if y[test].sum() < 5:
        logger.warning("Only %d AAA hits after the cutoff: the LLM-vs-judge comparison will be noise.",
                       y[test].sum())
    with ThreadPoolExecutor(max_workers=bcfg.llm_baseline_workers) as ex:
        p_llm = np.array(list(ex.map(one, test)))

    # The judge, trained on the same pre-cutoff films only.
    train = year < cut
    model = calibrated(judge_name, bcfg, int(y[train].sum())).fit(X[train], y[train])
    p_judge = model.predict_proba(X.iloc[test])[:, 1]

    preds = pd.DataFrame({"tconst": films["tconst"].to_numpy()[test], "year": year[test], "y": y[test],
                          "p_llm": p_llm, "p_judge": p_judge})
    ok = preds["p_llm"].notna()
    table = pd.DataFrame([
        {"model": f"LLM few-shot ({llm.name})", **metrics(preds.loc[ok, "y"], preds.loc[ok, "p_llm"])},
        {"model": f"Judge ({judge_name}), same films", **metrics(preds.loc[ok, "y"], preds.loc[ok, "p_judge"])},
    ])
    table.insert(1, "cutoff_year", cut)
    return preds, table
