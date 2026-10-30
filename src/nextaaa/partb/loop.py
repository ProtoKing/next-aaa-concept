"""Generate -> score -> refine: the loop that designs the next AAA title.

    for each island (one lead genre each, for a diverse portfolio):
        round 1:      GENERATE prompt                    -> N concepts
        rounds 2..R:  REFINE prompt with the scored history (OPRO) -> N new concepts
        every concept: validate -> novelty filter -> features -> judge -> drivers

Guardrails against "gaming" a learned judge (Goodhart's law)
    * Novelty filter: a concept whose plot embedding is closer to an existing film
      than `novelty_percentile` of real film-to-nearest-film similarities is rejected
      (and reported back to the generator). The threshold is relative, so it works
      for any embedding model. Repeats of earlier concepts are rejected the same way.
    * Familiar territory: concepts unlike any past film (ood_pct above
      `ood_percentile`) are flagged; their objective is halved and they cannot enter
      the final portfolio, because the judge is extrapolating there.
    * Calibrated probabilities with a bootstrap band, never a bare score.

Control arm (ablation)
    The SAME number of generator calls, with the first-round prompt every time and
    no score feedback. If the feedback loop does not beat this, the loop adds nothing,
    and the deck should say so.

References
    Yang et al. (2024). Large Language Models as Optimizers. ICLR.          (OPRO)
    Romera-Paredes et al. (2024). Mathematical discoveries from program search
        with large language models. Nature 625.                             (FunSearch: islands,
                                                                             LLM + evaluator)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import time

from ..config import PartBConfig
from . import prompts
from .dataset import FeatureSpace, concept_features
from .scorer import Judge
from .talent import concept_track_features
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
logger = logging.getLogger(__name__)

def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"

class _Progress:
    """Thread-safe 'rounds done / total' counter with elapsed time and a rough ETA."""
    def __init__(self, total: int):
        self.total, self.done, self.t0 = total, 0, time.monotonic()
        self.lock = threading.Lock()

    def tick(self, msg: str) -> None:
        with self.lock:
            self.done += 1
            elapsed = time.monotonic() - self.t0
            left = elapsed / self.done * (self.total - self.done)
            logger.info("[%d/%d rounds | %s elapsed | ~%s left] %s",
                        self.done, self.total, _fmt(elapsed), _fmt(left), msg)

def nn_similarity_threshold(emb: np.ndarray, percentile: float, chunk: int = 2048) -> float:
    """Percentile of each film's cosine similarity to its nearest OTHER film.

    Exact duplicate plots (similarity >= 0.999, e.g. a film listed twice or a
    stock one-line summary) are ignored, otherwise they would push the
    threshold to 1.0 and switch the novelty filter off.
    """
    best = np.empty(len(emb))
    for i in range(0, len(emb), chunk):
        sims = emb[i:i + chunk] @ emb.T
        idx = np.arange(i, min(i + chunk, len(emb)))
        sims[np.arange(len(idx)), idx] = -np.inf  # ignore self
        sims[sims >= 0.999] = -np.inf             # ignore exact duplicates
        best[idx] = sims.max(axis=1)
    best = best[np.isfinite(best)]
    return float(np.percentile(best, percentile)) if len(best) else 0.95


@dataclass
class LoopContext:
    generator: object
    embeddings: object
    judge: Judge
    space: FeatureSpace
    directors: pd.DataFrame          # shortlist, with id
    actors: pd.DataFrame
    film_emb: np.ndarray             # existing films' plot embeddings (normalised)
    film_titles: np.ndarray
    novelty_threshold: float
    evidence: str
    constraints: str
    allowed_genres: list[str]
    bcfg: PartBConfig


def _drivers_text(drivers: list[tuple[str, float]]) -> str:
    """e.g. "+ director track record (+2.1 pts), - budget (-0.4 pts)"."""
    return ", ".join(f"{'+' if v >= 0 else '-'} {g} ({100 * v:+.1f} pts)" for g, v in drivers) or "n/a"


def score_batch(raw: list, ctx: LoopContext, seen_emb: list[np.ndarray], meta: dict) -> list[dict]:
    """Validate, filter and score one generator reply. Returns one record per concept."""
    bcfg, records, valid = ctx.bcfg, [], []
    dir_ids, act_ids = set(ctx.directors["id"]), set(ctx.actors["id"])
    for c in raw if isinstance(raw, list) else []:
        clean, problems = prompts.validate_concept(c, ctx.allowed_genres, dir_ids, act_ids, bcfg)
        if clean is None:
            records.append({**meta, "title": str(c.get("title", "")) if isinstance(c, dict) else "",
                            "status": "invalid", "problems": "; ".join(problems)})
            continue
        valid.append((clean, problems))
    if not valid:
        return records

    emb = ctx.embeddings.embed([prompts.concept_text(c) for c, _ in valid])
    sims = emb @ ctx.film_emb.T
    nearest = sims.argmax(axis=1)
    scored, scored_emb = [], []
    for j, (c, problems) in enumerate(valid):
        rec = {**meta, **c, "problems": "; ".join(problems),
               "nearest_film": str(ctx.film_titles[nearest[j]]), "nearest_sim": float(sims[j, nearest[j]])}
        rep = max((float(emb[j] @ e) for e in seen_emb + scored_emb), default=-1.0)
        if not np.linalg.norm(emb[j]) > 0:  # e.g. local TF-IDF knows none of its words
            rec.update(status="invalid", problems="plot text could not be embedded; originality unknown",
                       nearest_sim=np.nan, nearest_film="")
        elif rec["nearest_sim"] > ctx.novelty_threshold:
            rec["status"] = "rejected_similar"
        elif rep > ctx.novelty_threshold:
            rec.update(status="rejected_repeat", nearest_film="(an earlier concept)")
        else:
            rec["status"] = "scored"
            scored.append(rec)
            scored_emb.append(emb[j])
        records.append(rec)
    seen_emb.extend(scored_emb)
    if not scored:
        return records

    d = ctx.directors.set_index("id")
    a = ctx.actors.set_index("id")
    tracks = [concept_track_features(d.loc[r["director_id"]], a.loc[r["lead_ids"]]) for r in scored]
    text_emb = np.vstack(scored_emb)
    X = concept_features(scored, tracks, text_emb, ctx.space)
    pred = ctx.judge.predict(X)
    drivers = ctx.judge.drivers(X)
    for i, r in enumerate(scored):
        r.update({k: float(pred.iloc[i][k]) for k in ("p_aaa_hit", "p_lo", "p_hi", "ood_pct")})
        r["in_distribution"] = bool(r["ood_pct"] <= bcfg.ood_percentile)
        r["objective"] = r["p_aaa_hit"] * (1.0 if r["in_distribution"] else 0.5)
        r["drivers"] = drivers[i]
        r["drivers_text"] = _drivers_text(drivers[i])
        r["director_name"] = d.loc[r["director_id"], "primaryName"]
        r["lead_names"] = list(a.loc[r["lead_ids"], "primaryName"])
    return records


def lead_genres(genre_econ: pd.DataFrame, n: int) -> list[str]:
    """One lead genre per island: the genres with the highest recent AAA-hit rate."""
    g = [x for x in genre_econ.index if x not in ("Unknown",)]
    return (g * n)[:n] if g else ["Drama"] * n


class _LockedEmbeddings:
    """Serialises embedding calls (fast anyway) so threads don't clash on the disk cache."""
    def __init__(self, inner):
        self.inner, self.lock = inner, threading.Lock()

    def embed(self, texts):
        with self.lock:
            return self.inner.embed(texts)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _run_island(ctx: LoopContext, arm: str, i: int, genre: str, progress: _Progress) -> list[dict]:
    """One island of one arm: rounds run in order, since each refine prompt needs the last scores."""
    bcfg, out = ctx.bcfg, []
    history, rejected, seen = [], [], []
    tag = f"{arm} island {i + 1} ({genre})"
    for rnd in range(1, bcfg.n_rounds + 1):
        if arm == "loop" and history:
            top = sorted(history, key=lambda r: r["objective"])[-bcfg.history_top_k:]
            msgs = prompts.refine_prompt(ctx.evidence, ctx.constraints, bcfg.concepts_per_call, genre,
                                         top, rejected[-8:])
        else:
            msgs = prompts.generate_prompt(ctx.evidence, ctx.constraints, bcfg.concepts_per_call, genre)
        meta = {"arm": arm, "island": i + 1, "lead_genre": genre, "round": rnd}

        logger.info("%s round %d: asking the generator...", tag, rnd)
        t = time.monotonic()
        try:
            reply = ctx.generator.complete_json(msgs, temperature=bcfg.generator_temperature,
                                                purpose=f"generate:{arm}:island{i + 1}:round{rnd}")
        except Exception as exc:  # one bad call should not end the run
            progress.tick(f"{tag} round {rnd}: generator failed ({exc}); skipped")
            continue
        secs = time.monotonic() - t

        try:
            recs = score_batch(reply.get("concepts", []), ctx, seen, meta)
        except RuntimeError as exc:
            progress.tick(f"{tag} round {rnd}: scoring failed ({exc}); skipped")
            continue
        out.extend(recs)
        history += [r for r in recs if r.get("status") == "scored"]
        rejected += [r for r in recs if r.get("status") == "rejected_similar"]
        best = max((r["objective"] for r in history), default=float("nan"))
        n_ok = sum(r.get("status") == "scored" for r in recs)
        n_rej = sum(r.get("status", "").startswith("rejected") for r in recs)
        progress.tick(f"{tag} round {rnd}: {n_ok} scored, {n_rej} rejected, "
                      f"best so far {best:.3f} (call took {secs:.0f}s)")
    return out


def run_arms(ctx: LoopContext, arms: list[str], islands: list[str]) -> list[dict]:
    """Every (arm, island) search is independent, so they run in parallel."""
    if isinstance(arms, str):
        arms = [arms]
    ctx = replace(ctx, embeddings=_LockedEmbeddings(ctx.embeddings))
    jobs = [(arm, i, g) for arm in arms for i, g in enumerate(islands)]
    workers = max(1, min(len(jobs), getattr(ctx.bcfg, "max_parallel_calls", 6)))
    progress = _Progress(len(jobs) * ctx.bcfg.n_rounds)
    logger.info("Starting %d searches (%s x %d islands), %d generator calls, %d at a time.",
                len(jobs), "+".join(arms), len(islands), progress.total, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: _run_island(ctx, *job, progress), jobs))
    logger.info("Done: %d rounds in %s.", progress.total, _fmt(time.monotonic() - progress.t0))
    return [r for res in results for r in res]


def run_arm(ctx: LoopContext, arm: str, islands: list[str]) -> list[dict]:
    return run_arms(ctx, [arm], islands)


def progress(records: pd.DataFrame) -> pd.DataFrame:
    """Best in-distribution score so far, per arm and round (mean over islands)."""
    ok = records[(records["status"] == "scored") & records["in_distribution"].fillna(False).astype(bool)]
    if ok.empty:
        return pd.DataFrame(columns=["arm", "round", "best_so_far"])
    rounds = sorted(records["round"].dropna().unique())
    best = ok.groupby(["arm", "island", "round"])["p_aaa_hit"].max()
    out = []
    for (arm, island), s in best.groupby(level=["arm", "island"]):
        # carry the best so far through rounds where nothing new qualified
        s = s.droplevel([0, 1]).reindex(rounds).cummax().ffill()
        out.append(pd.DataFrame({"arm": arm, "island": island, "round": rounds, "best": s.to_numpy()}))
    df = pd.concat(out)
    return (df.groupby(["arm", "round"])["best"].mean().rename("best_so_far").reset_index()
            .dropna())


def portfolio(records: pd.DataFrame, k: int) -> pd.DataFrame:
    """The final pitch: the best familiar, novel concept from each island (loop arm)."""
    ok = records[(records["arm"] == "loop") & (records["status"] == "scored")
                 & records["in_distribution"].fillna(False).astype(bool)]
    if ok.empty:
        return ok
    best = ok.sort_values("p_aaa_hit", ascending=False).drop_duplicates("island")
    rest = ok.drop(best.index).sort_values("p_aaa_hit", ascending=False)
    return pd.concat([best, rest]).head(k).reset_index(drop=True)
