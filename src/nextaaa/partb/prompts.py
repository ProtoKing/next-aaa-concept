"""The prompts, the concept schema, and its validation.

Three prompts (see the notebook for rendered examples):

1. GENERATE (first round of each island). The generator gets the evidence
   from Part A (genre economics, recent AAA hits, talent tables with track
   records), the constraints (original story, AAA budget, cast only from the
   tables by id) and a strict JSON schema whose fields map onto the judge's
   features, so every concept can be scored.

2. REFINE (later rounds; OPRO, Yang et al. 2024). The best concepts so far are
   listed from LOWEST to HIGHEST score, each with its uncertainty band and the
   "drivers" that moved its score, plus the concepts rejected for being too
   close to an existing film. The model is asked for new concepts that beat
   them all. Showing drivers (why each scored as it did) goes one step beyond
   OPRO, which shows only the score.

3. PREDICT (llm_baseline.py; backtest only). Lives with the baseline code.

The system prompt is shared by 1 and 2.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import PartBConfig

SYSTEM = (
    "You are a development executive at a major film studio, pitching ORIGINAL feature-film "
    "concepts for a AAA slate. Original means: not a sequel, prequel, remake, reboot or spin-off, "
    "and not an adaptation of an existing book, comic, game, TV show or other franchise. Never copy "
    "an existing film; you may draw on the genres, structures and themes audiences have rewarded. "
    "You only cast directors and lead actors from the tables you are given, by their id. "
    "Reply with a single JSON object that matches the schema, and nothing else."
)

SCHEMA = """{"concepts": [{
  "title": "string",
  "logline": "string, at most 35 words",
  "synopsis": "string, 80-150 words: setup, conflict, stakes, ending tone",
  "genres": ["1 to 3 genres from the allowed list"],
  "runtime_min": integer,
  "budget_usd": integer (production budget in US dollars),
  "director_id": "one id from the directors table, e.g. D04",
  "lead_ids": ["two different ids from the actors table, e.g. A07", "A12"],
  "comparables": ["two past films it would be pitched against"],
  "why_now": "one sentence: why audiences want this now"
}]}"""


# ---------------------------------------------------------------------------
# Evidence blocks
# ---------------------------------------------------------------------------

def _money(v) -> str:
    return "" if pd.isna(v) else f"${v / 1e6:,.0f}M"


def _words(text, n: int) -> str:
    if not isinstance(text, str):
        return ""
    w = text.split()
    return " ".join(w[:n]) + ("..." if len(w) > n else "")


def genre_table(genres: pd.DataFrame, top: int = 15) -> str:
    rows = ["genre | films | AAA-hit rate | median ROI | median budget"]
    for g, r in genres.head(top).iterrows():
        rows.append(f"{g} | {int(r['films'])} | {r['aaa_hit_pct']:.1f}% | {r['median_roi']:.1f}x | "
                    f"{_money(r['median_budget'])}")
    return "\n".join(rows)


def recent_hits_table(labels: pd.DataFrame, since: int, per_year: int = 3) -> str:
    hits = labels[labels["is_aaa_hit"] & (labels["startYear"] >= since)]
    hits = hits.sort_values(["startYear", "hit_score"], ascending=[False, False]).groupby("startYear").head(per_year)
    rows = ["year | title | genres | budget | ROI | premise"]
    for _, r in hits.iterrows():
        rows.append(f"{r['startYear']} | {r['primaryTitle']} | {r['genres']} | {_money(r['budget'])} | "
                    f"{r['roi_multiple']:.1f}x | {_words(r.get('overview'), 30)}")
    return "\n".join(rows)


def talent_table(short: pd.DataFrame) -> str:
    rows = ["id | name | films | AAA hits | AAA-hit rate | audience hits | known for | last film"]
    for _, r in short.iterrows():
        rows.append(f"{r['id']} | {r['primaryName']} | {int(r['prior_films'])} | {int(r['prior_aaa_hits'])} | "
                    f"{100 * r['prior_aaa_rate']:.0f}% | {int(r['prior_aud_hits'])} | {r['top_genres']} | "
                    f"{int(r['last_year'])}")
    return "\n".join(rows)


def evidence_block(genres_tbl: str, hits_tbl: str, dir_tbl: str, act_tbl: str, since: int) -> str:
    return (f"## Evidence from our analysis\n"
            f"### Genre economics, films released {since} onward (AAA hit = loved by audiences AND "
            f"commercially successful in its year)\n{genres_tbl}\n\n"
            f"### Recent AAA hits (top 3 per year)\n{hits_tbl}\n\n"
            f"### Directors you may choose from (track record up to now)\n{dir_tbl}\n\n"
            f"### Lead actors you may choose from\n{act_tbl}")


def constraints_block(allowed_genres: list[str], bcfg: PartBConfig) -> str:
    return (f"## Constraints\n"
            f"- ORIGINAL story only (no sequels, remakes, reboots, spin-offs or adaptations).\n"
            f"- Production budget between ${bcfg.min_budget_usd / 1e6:.0f}M and "
            f"${bcfg.max_budget_usd / 1e6:.0f}M (AAA scale), English-language.\n"
            f"- Runtime {bcfg.min_runtime}-{bcfg.max_runtime} minutes.\n"
            f"- Director and the two leads ONLY from the tables above, by id.\n"
            f"- Concepts in one reply must differ clearly in premise.\n"
            f"Allowed genres: {', '.join(allowed_genres)}")


# ---------------------------------------------------------------------------
# The two generator prompts
# ---------------------------------------------------------------------------

def generate_prompt(evidence: str, constraints: str, n: int, lead_genre: str) -> list[dict]:
    user = (f"## Goal\nPropose {n} distinct concepts for an original AAA feature film that would be both "
            f"an audience hit and a commercial hit.\nLead genre for this batch: {lead_genre} "
            f"(combine it with other genres if that makes a stronger film).\n\n"
            f"{evidence}\n\n{constraints}\n\n## Schema\n{SCHEMA}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _pct(p: float) -> str:
    """Probabilities as percentages with enough precision: hits are rare, so 0.004 -> '0.4%'."""
    return f"{100 * p:.1f}%" if p < 0.1 else f"{100 * p:.0f}%"


def _concept_line(r: dict) -> str:
    band = "" if pd.isna(r.get("p_lo")) else f" (range {_pct(r['p_lo'])}-{_pct(r['p_hi'])})"
    flag = "" if r.get("in_distribution", True) else "  [UNFAMILIAR: unlike any past film, score unreliable]"
    return (f"[{_pct(r['p_aaa_hit'])}{band}] \"{r['title']}\": {'/'.join(r['genres'])} | "
            f"{_money(r['budget_usd'])} | {r['runtime_min']} min | dir {r['director_id']} | "
            f"leads {', '.join(r['lead_ids'])}\n    drivers: {r['drivers_text']}{flag}")


def refine_prompt(evidence: str, constraints: str, n: int, lead_genre: str,
                  history: list[dict], rejected: list[dict]) -> list[dict]:
    lines = "\n".join(_concept_line(r) for r in sorted(history, key=lambda r: r["objective"]))
    rej = "\n".join(f"- \"{r['title']}\": too similar to the existing film \"{r['nearest_film']}\""
                    for r in rejected) or "- none"
    user = (f"## Goal\nPropose {n} NEW original AAA concepts that you expect to score higher than every "
            f"concept below.\nLead genre for this batch: {lead_genre}.\n\n"
            f"## Concepts so far, from LOWEST to HIGHEST estimated chance of becoming a AAA hit\n"
            f"(score = calibrated chance of becoming a AAA hit, from a model trained on past films; the base "
            f"rate for a film is a few percent. drivers = what raised (+) or lowered (-) each score most, in "
            f"percentage points, compared with an average film)\n{lines}\n\n"
            f"## Rejected for being too close to an existing film\n{rej}\n\n"
            f"Keep what the best concepts share; change what the worst share. Do not repeat or lightly "
            f"rename any concept above. Stay original.\n\n{evidence}\n\n{constraints}\n\n## Schema\n{SCHEMA}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _to_int(value) -> int | None:
    """Accept 150000000, "150000000", "$150M", "150 million", "1.2B"."""
    if isinstance(value, (int, float)) and np.isfinite(value):
        return int(value)
    if not isinstance(value, str):
        return None
    s = value.lower().replace(",", "").replace("$", "").strip()
    m = re.match(r"^([\d.]+)\s*(b|bn|billion|m|mn|million|k|thousand)?", s)
    if not m:
        return None
    mult = {"b": 1e9, "bn": 1e9, "billion": 1e9, "m": 1e6, "mn": 1e6, "million": 1e6,
            "k": 1e3, "thousand": 1e3}.get(m.group(2) or "", 1)
    return int(float(m.group(1)) * mult)


def validate_concept(c: dict, allowed_genres: list[str], director_ids: set[str], actor_ids: set[str],
                     bcfg: PartBConfig) -> tuple[dict | None, list[str]]:
    """Return (clean concept, problems). Hard problems -> None; soft ones are fixed and noted."""
    if not isinstance(c, dict):
        return None, ["not an object"]
    problems = []
    lookup = {g.lower(): g for g in allowed_genres}
    genres = [lookup[g.strip().lower()] for g in (c.get("genres") or []) if isinstance(g, str)
              and g.strip().lower() in lookup][:3]
    title = str(c.get("title") or "").strip()
    logline = str(c.get("logline") or "").strip()
    synopsis = str(c.get("synopsis") or "").strip()
    director = str(c.get("director_id") or "").strip().upper()
    leads = [str(a).strip().upper() for a in (c.get("lead_ids") or [])][:2]
    runtime, budget = _to_int(c.get("runtime_min")), _to_int(c.get("budget_usd"))

    if not title or not logline:
        problems.append("missing title or logline")
    if not genres:
        problems.append("no allowed genre")
    if director not in director_ids:
        problems.append(f"director id {director!r} not in the table")
    if len(leads) != 2 or len(set(leads)) != 2 or not set(leads) <= actor_ids:
        problems.append(f"lead ids {leads} invalid")
    if runtime is None or budget is None:
        problems.append("runtime or budget not a number")
    if problems:
        return None, problems

    if not bcfg.min_runtime <= runtime <= bcfg.max_runtime:
        problems.append(f"runtime {runtime} clipped")
        runtime = int(np.clip(runtime, bcfg.min_runtime, bcfg.max_runtime))
    if not bcfg.min_budget_usd <= budget <= bcfg.max_budget_usd:
        problems.append(f"budget {budget} clipped")
        budget = int(np.clip(budget, bcfg.min_budget_usd, bcfg.max_budget_usd))
    clean = {"title": title, "logline": logline, "synopsis": synopsis or logline, "genres": genres,
             "runtime_min": runtime, "budget_usd": budget, "director_id": director, "lead_ids": leads,
             "comparables": [str(x) for x in (c.get("comparables") or [])][:2],
             "why_now": str(c.get("why_now") or "")}
    return clean, problems


def concept_text(c: dict) -> str:
    """The text we embed for a concept (comparable to a TMDB overview)."""
    return f"{c['logline']} {c['synopsis']}"
