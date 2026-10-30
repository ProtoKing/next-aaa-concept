"""Pre-release features: what a studio knows BEFORE a film comes out.

The judge learns "which films became AAA hits" from features that exist at
greenlight time, and then scores new concepts with exactly the same features.
Anything known only after release (rating, votes, box office) is the label,
never a feature.

Features (one row per film or concept)
    g_<Genre>              multi-hot genres (IMDb's list)
    n_genres
    runtime                minutes
    log_budget             log10 production budget (base-year $ with CPI, else nominal)
    is_english             original language is English
    src_original, src_sequel, src_adaptation   source of the story (see detect_source)
    dir_* / lead_*         director and lead-actor track records, as of release (talent.py)
    txt_00..txt_15         the plot, as a 16-number summary of its text embedding (PCA)

Release year is deliberately NOT a feature: concepts are for next year, and a
year feature would make the judge extrapolate a trend line.

Source detection (for training rows; concepts are always "original")
    TMDB keywords such as "sequel", "remake", "based on comic" and title patterns
    such as "2", "II", "Part" mark sequels/franchises and adaptations. This is a
    heuristic: keywords are crowd-sourced, so a few films will be mislabelled.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .talent import TRACK_FEATURES

logger = logging.getLogger(__name__)

SEQUEL_KEYWORDS = ("sequel", "prequel", "remake", "reboot", "spin off", "spin-off", "franchise",
                   "superhero", "marvel cinematic universe", "dc extended universe", "shared universe")
ADAPTATION_KEYWORDS = ("based on comic", "based on novel", "based on book", "based on video game",
                       "based on tv series", "based on play", "based on true story",
                       "based on manga", "based on young adult novel", "based on short story",
                       "based on toy", "biography")
SEQUEL_TITLE = re.compile(r"(?:\b(?:2|3|4|5|6|7|8|9|II|III|IV|V|VI)\s*$)|(?:\b(?:Part|Chapter|Episode) [\w]+)"
                          r"|(?:\b(?:II|III|IV)\b:)|(?:\b\d:)", flags=re.I)


def detect_source(df: pd.DataFrame) -> pd.Series:
    """'sequel' | 'adaptation' | 'original' from TMDB keywords and the title."""
    kw = df.get("keywords", pd.Series("", index=df.index)).fillna("").str.lower()
    title = df["primaryTitle"].fillna("")
    sequel = kw.apply(lambda k: any(s in k for s in SEQUEL_KEYWORDS)) | title.str.contains(SEQUEL_TITLE)
    adaptation = kw.apply(lambda k: any(s in k for s in ADAPTATION_KEYWORDS))
    return pd.Series(np.select([sequel, adaptation], ["sequel", "adaptation"], "original"), index=df.index)


def genre_list(labels: pd.DataFrame) -> list[str]:
    g = labels["genres"].fillna("").str.split(",").explode().str.strip()
    return sorted(x for x in g.unique() if x and x != "Unknown")


@dataclass
class FeatureSpace:
    """Everything needed to turn a film or a concept into the same feature vector."""

    genres: list[str]
    pca: PCA | None
    text_cols: list[str]
    budget_col: str  # "budget_real" or "budget"

    @property
    def columns(self) -> list[str]:
        return ([f"g_{g}" for g in self.genres] + ["n_genres", "runtime", "log_budget", "is_english",
                "src_original", "src_sequel", "src_adaptation"] + TRACK_FEATURES + self.text_cols)

    # -- groups of columns, used to explain scores ("drivers") --
    def groups(self) -> dict[str, list[str]]:
        return {
            "genre mix": [f"g_{g}" for g in self.genres] + ["n_genres"],
            "budget": ["log_budget"],
            "runtime": ["runtime"],
            "director track record": [c for c in TRACK_FEATURES if c.startswith("dir_")],
            "lead cast track record": [c for c in TRACK_FEATURES if c.startswith("lead_")],
            "story (plot text)": self.text_cols,
            "original story": ["src_original", "src_sequel", "src_adaptation"],
        }


def build_feature_space(labels: pd.DataFrame, text_emb: np.ndarray, has_text: np.ndarray,
                        n_components: int, random_state: int) -> FeatureSpace:
    """Fit the PCA on the available plot embeddings and fix the column list."""
    pca = None
    k = min(n_components, int(has_text.sum()) - 1, text_emb.shape[1]) if has_text.any() else 0
    if k >= 2:
        pca = PCA(n_components=k, random_state=random_state).fit(text_emb[has_text])
        logger.info("Plot PCA: %d components explain %.0f%% of embedding variance",
                    k, 100 * pca.explained_variance_ratio_.sum())
    text_cols = [f"txt_{i:02d}" for i in range(k)]
    budget_col = "budget_real" if labels["budget_real"].notna().any() else "budget"
    return FeatureSpace(genres=genre_list(labels), pca=pca, text_cols=text_cols, budget_col=budget_col)


def film_features(films: pd.DataFrame, track: pd.DataFrame, text_emb: np.ndarray | None,
                  space: FeatureSpace) -> pd.DataFrame:
    """Feature rows for past films (index = tconst).

    films: rows of the Part A v2 labels (needs genres, runtimeMinutes, budget,
    budget_real, original_language, keywords, primaryTitle).
    text_emb: plot embeddings aligned with `films` (NaN rows where no plot).
    """
    X = pd.DataFrame(index=films["tconst"].to_numpy())
    genres = films["genres"].fillna("").str.split(",")
    for g in space.genres:
        X[f"g_{g}"] = genres.apply(lambda gs: float(g in gs)).to_numpy()
    X["n_genres"] = genres.apply(lambda gs: float(len([x for x in gs if x]))).to_numpy()
    X["runtime"] = pd.to_numeric(films["runtimeMinutes"], errors="coerce").to_numpy(dtype=float)
    budget = films[space.budget_col].fillna(films["budget"])
    X["log_budget"] = np.log10(budget.astype(float)).to_numpy()
    lang = films.get("original_language")
    X["is_english"] = ((lang == "en").astype(float).where(lang.notna())).to_numpy() if lang is not None else np.nan
    src = detect_source(films)
    for s in ("original", "sequel", "adaptation"):
        X[f"src_{s}"] = (src == s).astype(float).to_numpy()
    X = X.join(track.reindex(X.index))
    if text_emb is not None and space.pca is not None:
        txt = np.full((len(films), len(space.text_cols)), np.nan)
        ok = ~np.isnan(text_emb).any(axis=1)
        if ok.any():
            txt[ok] = space.pca.transform(text_emb[ok])
        X[space.text_cols] = txt
    X.index.name = "tconst"
    return X.reindex(columns=space.columns).astype(float)


def concept_features(concepts: list[dict], track_rows: list[dict], text_emb: np.ndarray | None,
                     space: FeatureSpace) -> pd.DataFrame:
    """Feature rows for generated concepts (same columns as film_features).

    Concepts are original stories in English with a budget in base-year dollars.
    """
    rows = []
    for c, track in zip(concepts, track_rows):
        row = {f"g_{g}": float(g in c["genres"]) for g in space.genres}
        row.update({"n_genres": float(len(c["genres"])), "runtime": float(c["runtime_min"]),
                    "log_budget": float(np.log10(c["budget_usd"])), "is_english": 1.0,
                    "src_original": 1.0, "src_sequel": 0.0, "src_adaptation": 0.0})
        row.update({k: float(v) if pd.notna(v) else np.nan for k, v in track.items()})
        rows.append(row)
    X = pd.DataFrame(rows)
    if text_emb is not None and space.pca is not None and len(concepts):
        X[space.text_cols] = space.pca.transform(text_emb)
    return X.reindex(columns=space.columns).astype(float)
