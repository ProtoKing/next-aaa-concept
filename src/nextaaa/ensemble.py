"""Five methods, one vote each: who is a hit on a given pair of axes?

v2 runs the SAME procedure twice ("stages"):

    Audience stage    quality (rating) x reach (votes)      -> Hit / Hidden Gem / Loser / Popular but Divisive
    Commercial stage  ROI x revenue (box office)            -> Blockbuster / Small-budget winner / Flop / Costly grosser

Both axes of each stage are year-relative robust z-scores (see normalize.py):
0 = the median film of its year, positive = better than its year's median.

The five voters, and what a "hit vote" means for each
----------------------------------------------------
Two PARTITION methods split every film into a group:
  1. KMeans          the film is in the cluster whose centre is furthest top-right.
                     k from elbow + silhouette (default 4, the quadrant hypothesis).
  2. Median split    the film is above its year's median on both axes.
                     No algorithm at all: the baseline any method must beat.

Three OUTLIER methods ask "is this film unusual?". A hit vote = unusual AND in
the top-right corner (unusual in the bottom-left is a spectacular flop, not a hit):
  3. DBSCAN          outside the main dense mass of films (eps from the k-distance knee).
  4. HDBSCAN         same density idea without choosing eps: it builds DBSCAN for
                     every eps at once into a hierarchy (Campello, Moulavi & Sander
                     2013). Its outlier score, GLOSH (Campello et al. 2015), compares
                     the density at which a film joins its cluster with the densest
                     point of that cluster: 0 = in the dense heart, near 1 = far out.
                     The 10% highest GLOSH scores count as unusual. (We use GLOSH
                     rather than HDBSCAN's "noise" label: on one continuous cloud of
                     films, HDBSCAN either forces artificial splits or labels most
                     films noise. Seen on the synthetic test data; the method
                     summary's noise_pct column lets you check the real data.)
  5. Isolation Forest  random axis-aligned cuts isolate unusual points in fewer
                     cuts; the average path length becomes an anomaly score
                     (Liu, Ting & Zhou 2008). No density or distance needed.

A film is a hit in a stage when at least `hit_min_votes` (3) of the 5 agree,
and "mega" when all `mega_min_votes` (5) agree. Because the outlier votes
also require the top-right corner, a 3-vote hit is (roughly) "top-right AND at
least one outlier method finds it exceptional" - which is what a studio means
by a hit: clearly above its year on both axes, and unusually so.

Why five instead of one? Each method has a known blind spot (KMeans always
returns k groups; DBSCAN depends on eps; GLOSH and Isolation Forest need a cut-off;
Isolation Forest ignores density). A majority vote is robust to any single
method's quirk, and the agreement matrix (Cohen's kappa) shows the stakeholders
how much the methods agree, rather than asking them to trust one.

References
    Campello, Moulavi, Sander (2013). Density-based clustering based on
        hierarchical density estimates. PAKDD.
    Campello, Moulavi, Zimek, Sander (2015). Hierarchical density estimates for data
        clustering, visualization, and outlier detection. ACM TKDD 10(1).  (GLOSH)
    McInnes, Healy, Astels (2017). hdbscan: Hierarchical density based clustering. JOSS 2(11).
    Liu, Ting, Zhou (2008). Isolation forest. IEEE ICDM. (Extended: ACM TKDD 6(1), 2012.)
    Ester, Kriegel, Sander, Xu (1996). DBSCAN. KDD.
    Rousseeuw (1987). Silhouettes. J. Comput. Appl. Math. 20.
    Cohen (1960). A coefficient of agreement for nominal scales. Educ. Psychol. Meas. 20(1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import IsolationForest
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import clustering as v1

try:  # the hdbscan package provides GLOSH outlier scores; sklearn's HDBSCAN does not
    import hdbscan as _hdbscan_lib
except ImportError:  # pragma: no cover - depends on the environment
    _hdbscan_lib = None

logger = logging.getLogger(__name__)

METHODS = ["kmeans", "median", "dbscan", "hdbscan", "iforest"]
METHOD_LABELS = {"kmeans": "KMeans", "median": "Median split", "dbscan": "DBSCAN",
                 "hdbscan": "HDBSCAN", "iforest": "Isolation Forest"}


@dataclass(frozen=True)
class Stage:
    """One 2-D hit question: which two axes, and what to call each corner."""

    key: str          # column prefix, e.g. "aud"
    title: str        # "Audience"
    y: str            # vertical-axis z column
    x: str            # horizontal-axis z column
    y_label: str
    x_label: str
    # (high on y, high on x) -> corner name. (True, True) is the hit corner.
    corners: dict = field(default_factory=dict)

    @property
    def hit_name(self) -> str:
        return self.corners[(True, True)]


AUDIENCE = Stage(
    key="aud", title="Audience", y="z_quality", x="z_reach",
    y_label="Quality: rating vs. same-year films (robust z)",
    x_label="Reach: votes vs. same-year films (robust z)",
    corners={(True, True): "Hit", (True, False): "Hidden Gem",
             (False, False): "Loser", (False, True): "Popular but Divisive"},
)
COMMERCIAL = Stage(
    key="com", title="Commercial", y="z_roi", x="z_revenue",
    y_label="ROI: multiple vs. same-year films (robust z)",
    x_label="Revenue: box office vs. same-year films (robust z)",
    corners={(True, True): "Blockbuster", (True, False): "Small-budget winner",
             (False, False): "Flop", (False, True): "Costly grosser"},
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def corner_of(df: pd.DataFrame, stage: Stage) -> pd.Series:
    """Corner name from the signs of the two year-relative z-scores."""
    hi_y, hi_x = df[stage.y] >= 0, df[stage.x] >= 0
    names = np.select([hi_y & hi_x, hi_y & ~hi_x, ~hi_y & ~hi_x],
                      [stage.corners[(True, True)], stage.corners[(True, False)],
                       stage.corners[(False, False)]],
                      default=stage.corners[(False, True)])
    return pd.Series(names, index=df.index)


def name_partition(centres: np.ndarray, stage: Stage) -> tuple[dict[int, str], int]:
    """Name KMeans clusters after the stage's corners; return (names, hit cluster id).

    centres are in (y, x) order. With k = 4, clusters are matched one-to-one to
    the four corner prototypes (Hungarian algorithm). Otherwise each cluster is
    described by its centre. The hit cluster is the one furthest top-right.
    """
    k = len(centres)
    corner_keys = list(stage.corners)
    if k == 4:
        protos = np.array([[1 if hy else -1, 1 if hx else -1] for hy, hx in corner_keys], dtype=float)
        cost = np.linalg.norm(centres[:, None, :] - protos[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        names = {int(r): stage.corners[corner_keys[c]] for r, c in zip(rows, cols)}
    else:
        names = {i: f"{stage.title} cluster {i} (y {cy:+.1f}, x {cx:+.1f})"
                 for i, (cy, cx) in enumerate(centres)}
    score = centres.sum(axis=1)
    top_right = np.where((centres[:, 0] > 0) & (centres[:, 1] > 0))[0]
    hit_id = int(top_right[np.argmax(score[top_right])]) if len(top_right) else int(np.argmax(score))
    if k == 4 and names[hit_id] != stage.hit_name:
        logger.warning("%s: KMeans corner names and the top-right cluster disagree.", stage.title)
    return names, hit_id


def outside_main_mass(labels: np.ndarray) -> np.ndarray:
    """True for films NOT in the largest density cluster (noise or minor clusters).

    For DBSCAN and HDBSCAN, the largest cluster is the mass of ordinary films.
    Everything else (noise, or a small separate pocket) is away from the ordinary.
    """
    clustered = labels[labels >= 0]
    if len(clustered) == 0:
        return np.ones(len(labels), dtype=bool)
    main = np.bincount(clustered).argmax()
    return labels != main


def glosh_scores(X: np.ndarray, min_cluster_size: int, min_samples: int, max_points: int,
                 random_state: int) -> tuple[np.ndarray, np.ndarray, str]:
    """HDBSCAN's GLOSH outlier score for every film: (scores, labels, how).

    Uses the `hdbscan` package when installed (exact GLOSH). Without it, falls
    back to the distance to the min_samples-th nearest neighbour (the classic
    kNN outlier score, Ramaswamy et al. 2000), which is what GLOSH reduces to
    when all films form one cluster; since we threshold on a percentile, only
    the ranking matters. Large data: fit a sample, give the rest the score of
    their nearest sampled film.
    """
    n = len(X)
    rng = np.random.default_rng(random_state)
    idx = np.arange(n) if n <= max_points else rng.choice(n, size=max_points, replace=False)
    scale = len(idx) / n
    if _hdbscan_lib is not None:
        model = _hdbscan_lib.HDBSCAN(min_cluster_size=max(5, int(min_cluster_size * scale)),
                                     min_samples=max(2, int(round(min_samples * scale))),
                                     allow_single_cluster=True).fit(X[idx])
        fitted_scores, fitted_labels = np.nan_to_num(model.outlier_scores_), model.labels_
        how = "GLOSH (hdbscan package)"
    else:
        logger.warning("Package 'hdbscan' not installed: using the kNN-distance approximation "
                       "of GLOSH. Install it with: pip install hdbscan")
        k = max(2, int(round(min_samples * scale)))
        dist, _ = NearestNeighbors(n_neighbors=k).fit(X[idx]).kneighbors(X[idx])
        fitted_scores, fitted_labels = dist[:, -1], np.zeros(len(idx), dtype=int)
        how = "kNN-distance approximation of GLOSH"
    if len(idx) == n:
        return fitted_scores, fitted_labels, how
    scores, labels = np.empty(n), np.empty(n, dtype=int)
    scores[idx], labels[idx] = fitted_scores, fitted_labels
    rest = np.setdiff1d(np.arange(n), idx)
    _, nearest = NearestNeighbors(n_neighbors=1).fit(X[idx]).kneighbors(X[rest])
    scores[rest], labels[rest] = fitted_scores[nearest[:, 0]], fitted_labels[nearest[:, 0]]
    return scores, labels, how


def fit_iforest(X: np.ndarray, contamination, n_estimators: int,
                random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """(is_anomaly, anomaly_score). Higher score = more unusual (0 to 1 scale
    of the original paper: above 0.5 is the paper's anomaly threshold)."""
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                            random_state=random_state).fit(X)
    is_anomaly = model.predict(X) == -1
    score = -model.score_samples(X)  # sklearn returns the negated paper score
    return is_anomaly, score


# ---------------------------------------------------------------------------
# Running a stage
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: Stage
    frame: pd.DataFrame            # input rows + all <key>_* columns
    kmeans: v1.KMeansDiagnostics
    k_used: int
    kdist: np.ndarray
    eps: float
    method_summary: pd.DataFrame   # one row per method
    agreement: pd.DataFrame        # 5 x 5 Cohen's kappa
    vote_counts: pd.Series         # films by number of hit votes (0-5)


def run_stage(df: pd.DataFrame, stage: Stage, cfg, k: int | None = None) -> StageResult:
    """Run all five methods on one stage and add the vote columns.

    Added columns (p = stage.key):
        p_corner                  corner by year-median split (also the median voter)
        p_kmeans                  KMeans cluster name
        p_dbscan_id, p_hdbscan_id raw density labels (-1 = noise)
        p_glosh_score             HDBSCAN's GLOSH outlier score (0 = dense heart, ~1 = far out)
        p_iforest_score           Isolation Forest anomaly score
        p_unusual_<outlier method> True if DBSCAN / HDBSCAN / Isolation Forest finds it unusual
        p_vote_<method>           True if that method calls the film a hit
        p_votes                   number of hit votes (0-5)
        p_hit, p_mega             >= hit_min_votes, >= mega_min_votes
    """
    p, rs = stage.key, cfg.random_state
    out = df.copy()
    X = StandardScaler().fit_transform(out[[stage.y, stage.x]].to_numpy(dtype=float))
    n = len(out)
    logger.info("%s stage: %s films", stage.title, f"{n:,}")

    out[f"{p}_corner"] = corner_of(out, stage)
    in_hit_corner = (out[f"{p}_corner"] == stage.hit_name).to_numpy()

    # 1) KMeans -----------------------------------------------------------------
    diag = v1.kmeans_diagnostics(X, cfg.k_range, cfg.silhouette_sample_size, rs)
    k_used = k or cfg.k_hypothesis
    km_labels, centres = v1.fit_kmeans(X, k_used, rs)
    names, hit_id = name_partition(centres, stage)
    out[f"{p}_kmeans"] = pd.Series(km_labels, index=out.index).map(names)
    out[f"{p}_vote_kmeans"] = km_labels == hit_id

    # 2) Median split -------------------------------------------------------------
    out[f"{p}_vote_median"] = in_hit_corner

    # 3) DBSCAN -------------------------------------------------------------------------
    kdist = v1.k_distance_curve(X, cfg.dbscan_min_samples, cfg.silhouette_sample_size, rs)
    eps = v1.suggest_eps(kdist)
    db = v1.fit_dbscan(X, eps, cfg.dbscan_min_samples, cfg.dbscan_max_points, rs)
    db_out = outside_main_mass(db)
    out[f"{p}_dbscan_id"] = db
    out[f"{p}_unusual_dbscan"] = db_out
    out[f"{p}_vote_dbscan"] = db_out & in_hit_corner

    # 4) HDBSCAN (GLOSH outlier score) ----------------------------------------------------
    mcs = max(cfg.hdbscan_min_cluster_floor, int(cfg.hdbscan_min_cluster_frac * n))
    glosh, hdb, glosh_how = glosh_scores(X, mcs, cfg.dbscan_min_samples, cfg.dbscan_max_points, rs)
    hdb_out = glosh >= np.quantile(glosh, cfg.glosh_quantile)
    out[f"{p}_hdbscan_id"] = hdb
    out[f"{p}_glosh_score"] = glosh
    out[f"{p}_unusual_hdbscan"] = hdb_out
    out[f"{p}_vote_hdbscan"] = hdb_out & in_hit_corner

    # 5) Isolation Forest -------------------------------------------------------------
    iso_flag, iso_score = fit_iforest(X, cfg.iforest_contamination, cfg.iforest_n_estimators, rs)
    out[f"{p}_iforest_score"] = iso_score
    out[f"{p}_unusual_iforest"] = iso_flag
    out[f"{p}_vote_iforest"] = iso_flag & in_hit_corner

    # Votes -----------------------------------------------------------------------------
    vote_cols = [f"{p}_vote_{m}" for m in METHODS]
    out[f"{p}_votes"] = out[vote_cols].sum(axis=1).astype(int)
    out[f"{p}_hit"] = out[f"{p}_votes"] >= cfg.hit_min_votes
    out[f"{p}_mega"] = out[f"{p}_votes"] >= cfg.mega_min_votes

    sil_k = float(diag.table.set_index("k")["silhouette"].get(k_used, np.nan))
    db_sil = v1.dbscan_silhouettes(X, db, cfg.silhouette_sample_size, rs)
    hdb_sil = v1.dbscan_silhouettes(X, np.where(hdb_out, -1, 0), cfg.silhouette_sample_size, rs)
    summary = pd.DataFrame([
        {"method": "KMeans", "setting": f"k = {k_used} (elbow {diag.k_elbow}, silhouette best {diag.k_silhouette})",
         "flagged_unusual_pct": np.nan, "silhouette": sil_k, "clusters": k_used},
        {"method": "Median split", "setting": "above own year's median on both axes",
         "flagged_unusual_pct": np.nan, "silhouette": np.nan, "clusters": 4},
        {"method": "DBSCAN", "setting": f"eps = {eps:.3f}, min_samples = {cfg.dbscan_min_samples}",
         "flagged_unusual_pct": 100 * db_out.mean(), "silhouette": db_sil["core_vs_outliers"],
         "clusters": len(set(db) - {-1}), "noise_pct": 100 * np.mean(db == -1)},
        {"method": "HDBSCAN", "setting": f"{glosh_how}; top {100 * (1 - cfg.glosh_quantile):.0f}% "
                                         f"(min_cluster_size = {mcs}, min_samples = {cfg.dbscan_min_samples})",
         "flagged_unusual_pct": 100 * hdb_out.mean(), "silhouette": hdb_sil["core_vs_outliers"],
         "clusters": len(set(hdb) - {-1}), "noise_pct": 100 * np.mean(hdb == -1)},
        {"method": "Isolation Forest", "setting": f"contamination = {cfg.iforest_contamination}, "
                                                  f"{cfg.iforest_n_estimators} trees",
         "flagged_unusual_pct": 100 * iso_flag.mean(), "silhouette": np.nan, "clusters": np.nan},
    ])
    summary["hit_votes_pct"] = [100 * out[c].mean() for c in vote_cols]
    for _, row in summary.iterrows():
        logger.info("  %-16s hit votes %5.1f%%  unusual %s", row["method"], row["hit_votes_pct"],
                    "-" if pd.isna(row["flagged_unusual_pct"]) else f"{row['flagged_unusual_pct']:.1f}%")

    return StageResult(stage=stage, frame=out, kmeans=diag, k_used=k_used, kdist=kdist, eps=eps,
                       method_summary=summary, agreement=agreement_matrix(out, p),
                       vote_counts=out[f"{p}_votes"].value_counts().sort_index())


def agreement_matrix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """5 x 5 Cohen's kappa between the methods' hit votes (1 = identical, 0 = chance)."""
    labels = [METHOD_LABELS[m] for m in METHODS]
    mat = pd.DataFrame(np.eye(len(METHODS)), index=labels, columns=labels)
    for i, a in enumerate(METHODS):
        for j, b in enumerate(METHODS):
            if j > i:
                va, vb = df[f"{prefix}_vote_{a}"], df[f"{prefix}_vote_{b}"]
                kappa = cohen_kappa_score(va, vb) if va.nunique() > 1 or vb.nunique() > 1 else 1.0
                mat.iloc[i, j] = mat.iloc[j, i] = kappa
    return mat
