"""Turn the quality x reach plane into hit labels, three independent ways.

The hypothesis (from the project brief) is four quadrants:

                       low reach              high reach
    high quality   |  Hidden Gem          |  Hit                   |
    low quality    |  Loser               |  Popular but Divisive  |

We test it with three methods that make different assumptions, then combine:

1. KMeans (k chosen with the elbow method and silhouette score)
   Splits the films into k compact, round groups. Caveat to say out loud in the
   deck: KMeans always returns k groups, even when the data is one continuous
   cloud. It *partitions*; it does not prove the groups exist. The silhouette
   score is how we check whether they do (near 0 = overlapping, near 1 = crisp).

2. DBSCAN (eps chosen with the k-distance plot)
   Finds dense regions and labels points in sparse regions as "noise". It does
   not need k. On this data it typically finds one big dense core of ordinary
   films plus sparse outliers, and the outliers in the top-right corner are
   exactly the exceptional hits. So we use DBSCAN as an *outlier detector*:
   it tells us WHO is exceptional, the quadrant tells us HOW.

3. Median split (the sanity baseline)
   Above/below the median on each axis. No algorithm at all; if KMeans and
   DBSCAN cannot beat this for interpretability, we should say so.

Final label: a film `is_hit` when at least 2 of the 3 methods call it a hit
(a majority vote, which is robust to any one method's quirks). Hits that DBSCAN
also flags as exceptional become "Mega-hit", the AAA-grade tier.

References
    Ester, Kriegel, Sander, Xu (1996). A density-based algorithm for discovering
        clusters in large spatial databases with noise. KDD.
    Schubert, Sander, Ester, Kriegel, Xu (2017). DBSCAN revisited, revisited.
        ACM Transactions on Database Systems 42(3).
    Rousseeuw (1987). Silhouettes: a graphical aid to the interpretation and
        validation of cluster analysis. J. Computational and Applied Math 20.
    Satopää, Albrecht, Irwin, Raghavan (2011). Finding a "Kneedle" in a
        haystack: detecting knee points in system behavior. ICDCS Workshops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# Each quadrant's "prototype" position in standardised (quality, reach) space.
QUADRANTS: dict[str, tuple[int, int]] = {
    "Hit": (+1, +1),
    "Hidden Gem": (+1, -1),
    "Loser": (-1, -1),
    "Popular but Divisive": (-1, +1),
}
TYPICAL = "Typical (dense region)"


# ---------------------------------------------------------------------------
# Knee / elbow detection
# ---------------------------------------------------------------------------

def find_knee(x, y) -> float:
    """Return the x where a curve bends most sharply (the "elbow" or "knee")."""
    return float(np.asarray(x, dtype=float)[find_knee_index(x, y)])


def find_knee_index(x, y) -> int:
    """Position of the point where a curve bends most sharply.

    Kneedle idea (Satopää et al. 2011), simplified: rescale both axes to 0-1,
    draw a straight line from the first point to the last, and pick the point
    farthest from that line. Works for falling curves (elbow plot) and rising
    ones (k-distance plot) alike.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return 0
    xn = (x - x.min()) / (np.ptp(x) or 1.0)
    yn = (y - y.min()) / (np.ptp(y) or 1.0)
    # Distance from each point to the chord between the first and last point.
    p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    chord = (p1 - p0) / np.linalg.norm(p1 - p0)
    rel = np.column_stack([xn, yn]) - p0
    dist = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0])
    return int(np.argmax(dist))


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

@dataclass
class KMeansDiagnostics:
    table: pd.DataFrame  # one row per k: inertia, silhouette
    k_elbow: int         # k suggested by the elbow of the inertia curve
    k_silhouette: int    # k with the highest silhouette score


def kmeans_diagnostics(X: np.ndarray, k_values, sample_size: int,
                       random_state: int) -> KMeansDiagnostics:
    """Fit KMeans for each k and record the two model-selection signals.

    inertia     total squared distance from films to their cluster centre.
                Always falls as k grows; the "elbow" is where adding another
                cluster stops buying much. That k is a good trade-off.
    silhouette  for each film: (b - a) / max(a, b), where a = mean distance to
                its own cluster and b = mean distance to the nearest other
                cluster. Averaged over films. Higher = better separated.
                Computed on a random sample because it is O(n^2).
    """
    rows = []
    n_sample = min(sample_size, len(X))
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=n_sample, random_state=random_state)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        logger.info("KMeans k=%d  inertia=%.0f  silhouette=%.3f", k, km.inertia_, sil)
    table = pd.DataFrame(rows)
    k_elbow = int(find_knee(table["k"], table["inertia"]))
    k_sil = int(table.loc[table["silhouette"].idxmax(), "k"])
    return KMeansDiagnostics(table=table, k_elbow=k_elbow, k_silhouette=k_sil)


def fit_kmeans(X: np.ndarray, k: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit the final KMeans. Returns (labels, centres)."""
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
    return km.labels_, km.cluster_centers_


def _level(value: float) -> str:
    return "High" if value > 0.5 else "Low" if value < -0.5 else "Mid"


def name_kmeans_clusters(centres: np.ndarray) -> tuple[dict[int, str], int]:
    """Give each KMeans cluster a business name, and identify the hit cluster.

    k = 4: match clusters one-to-one to the four quadrant prototypes with the
           Hungarian algorithm (minimum total distance), so names never repeat.
    other k: describe each centre, e.g. "Quality High / Reach Mid".

    The hit cluster is the one whose centre is furthest into the top-right
    (largest quality + reach) among centres that are positive on both axes.
    """
    k = len(centres)
    if k == 4:
        protos = np.array(list(QUADRANTS.values()), dtype=float)
        cost = np.linalg.norm(centres[:, None, :] - protos[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        names = {int(r): list(QUADRANTS)[c] for r, c in zip(rows, cols)}
    else:
        names = {i: f"Quality {_level(q)} / Reach {_level(r)}" for i, (q, r) in enumerate(centres)}

    score = centres.sum(axis=1)
    top_right = np.where((centres[:, 0] > 0) & (centres[:, 1] > 0))[0]
    hit_id = int(top_right[np.argmax(score[top_right])]) if len(top_right) else int(np.argmax(score))
    if k == 4 and names[hit_id] != "Hit":
        logger.warning("KMeans quadrant naming and top-right cluster disagree; "
                       "the four-quadrant hypothesis fits this data poorly.")
    if k != 4:
        names[hit_id] = names[hit_id] + " (Hit)"
    return names, hit_id


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def k_distance_curve(X: np.ndarray, min_samples: int, sample_size: int,
                     random_state: int) -> np.ndarray:
    """Sorted distance from each point to its `min_samples`-th nearest neighbour.

    This is the standard way to pick DBSCAN's eps (Ester et al. 1996): points
    inside dense regions have small k-distances; outliers have large ones.
    The knee of the sorted curve is where "dense" turns into "sparse".

    Neighbours are searched in the FULL data (so density is measured
    correctly); only the query points are sampled, for speed.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    # n_neighbors counts the query point itself, matching DBSCAN's definition
    # of a core point: at least min_samples points (itself included) within eps.
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dist, _ = nn.kneighbors(X[idx])
    return np.sort(dist[:, -1])


def suggest_eps(kdist_sorted: np.ndarray) -> float:
    """eps = the distance value at the knee of the sorted k-distance curve."""
    i = find_knee_index(np.arange(len(kdist_sorted)), kdist_sorted)
    return float(kdist_sorted[i])


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int, max_points: int,
               random_state: int) -> np.ndarray:
    """Fit DBSCAN; label -1 means noise (an outlier).

    For very large data we fit on a random sample (DBSCAN's memory grows with
    neighbourhood size) and then give every other point the label of its
    nearest core point if that point is within eps, else -1. min_samples is
    scaled down with the sample so the density threshold stays the same.
    """
    n = len(X)
    if n <= max_points:
        return DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_points, replace=False)
    scaled_min = max(2, int(round(min_samples * max_points / n)))
    db = DBSCAN(eps=eps, min_samples=scaled_min).fit(X[idx])

    labels = np.full(n, -1, dtype=int)
    labels[idx] = db.labels_
    core = idx[db.core_sample_indices_]
    if len(core):
        rest = np.setdiff1d(np.arange(n), idx)
        nn = NearestNeighbors(n_neighbors=1).fit(X[core])
        dist, nearest = nn.kneighbors(X[rest])
        within = dist[:, 0] <= eps
        labels[rest[within]] = labels[core[nearest[within, 0]]]
    logger.info("DBSCAN fitted on a %s-point sample (min_samples scaled to %d).",
                f"{max_points:,}", scaled_min)
    return labels


def dbscan_eps_sweep(X: np.ndarray, eps_values, min_samples: int, max_points: int,
                     random_state: int) -> pd.DataFrame:
    """How sensitive is DBSCAN to eps? Clusters found and share of noise per eps."""
    rows = []
    for eps in eps_values:
        labels = fit_dbscan(X, eps, min_samples, max_points, random_state)
        rows.append({"eps": eps, "n_clusters": len(set(labels) - {-1}),
                     "noise_share": float(np.mean(labels == -1))})
    return pd.DataFrame(rows)


def dbscan_silhouettes(X: np.ndarray, labels: np.ndarray, sample_size: int,
                       random_state: int) -> dict:
    """Two silhouette readings for DBSCAN.

    non_noise:          silhouette over the clusters only (noise excluded).
                        Undefined when DBSCAN finds a single dense cluster,
                        which is itself a finding: no natural separation.
    core_vs_outliers:   silhouette of the two-way split "in a dense region"
                        vs "outlier". Tells us how distinct the outliers are.
    """
    out = {"non_noise": np.nan, "core_vs_outliers": np.nan}
    mask = labels != -1
    if len(set(labels[mask])) >= 2:
        out["non_noise"] = silhouette_score(X[mask], labels[mask],
                                            sample_size=min(sample_size, int(mask.sum())),
                                            random_state=random_state)
    if 0 < mask.sum() < len(labels):
        out["core_vs_outliers"] = silhouette_score(X, mask.astype(int),
                                                   sample_size=min(sample_size, len(X)),
                                                   random_state=random_state)
    return out


# ---------------------------------------------------------------------------
# Quadrant naming helpers
# ---------------------------------------------------------------------------

def quadrant_of(quality: pd.Series, reach: pd.Series,
                q_cut: float = 0.0, r_cut: float = 0.0) -> pd.Series:
    """Name each film's quadrant given cut points on the two axes."""
    hi_q, hi_r = quality >= q_cut, reach >= r_cut
    names = np.select(
        [hi_q & hi_r, hi_q & ~hi_r, ~hi_q & ~hi_r],
        ["Hit", "Hidden Gem", "Loser"],
        default="Popular but Divisive",
    )
    return pd.Series(names, index=quality.index)


def name_dbscan(frame: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Films in any dense region are "Typical"; noise points (label -1) are
    exceptional films, named by the quadrant they sit in.

    DBSCAN sometimes also finds a few small secondary dense pockets. They are
    still dense, i.e. not exceptional, so they count as Typical here. The raw
    DBSCAN cluster ids stay available in the `dbscan_id` column.
    """
    quadrant = quadrant_of(frame["x_quality"], frame["x_reach"])
    names = "Exceptional " + quadrant
    return names.where(pd.Series(labels == -1, index=frame.index), TYPICAL)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

@dataclass
class ClusteringResult:
    frame: pd.DataFrame           # films with all cluster and hit columns
    kmeans: KMeansDiagnostics
    k_used: int
    kdist: np.ndarray             # sorted k-distance curve
    eps: float
    dbscan_silhouette: dict
    eps_sweep: pd.DataFrame
    agreement: pd.DataFrame       # how much the three methods agree


def run_clustering(frame: pd.DataFrame, X: np.ndarray, cfg, k: int | None = None) -> ClusteringResult:
    """Run all three methods and assemble the final hit labels.

    k: number of KMeans clusters. Default is the four-quadrant hypothesis
       (cfg.k_hypothesis); the diagnostics report what the elbow and the
       silhouette suggest, so you can compare and override.
    """
    rs = cfg.random_state
    out = frame.copy()

    # 1) KMeans --------------------------------------------------------------
    diag = kmeans_diagnostics(X, cfg.k_range, cfg.silhouette_sample_size, rs)
    k_used = k or cfg.k_hypothesis
    logger.info("Elbow suggests k=%d, silhouette suggests k=%d; using k=%d.",
                diag.k_elbow, diag.k_silhouette, k_used)
    km_labels, centres = fit_kmeans(X, k_used, rs)
    km_names, hit_id = name_kmeans_clusters(centres)
    out["kmeans_id"] = km_labels
    out["cluster_kmeans"] = pd.Series(km_labels, index=out.index).map(km_names)
    out["hit_kmeans"] = km_labels == hit_id

    # 2) DBSCAN --------------------------------------------------------------
    kdist = k_distance_curve(X, cfg.dbscan_min_samples, cfg.silhouette_sample_size, rs)
    eps = suggest_eps(kdist)
    db_labels = fit_dbscan(X, eps, cfg.dbscan_min_samples, cfg.dbscan_max_points, rs)
    out["dbscan_id"] = db_labels
    out["cluster_dbscan"] = name_dbscan(out, db_labels)
    out["hit_dbscan"] = out["cluster_dbscan"] == "Exceptional Hit"
    db_sil = dbscan_silhouettes(X, db_labels, cfg.silhouette_sample_size, rs)
    sweep = dbscan_eps_sweep(X, [eps * f for f in (0.5, 0.75, 1.0, 1.5, 2.0)],
                             cfg.dbscan_min_samples, cfg.dbscan_max_points, rs)
    logger.info("DBSCAN eps=%.3f: %d clusters, %.1f%% noise", eps,
                len(set(db_labels) - {-1}), 100 * np.mean(db_labels == -1))

    # 3) Median split ----------------------------------------------------------
    out["quadrant_median"] = quadrant_of(out["x_quality"], out["x_reach"],
                                         out["x_quality"].median(), out["x_reach"].median())
    out["hit_quadrant"] = out["quadrant_median"] == "Hit"

    # Majority vote ------------------------------------------------------------
    out["hit_votes"] = out[["hit_kmeans", "hit_dbscan", "hit_quadrant"]].sum(axis=1)
    out["is_hit"] = out["hit_votes"] >= 2
    out["hit_tier"] = np.select([out["is_hit"] & out["hit_dbscan"], out["is_hit"]],
                                ["Mega-hit", "Hit"], default="Not a hit")

    return ClusteringResult(frame=out, kmeans=diag, k_used=k_used, kdist=kdist, eps=eps,
                            dbscan_silhouette=db_sil, eps_sweep=sweep,
                            agreement=label_agreement(out))


def label_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairwise agreement between the three hit flags.

    Cohen's kappa: 1 = perfect agreement, 0 = no better than chance. Raw
    "% agree" is misleading when hits are rare (two methods that both say
    "not a hit" to everything agree 85% of the time), which is why we use kappa.
    Adjusted Rand index compares the full KMeans vs median-split groupings.
    """
    flags = ["hit_kmeans", "hit_dbscan", "hit_quadrant"]
    rows = []
    for i, a in enumerate(flags):
        for b in flags[i + 1:]:
            rows.append({"pair": f"{a} vs {b}",
                         "cohen_kappa": cohen_kappa_score(frame[a], frame[b]),
                         "pct_agree": float((frame[a] == frame[b]).mean())})
    rows.append({"pair": "cluster_kmeans vs quadrant_median (ARI)",
                 "cohen_kappa": adjusted_rand_score(frame["cluster_kmeans"], frame["quadrant_median"]),
                 "pct_agree": np.nan})
    return pd.DataFrame(rows)


def cluster_profile(frame: pd.DataFrame, label_col: str, n_examples: int = 3) -> pd.DataFrame:
    """One row per cluster: size, typical rating and votes, and example films."""
    g = frame.groupby(label_col, observed=True)
    prof = pd.DataFrame({
        "films": g.size(),
        "share_pct": 100 * g.size() / len(frame),
        "median_rating": g["averageRating"].median(),
        "median_votes": g["numVotes"].median(),
        "median_reach_pct": g["reach_pct"].median(),
    })
    examples = (frame.sort_values("numVotes", ascending=False)
                .groupby(label_col, observed=True)["primaryTitle"]
                .apply(lambda s: "; ".join(s.head(n_examples))))
    prof["examples (most voted)"] = examples
    return prof.sort_values("median_rating", ascending=False).round(2)
