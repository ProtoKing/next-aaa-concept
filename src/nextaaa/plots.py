"""Every figure for the deck, in one consistent, colour-blind-aware style.

Design rules followed throughout (so the slides look like one system):
  * Colour carries meaning only where it must. Context is grey; the story is in colour.
  * Each cluster keeps the SAME colour in every chart (colour follows the entity).
  * At most three saturated colours in a scatter, each paired with a different
    marker shape, so the charts still work for colour-blind viewers and in print.
  * Thin marks, hairline solid grid, no dual axes, selective direct labels.

Every function returns the matplotlib Figure; `save()` writes a PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .clustering import QUADRANTS, TYPICAL
from .ratings import weighted_rating

# ---- Palette (validated categorical order; light surface) --------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
CONTEXT = "#b9b8b1"  # grey for "everything else"
CORE = "#dcdbd5"     # lighter grey for DBSCAN's dense core of typical films
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TIER_LIGHT, TIER_DARK = "#86b6ef", "#1c5cab"  # ordinal ramp: Hit < Mega-hit

# Fixed identity per quadrant: colour + marker (secondary encoding).
QUADRANT_STYLE = {
    "Hit": (BLUE, "o"),
    "Hidden Gem": (AQUA, "^"),
    "Popular but Divisive": (ORANGE, "s"),
    "Loser": (CONTEXT, "."),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
    "lines.linewidth": 2, "axes.axisbelow": True,  # grid behind the data
})

_thousands = FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")


def save(fig: plt.Figure, path: Path, dpi: int = 200) -> Path:
    """Write a figure to PNG and close it (keeps memory flat in long runs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _titles(ax, title: str, subtitle: str | None = None) -> None:
    """Bold title with an optional grey subtitle underneath (no overlap)."""
    ax.set_title(title, pad=26 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def _style_for(name: str, i: int = 0) -> tuple[str, str]:
    """Colour and marker for a cluster name (quadrant names have fixed styles)."""
    for quad, style in QUADRANT_STYLE.items():
        if name.endswith(quad) or name.startswith(quad):
            return style
    if name == TYPICAL:
        return CORE, "."
    return SERIES[i % len(SERIES)], "o"


# ---------------------------------------------------------------------------
# 1. Highest-rated film of the target year
# ---------------------------------------------------------------------------

def plot_rating_vs_votes(movies: pd.DataFrame, info: dict, leaderboard: pd.DataFrame,
                         n_label: int = 5) -> plt.Figure:
    """Rating vs votes for one year, with lines of equal shrunk rating.

    Each grey curve joins all (votes, rating) points that earn the same shrunk
    rating. That makes the trade-off visible: the fewer votes a film has, the
    higher its raw rating must be to reach the same curve.
    """
    year, m, C = info["year"], info["m"], info["C"]
    pool = movies[(movies["startYear"] == year) & (movies["numVotes"] >= info["min_votes"])]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(pool["numVotes"], pool["averageRating"], s=6, color=CONTEXT, alpha=0.5, lw=0)
    ax.set_xscale("log")

    # Iso-curves: solve shrunk = level for R given v -> R = (level*(v+m) - m*C) / v
    v = np.logspace(np.log10(max(pool["numVotes"].min(), 1)), np.log10(pool["numVotes"].max()), 200)
    for level in (6.5, 7.0, 7.5, 8.0):
        r = (level * (v + m) - m * C) / v
        ok = (r <= 10) & (r >= 1)
        if ok.any():
            ax.plot(v[ok], r[ok], color=INK_2, lw=0.8, alpha=0.6)
            # Label each curve where it enters the chart at the top.
            ax.text(v[ok][0], r[ok][0], f"{level}", fontsize=8, color=INK_2, ha="right", va="bottom")

    top = leaderboard.head(n_label)
    ax.scatter(top["numVotes"], top["averageRating"], s=60, color=BLUE,
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    # Short "#n" tags on the points; full titles in a key, so labels never collide.
    key = []
    for _, row in top.iterrows():
        ax.annotate(f"#{row['rank_shrunk']}", (row["numVotes"], row["averageRating"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8.5, color=INK, fontweight="bold")
        key.append(f"#{row['rank_shrunk']}  {row['primaryTitle']}  "
                   f"({row['averageRating']:.1f}, {row['numVotes']:,.0f} votes)")
    ax.text(0.01, 0.02, "\n".join(key), transform=ax.transAxes, fontsize=8.5, color=INK,
            va="bottom", ha="left", bbox={"facecolor": SURFACE, "edgecolor": GRID, "boxstyle": "round,pad=0.5"})

    ax.xaxis.set_major_formatter(_thousands)
    ax.set_xlabel("Number of IMDb votes (log scale)")
    ax.set_ylabel("averageRating")
    _titles(ax, f"Highest-rated films of {year}: rating balanced against votes",
            f"Blue = top {n_label} by shrunk rating (m = {m:,.0f} votes, C = {C:.2f}). "
            f"Grey curves join points with equal shrunk rating.")
    return fig


def plot_m_sensitivity(long: pd.DataFrame, default_q: float, top_n: int = 8) -> plt.Figure:
    """Bump chart: each film's rank as the prior strength m increases."""
    qs = sorted(long["quantile"].unique())
    m_by_q = long.groupby("quantile")["m"].first()
    at_default = long[np.isclose(long["quantile"], default_q)].nsmallest(3, "rank")["tconst"].tolist()
    show = long.loc[long["rank"] <= top_n, "tconst"].unique()

    pos = {q: i for i, q in enumerate(qs)}  # evenly spaced x positions
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tconst, g in long[long["tconst"].isin(show)].groupby("tconst"):
        g = g.sort_values("quantile")
        if tconst in at_default:
            color, lw, z = SERIES[at_default.index(tconst)], 2.2, 3
        else:
            color, lw, z = CONTEXT, 1.2, 1
        ax.plot(g["quantile"].map(pos), g["rank"], color=color, lw=lw, marker="o", ms=4, zorder=z)
        last = g.iloc[-1]
        if last["rank"] <= top_n:
            ax.text(len(qs) - 1 + 0.08, last["rank"], last["primaryTitle"], fontsize=8,
                    color=INK if tconst in at_default else INK_2, va="center")

    ax.set_ylim(top_n + 0.5, 0.5)
    ax.set_yticks(range(1, top_n + 1))
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([f"p{q * 100:g}\n{m_by_q[q]:,.0f}" for q in qs])
    default_pos = min(pos, key=lambda q: abs(q - default_q))
    ax.axvline(pos[default_pos], color=INK_2, lw=0.8)
    ax.set_xlabel("Prior strength m (percentile of votes, and the vote count it equals)")
    ax.set_ylabel("Rank")
    _titles(ax, "How robust is the #1? Rank as we demand more votes",
            "Left: trusts each film's own rating. Right: demands many votes. "
            "Coloured = top 3 at our default (vertical line).")
    return fig


# ---------------------------------------------------------------------------
# 2. Most popular actor
# ---------------------------------------------------------------------------

def plot_actor_side_by_side(pop: pd.DataFrame, year: int, top_n: int = 10) -> plt.Figure:
    """Two bar charts: top actors by Reach and by Net goodwill."""
    a = pop.nsmallest(top_n, "rank_reach").iloc[::-1]
    b = pop.nsmallest(top_n, "rank_goodwill").iloc[::-1]
    both = set(a["nconst"]) & set(b["nconst"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, d, col, title in ((axes[0], a, "reach", "Reach"),
                              (axes[1], b, "net_goodwill", "Net goodwill (reach x quality)")):
        colors = [BLUE if n in both else TIER_LIGHT for n in d["nconst"]]
        ax.barh(d["primaryName"], d[col], color=colors, height=0.7)
        ax.xaxis.set_major_formatter(_thousands)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    axes[0].set_xlabel("Billing-weighted votes")
    axes[1].set_xlabel("Billing-weighted votes x (shrunk rating - average)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle(f"Most popular actor of {year}: it depends on the definition",
                 x=0.01, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.925, f"Dark blue = in the top {top_n} under both definitions.",
             fontsize=9, color=INK_2)
    return fig


# ---------------------------------------------------------------------------
# 3. Choosing the number of clusters
# ---------------------------------------------------------------------------

def plot_elbow_silhouette(table: pd.DataFrame, k_elbow: int, k_sil: int, k_used: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(table["k"], table["inertia"], color=BLUE, marker="o", ms=6)
    elbow = table.loc[table["k"] == k_elbow]
    ax.scatter(elbow["k"], elbow["inertia"], s=140, facecolor="none", edgecolor=ORANGE, lw=2, zorder=3)
    ax.annotate(f"elbow: k = {k_elbow}", (k_elbow, elbow["inertia"].iloc[0]), xytext=(10, 10),
                textcoords="offset points", color=INK, fontsize=9)
    ax.set_title("Elbow method")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia (within-cluster squared distance)")

    ax = axes[1]
    colors = [BLUE if k == k_sil else TIER_LIGHT for k in table["k"]]
    ax.bar(table["k"], table["silhouette"], color=colors, width=0.6)
    for k, s in zip(table["k"], table["silhouette"]):
        if k in (k_sil, k_used):
            ax.text(k, s, f"{s:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_title("Silhouette score (higher = better separated)")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Mean silhouette")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, f"Elbow suggests k = {k_elbow}; silhouette peaks at k = {k_sil}; "
                          f"the four-quadrant hypothesis uses k = {k_used}.", fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def plot_k_distance(kdist: np.ndarray, eps: float, min_samples: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(np.arange(len(kdist)), kdist, color=BLUE)
    ax.axhline(eps, color=ORANGE, lw=1)
    ax.text(0, eps, f" eps = {eps:.3f} (knee)", color=INK, va="bottom", fontsize=9)
    ax.set_xlabel("Films, sorted by distance")
    ax.set_ylabel(f"Distance to {min_samples}th nearest neighbour")
    _titles(ax, "DBSCAN: choosing eps with the k-distance plot",
            "Flat part = films in dense regions. Steep rise = sparse outliers. The knee separates them.")
    return fig


# ---------------------------------------------------------------------------
# 4. The clusters themselves
# ---------------------------------------------------------------------------

def _scatter_groups(ax, frame: pd.DataFrame, label_col: str) -> None:
    """Grey/background groups first so coloured groups sit on top."""
    names = list(frame[label_col].value_counts().index)
    names.sort(key=lambda n: 0 if _style_for(n)[0] in (CONTEXT, CORE) else 1)
    for i, name in enumerate(names):
        d = frame[frame[label_col] == name]
        color, marker = _style_for(name, i)
        background = color in (CONTEXT, CORE)
        ax.scatter(d["reach_z"], d["averageRating"], s=3 if background else 9, color=color, marker=marker,
                   alpha=0.45 if background else 0.7, lw=0, label=f"{name} ({len(d):,})",
                   rasterized=True)
    ax.set_xlabel("Year-adjusted reach (reach_z)")
    ax.set_ylabel("averageRating")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, markerscale=2.5,
                    handletextpad=0.3, columnspacing=1.0)
    for h in leg.legend_handles:
        h.set_alpha(1)


def plot_cluster_maps(frame: pd.DataFrame, k_used: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
    _scatter_groups(axes[0], frame, "cluster_kmeans")
    axes[0].set_title(f"KMeans (k = {k_used})")
    _scatter_groups(axes[1], frame, "cluster_dbscan")
    axes[1].set_title("DBSCAN: dense core vs outliers")
    _scatter_groups(axes[2], frame, "quadrant_median")
    axes[2].set_title("Median split (baseline)")
    fig.tight_layout()
    return fig


def plot_hit_tiers(frame: pd.DataFrame, n_label: int = 8) -> plt.Figure:
    """The final label: Mega-hit / Hit / Not a hit, with the biggest mega-hits named."""
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {"Not a hit": (CONTEXT, 3, 0.35), "Hit": (TIER_LIGHT, 8, 0.7), "Mega-hit": (TIER_DARK, 14, 0.9)}
    for tier, (color, size, alpha) in styles.items():
        d = frame[frame["hit_tier"] == tier]
        ax.scatter(d["reach_z"], d["averageRating"], s=size, color=color, alpha=alpha, lw=0,
                   label=f"{tier} ({len(d):,}, {100 * len(d) / len(frame):.1f}%)", rasterized=True)
    # Number tags on the dots + a key, so titles never overlap each other.
    mega = frame[frame["hit_tier"] == "Mega-hit"].nlargest(n_label, "shrunk_rating")
    key = []
    for i, (_, row) in enumerate(mega.iterrows(), start=1):
        ax.annotate(str(i), (row["reach_z"], row["averageRating"]), xytext=(4, 4 if i % 2 else -11),
                    textcoords="offset points", fontsize=8, color=INK, fontweight="bold")
        key.append(f"{i}  {row['primaryTitle']} ({row['startYear']})")
    if key:
        ax.text(0.01, 0.02, "Top mega-hits by shrunk rating\n" + "\n".join(key), transform=ax.transAxes,
                fontsize=8, color=INK, va="bottom",
                bbox={"facecolor": SURFACE, "edgecolor": GRID, "boxstyle": "round,pad=0.5"})
    ax.set_xlabel("Year-adjusted reach (reach_z)")
    ax.set_ylabel("averageRating")
    _titles(ax, "The hit label: majority vote of KMeans, DBSCAN and median split",
            "Hit = at least 2 of 3 methods agree. Mega-hit = a Hit that DBSCAN also flags as an "
            "exceptional outlier.")
    leg = ax.legend(loc="lower right", markerscale=2)
    for h in leg.legend_handles:
        h.set_alpha(1)
    return fig


def plot_divisiveness(spread: pd.DataFrame, label_col: str) -> plt.Figure:
    """Box plot: spread of individual MovieLens ratings per cluster."""
    order = (spread.groupby(label_col)["rating_std"].median().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [spread.loc[spread[label_col] == o, "rating_std"] for o in order]
    box_kw = dict(patch_artist=True, widths=0.55, showfliers=False, medianprops={"color": INK, "lw": 1.5})
    try:  # matplotlib >= 3.10 uses `orientation`; older versions use `vert`
        bp = ax.boxplot(data, orientation="horizontal", **box_kw)
    except TypeError:
        bp = ax.boxplot(data, vert=False, **box_kw)
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor(_style_for(name)[0])
        patch.set_alpha(0.75)
        patch.set_edgecolor(SURFACE)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{o} (n={len(d)})" for o, d in zip(order, data)])
    ax.set_xlabel("Std. dev. of individual MovieLens ratings (higher = more divisive)")
    _titles(ax, "Is 'Popular but Divisive' really divisive?",
            "Films in both IMDb and MovieLens 20M (released up to 2015), each with enough ratings.")
    ax.grid(axis="y", visible=False)
    return fig


# ---------------------------------------------------------------------------
# 5. Trends
# ---------------------------------------------------------------------------

def _small_multiples(n: int, ncols: int = 4, height: float = 2.6):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, height * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def _smooth(d: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Centred rolling mean, to show the trend rather than single-year noise."""
    return d[col].rolling(window, center=True, min_periods=1).mean()


def _ranked_genres(trends: pd.DataFrame, col: str) -> list[str]:
    """Genres by average share, largest first, without the residual "Other" bucket."""
    order = trends.groupby("genre")[col].mean().sort_values(ascending=False).index
    return [g for g in order if g != "Other"]


def plot_imdb_genre_trends(trends: pd.DataFrame, window: int = 5) -> plt.Figure:
    """Per genre: share of films made (grey) vs share of audience votes (blue)."""
    genres = _ranked_genres(trends, "vote_share")
    fig, axes = _small_multiples(len(genres), ncols=3)
    for ax, genre in zip(axes, genres):
        d = trends[trends["genre"] == genre].sort_values("year")
        ax.plot(d["year"], _smooth(d, "film_share", window), color=CONTEXT, lw=1.8, label="Share of films made")
        ax.plot(d["year"], _smooth(d, "vote_share", window), color=BLUE, lw=2, label="Share of audience votes")
        ax.set_title(genre, fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Supply vs demand by genre (IMDb): blue above grey = the genre over-performs",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.supylabel(f"% of the year's total ({window}-year rolling mean)", fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def plot_movielens_genre_trends(trends: pd.DataFrame, window: int = 3) -> plt.Figure:
    genres = _ranked_genres(trends, "rating_share")
    fig, axes = _small_multiples(len(genres), ncols=3)
    for ax, genre in zip(axes, genres):
        d = trends[trends["genre"] == genre].sort_values("year")
        ax.plot(d["year"], _smooth(d, "rating_share", window), color=BLUE)
        ax.set_title(genre, fontsize=10)
    fig.suptitle("What MovieLens users rated, by genre (share of each year's ratings)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.supylabel(f"% of the year's ratings ({window}-year rolling mean)", fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def plot_movielens_yearly(yearly: pd.DataFrame) -> plt.Figure:
    panels = [("n_ratings", "Ratings per year", _thousands),
              ("mean_rating", "Average rating given (0.5-5)", None),
              ("median_film_age", "Median age of the film being rated (years)", None),
              ("top1pct_share", "% of ratings going to the top 1% of films", None)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (col, title, fmt) in zip(axes, panels):
        ax.plot(yearly["year"], yearly[col], color=BLUE, marker="o", ms=3)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        if fmt is not None:
            ax.yaxis.set_major_formatter(fmt)
    fig.suptitle(f"How MovieLens users behave over time ({yearly['year'].min()}-{yearly['year'].max()})",
                 x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig
