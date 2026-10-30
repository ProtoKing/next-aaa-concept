"""Figures added in v2 (money, five methods, by-year and all-time hits).

Same visual system as plots.py (imported from there): grey for context,
colour only where it carries meaning, one fixed colour per verdict across every
chart, thin marks, solid hairline grid, no dual axes.

Verdict colours (fixed everywhere):
    AAA hit                 blue
    Loved, unprofitable     aqua
    Profitable, not loved   orange
    Neither                 grey
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ensemble import METHOD_LABELS, METHODS, Stage
from .plots import (AQUA, BLUE, CONTEXT, GRID, INK, INK_2, ORANGE, SURFACE, TIER_DARK,
                    TIER_LIGHT, _thousands, _titles, plot_elbow_silhouette, plot_k_distance)

VERDICT_STYLE = {  # colour, marker, size, alpha, draw order (higher on top)
    "Neither": (CONTEXT, ".", 4, 0.35, 0),
    "Profitable, not loved": (ORANGE, "s", 12, 0.75, 2),
    "Loved, unprofitable": (AQUA, "^", 14, 0.75, 2),
    "AAA hit": (BLUE, "o", 20, 0.9, 3),
}
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _money(v: float) -> str:
    return f"${v / 1e9:.1f}B" if v >= 1e9 else f"${v / 1e6:.0f}M" if v >= 1e6 else f"${v / 1e3:.0f}k"


def _key_box(ax, lines: list[str], title: str, loc=(0.01, 0.02)) -> None:
    if lines:
        ax.text(*loc, title + "\n" + "\n".join(lines), transform=ax.transAxes, fontsize=8,
                color=INK, va="bottom", ha="left",
                bbox={"facecolor": SURFACE, "edgecolor": GRID, "boxstyle": "round,pad=0.5"})


def _legend(ax, **kw):
    leg = ax.legend(markerscale=1.8, **kw)
    for h in leg.legend_handles:
        h.set_alpha(1)
    return leg


# ---------------------------------------------------------------------------
# 2023 actors, three definitions
# ---------------------------------------------------------------------------

def plot_actors_three_way(pop: pd.DataFrame, box: pd.DataFrame | None, year: int,
                          top_n: int = 10) -> plt.Figure:
    panels = [(pop.nsmallest(top_n, "rank_reach"), "reach", "Reach", "Billing-weighted votes"),
              (pop.nsmallest(top_n, "rank_goodwill"), "net_goodwill", "Net goodwill",
               "Weighted votes x (shrunk rating - average)")]
    if box is not None and len(box):
        panels.append((box.nsmallest(top_n, "rank_box_office"), "box_office", "Box office",
                       "Billing-weighted worldwide revenue (US$)"))
    tops = [set(d["nconst"]) for d, *_ in panels]
    times_top = {n: sum(n in t for t in tops) for t in tops for n in t}
    shade = {1: TIER_LIGHT, 2: BLUE, 3: TIER_DARK}

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.6))
    for ax, (d, col, title, xlabel) in zip(np.atleast_1d(axes), panels):
        d = d.iloc[::-1]
        ax.barh(d["primaryName"], d[col], color=[shade[times_top[n]] for n in d["nconst"]], height=0.7)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _money(v) if v else "0")
                                     if col == "box_office" else _thousands)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.suptitle(f"Most popular actor of {year}: three definitions", x=0.01, y=0.985, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    n = len(panels)
    fig.text(0.01, 0.925, f"Shade = in how many of the {n} top-{top_n} lists the actor appears "
                          f"(darkest = all {n}).", fontsize=9, color=INK_2)
    return fig


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

def plot_budget_vs_revenue(df: pd.DataFrame, breakeven: float, n_label: int = 6) -> plt.Figure:
    """Every film with money data: budget vs box office, coloured by verdict."""
    fin = df[df["has_finance"]]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for verdict, (color, marker, size, alpha, z) in sorted(VERDICT_STYLE.items(), key=lambda kv: kv[1][4]):
        d = fin[fin["verdict"] == verdict]
        ax.scatter(d["budget"], d["revenue"], s=size, color=color, marker=marker, alpha=alpha, lw=0,
                   label=f"{verdict} ({len(d):,})", zorder=z, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo = max(fin["budget"].min(), 1e4)
    hi = fin["budget"].max()
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color=INK_2, lw=0.8)
    ax.plot(xs, breakeven * xs, color=INK, lw=1.2)
    ax.text(hi, hi, "  box office = budget", fontsize=8, color=INK_2, va="center")
    ax.text(hi, breakeven * hi, f"  break-even ({breakeven:g}x)", fontsize=8, color=INK, va="center")
    top = fin[fin["is_aaa_hit"]].nlargest(n_label, "hit_score")
    lines = []
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        ax.annotate(str(i), (r["budget"], r["revenue"]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=INK, zorder=5)
        lines.append(f"{i}  {r['primaryTitle']} ({r['startYear']}): {r['roi_multiple']:.1f}x")
    _key_box(ax, lines, "Top AAA hits by hit score", loc=(0.55, 0.02))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _money(v)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _money(v)))
    ax.set_xlabel("Production budget (nominal US$, log scale)")
    ax.set_ylabel("Worldwide box office (nominal US$, log scale)")
    _titles(ax, "Budget vs. box office: who cleared break-even, and who also won audiences",
            f"Films above the dark line made at least {breakeven:g}x their budget, the industry "
            "rule of thumb for profit.")
    _legend(ax, loc="upper left")
    return fig


def plot_budget_tiers(tiers: pd.DataFrame) -> plt.Figure:
    """Three small panels per budget tier: break-even rate, AAA-hit rate, median ROI."""
    cols = [("profitable_pct", "% that cleared break-even", "{:.0f}%"),
            ("aaa_hit_pct", "% that became AAA hits", "{:.1f}%"),
            ("median_roi", "Median ROI multiple", "{:.1f}x")]
    cols = [c for c in cols if c[0] in tiers.columns]
    labels = [f"{t}\n(n={int(n):,})" for t, n in zip(tiers.index.astype(str), tiers["films"])]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.2), sharey=True)
    for ax, (col, title, fmt) in zip(np.atleast_1d(axes), cols):
        y = np.arange(len(tiers))
        ax.barh(y, tiers[col], color=BLUE, height=0.6)
        for yi, v in zip(y, tiers[col]):
            if pd.notna(v):
                ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=9, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, float(np.nanmax(tiers[col])) * 1.25 or 1)
    fig.suptitle("Does spending more make a hit more likely?", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# The five methods
# ---------------------------------------------------------------------------

def plot_stage_k(stage_result) -> plt.Figure:
    """Elbow + silhouette for one stage (reuses the v1 chart)."""
    r = stage_result
    fig = plot_elbow_silhouette(r.kmeans.table, r.kmeans.k_elbow, r.kmeans.k_silhouette, r.k_used)
    fig.suptitle(f"{r.stage.title} stage: how many KMeans clusters?", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_stage_kdist(stage_result, min_samples: int) -> plt.Figure:
    r = stage_result
    fig = plot_k_distance(r.kdist, r.eps, min_samples)
    fig.axes[0].set_title(f"{r.stage.title} stage, DBSCAN: choosing eps with the k-distance plot", pad=26)
    return fig


def plot_method_maps(stage_result, hit_min: int, mega_min: int) -> plt.Figure:
    """One panel per method showing whom it votes a hit, plus the final vote."""
    r, st = stage_result, stage_result.stage
    df = r.frame
    p = st.key
    hit_corner = df[f"{p}_corner"] == st.hit_name
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, m in zip(axes, METHODS):
        vote = df[f"{p}_vote_{m}"].astype(bool)
        ax.scatter(df[st.x], df[st.y], s=3, color=CONTEXT, alpha=0.35, lw=0, rasterized=True,
                   label="No hit vote")
        if f"{p}_unusual_{m}" in df:
            # Unusual but outside the hit corner (e.g. spectacular flops): shown so
            # the viewer sees the method's whole opinion, not just its hit votes.
            d = df[df[f"{p}_unusual_{m}"] & ~hit_corner]
            ax.scatter(d[st.x], d[st.y], s=10, color=ORANGE, marker="s", alpha=0.7, lw=0,
                       label=f"Unusual, other corner ({len(d):,})", rasterized=True)
        d = df[vote]
        ax.scatter(d[st.x], d[st.y], s=10, color=BLUE, alpha=0.8, lw=0,
                   label=f"Hit vote ({len(d):,})", rasterized=True)
        ax.set_title(METHOD_LABELS[m], fontsize=11)
        _legend(ax, loc="lower right", fontsize=8)
    ax = axes[5]
    ax.scatter(df[st.x], df[st.y], s=3, color=CONTEXT, alpha=0.35, lw=0, rasterized=True)
    hit = df[f"{p}_hit"] & ~df[f"{p}_mega"]
    mega = df[f"{p}_mega"]
    ax.scatter(df.loc[hit, st.x], df.loc[hit, st.y], s=10, color=TIER_LIGHT, lw=0,
               label=f"Hit: {hit_min}-{mega_min - 1} votes ({int(hit.sum()):,})", rasterized=True)
    ax.scatter(df.loc[mega, st.x], df.loc[mega, st.y], s=14, color=TIER_DARK, lw=0,
               label=f"Mega: all {mega_min} ({int(mega.sum()):,})", rasterized=True)
    ax.set_title(f"Majority vote (>= {hit_min} of 5)", fontsize=11, fontweight="bold")
    _legend(ax, loc="lower right", fontsize=8)
    for ax in axes:
        ax.axhline(0, color=GRID, lw=1)
        ax.axvline(0, color=GRID, lw=1)
    fig.supxlabel(st.x_label, fontsize=10, color=INK_2)
    fig.supylabel(st.y_label, fontsize=10, color=INK_2)
    fig.suptitle(f"{st.title} stage: whom each method calls a '{st.hit_name}'", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def _heatmap(ax, mat: pd.DataFrame, fmt: str = "{:.2f}", vmin: float = 0, vmax: float = 1) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUES)
    ax.imshow(mat.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.notna(v):
                dark = (v - vmin) / (vmax - vmin) > 0.55
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                        color=SURFACE if dark else INK)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_agreement(audience, commercial=None) -> plt.Figure:
    results = [r for r in (audience, commercial) if r is not None]
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.2))
    for ax, r in zip(np.atleast_1d(axes), results):
        _heatmap(ax, r.agreement)
        ax.set_title(f"{r.stage.title} stage", fontsize=11)
    fig.suptitle("How much do the five methods agree? (Cohen's kappa: 1 = identical, 0 = chance)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Verdicts and hit lists
# ---------------------------------------------------------------------------

def plot_verdict_map(df: pd.DataFrame) -> plt.Figure:
    """Audience score vs commercial score (within-year percentiles), by verdict."""
    fin = df[df["has_finance"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    for verdict, (color, marker, size, alpha, z) in sorted(VERDICT_STYLE.items(), key=lambda kv: kv[1][4]):
        d = fin[fin["verdict"] == verdict]
        ax.scatter(d["commercial_score"], d["audience_score"], s=size, color=color, marker=marker,
                   alpha=alpha, lw=0, label=f"{verdict} ({len(d):,})", zorder=z, rasterized=True)
    # No film labels here: the top hits all sit near (100, 100) and would overlap.
    # They are named on the budget chart and the all-time heatmap instead.
    for v in (50,):
        ax.axhline(v, color=GRID, lw=1)
        ax.axvline(v, color=GRID, lw=1)
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Commercial score: mean percentile of ROI and box office within its year")
    ax.set_ylabel("Audience score: mean percentile of rating and votes within its year")
    _titles(ax, "Two gates: winning the audience and winning at the box office",
            "Only films with budget and revenue data. Verdicts come from the five-method votes, "
            "so edges are soft.")
    _legend(ax, loc="upper left")
    return fig


def plot_hits_per_year(yearly: pd.DataFrame, min_money_films: int = 20) -> plt.Figure:
    """Is it getting harder to make a hit? Two rates over time, same x-axis."""
    d = yearly[yearly["films_with_money"] >= min_money_films].sort_values("year")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, col, title in ((axes[0], "profitable_pct", "% of films with money data that cleared break-even"),
                           (axes[1], "aaa_hit_pct_of_money_films", "% that became AAA hits")):
        ax.plot(d["year"], d[col], color=CONTEXT, lw=1, marker="o", ms=2.5)
        ax.plot(d["year"], d[col].rolling(5, center=True, min_periods=2).mean(), color=BLUE, lw=2.2)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
    axes[1].set_xlabel("Release year")
    fig.suptitle("Hit rates by release year (grey: each year, blue: 5-year rolling mean)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, f"Years with fewer than {min_money_films} films with money data are hidden.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def plot_all_time_heatmap(top: pd.DataFrame, n: int = 20) -> plt.Figure:
    """The all-time AAA hits: each film's within-year percentile on every axis."""
    d = top.head(n)
    mat = pd.DataFrame({
        "Quality": d["pct_quality"].to_numpy(), "Reach": d["pct_reach"].to_numpy(),
        "ROI": d["pct_roi"].to_numpy(), "Box office": d["pct_revenue"].to_numpy(),
        "Hit score": d["hit_score"].to_numpy(),
    }, index=[f"{r} {t} ({y})" for r, t, y in zip(d["rank_all_time"], d["primaryTitle"], d["startYear"])])
    fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(mat) + 1.8))
    # Colour range from the data: these films are all near the top, so a fixed
    # 0-100 (or 50-100) scale would paint every cell the same dark blue.
    vmin = float(np.floor(np.nanmin(mat.to_numpy()) / 5) * 5) - 5
    _heatmap(ax, mat, fmt="{:.0f}", vmin=max(0.0, vmin), vmax=100)
    ax.set_xticklabels(mat.columns, rotation=0, ha="center")
    _titles(ax, "All-time AAA hits: who dominated their own year the most",
            f"Cells = percentile within the film's release year (100 = best of its year). "
            f"Shading runs from {max(0.0, vmin):.0f} to 100.")
    fig.tight_layout()
    return fig


def plot_genre_economics(genres: pd.DataFrame, breakeven: float, since: int, top_n: int = 15) -> plt.Figure:
    d = genres.head(top_n).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(d) + 2), sharey=True)
    y = np.arange(len(d))
    for ax, col, title, fmt in ((axes[0], "aaa_hit_pct", "% of films that became AAA hits", "{:.1f}%"),
                                (axes[1], "median_roi", "Median ROI multiple", "{:.1f}x")):
        ax.hlines(y, 0, d[col], color=GRID, lw=1.5)
        ax.scatter(d[col], y, color=BLUE, s=40, zorder=3)
        for yi, v in zip(y, d[col]):
            ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(float(d[col].max()), breakeven if col == "median_roi" else 0) * 1.25)
    axes[1].axvline(breakeven, color=INK_2, lw=1)
    axes[1].text(breakeven, len(d) - 0.4, f" break-even {breakeven:g}x", fontsize=8, color=INK_2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{g} (n={int(n):,})" for g, n in zip(d.index, d["films"])])
    fig.suptitle(f"Which genres turn into AAA hits? Films released {since} onward with money data",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig
