"""Part B figures, in the same visual system as Part A (see plots.py)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from matplotlib.ticker import PercentFormatter

from ..plots import AQUA, BLUE, CONTEXT, INK, INK_2, ORANGE, _titles
from .scorer import MODEL_LABELS

MODEL_COLORS = {"logistic": CONTEXT, "hgb": ORANGE, "tabpfn": AQUA}


def _pr(ax, y, p, color, label, lw=2.0):
    prec, rec, _ = precision_recall_curve(y, p)
    # drop sklearn's sentinel point (recall 0, precision 1), which draws a fake vertical line
    ax.plot(rec[:-1], prec[:-1], color=color, lw=lw, label=label, drawstyle="steps-post")


def plot_backtest(bt) -> plt.Figure:
    """Precision-recall on the test years, every candidate; the chosen one in blue."""
    pred, test = bt.predictions, bt.test.set_index("model")
    base = pred["y"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name in test.index:
        chosen = name == bt.chosen
        color = BLUE if chosen else MODEL_COLORS.get(name, CONTEXT)
        _pr(ax, pred["y"], pred[f"p_{name}"], color,
            f"{MODEL_LABELS.get(name, name)}{' (chosen)' if chosen else ''}: "
            f"PR-AUC {test.loc[name, 'pr_auc']:.2f} ({test.loc[name, 'lift']:.1f}x base rate)",
            lw=2.4 if chosen else 1.4)
    ax.axhline(base, color=INK_2, lw=0.8)
    ax.text(0.01, base, f"random guessing = base rate {100 * base:.1f}%", color=INK_2, fontsize=8,
            va="bottom", ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Recall: share of real AAA hits the model finds")
    ax.set_ylabel("Precision: share of flagged films that were AAA hits")
    t0, t1 = bt.years["test"]
    _titles(ax, f"Backtest: predicting AAA hits released {t0}-{t1}, trained on films up to {bt.years['train_end']}",
            "Model chosen on earlier validation years, so the test years were never used to pick it.")
    ax.legend(loc="upper right", fontsize=8.5)
    return fig


def plot_calibration(bt, n_bins: int = 8) -> plt.Figure:
    """Does a predicted 30% come true about 30% of the time? (chosen model, test years)"""
    pred = bt.predictions
    p, y = pred[f"p_{bt.chosen}"].to_numpy(), pred["y"].to_numpy()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    hi = max(df["p"].max(), df["y"].max()) * 1.15
    ax.plot([0, hi], [0, hi], color=INK_2, lw=0.8)
    ax.plot(df["p"], df["y"], color=BLUE, marker="o", ms=7)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Predicted chance of a AAA hit (bin average)")
    ax.set_ylabel("Observed share of AAA hits")
    _titles(ax, f"Calibration of the judge ({MODEL_LABELS.get(bt.chosen, bt.chosen)}), test years",
            f"Each point = one of {len(df)} equal-size groups of test films (~{int(df['n'].median())} each). "
            "On the grey line, predictions mean what they say.")
    return fig


def plot_group_importance(imp: pd.DataFrame) -> plt.Figure:
    d = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.6))
    ax.barh(d["group"], d["pr_auc_drop"], xerr=d["sd"], color=BLUE, height=0.6,
            error_kw={"ecolor": INK_2, "lw": 1})
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in test PR-AUC when this information is shuffled")
    _titles(ax, "What the judge relies on", "Bigger drop = more important for spotting AAA hits.")
    return fig


def plot_loop_progress(prog: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    styles = {"loop": (BLUE, "Generate, score, refine (feedback)"), "no_feedback": (CONTEXT, "Same calls, no feedback (control)")}
    for arm, (color, label) in styles.items():
        d = prog[prog["arm"] == arm]
        if len(d):
            ax.plot(d["round"], d["best_so_far"], color=color, marker="o", lw=2.2, label=label)
            ax.text(d["round"].iloc[-1], d["best_so_far"].iloc[-1], f"  {d['best_so_far'].iloc[-1]:.1%}",
                    color=INK, fontsize=9, va="center")
    ax.set_xticks(sorted(prog["round"].unique()) if len(prog) else [])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Round")
    ax.set_ylabel("Best P(AAA hit) so far (mean over islands)")
    _titles(ax, "Does feedback help the generator find stronger concepts?",
            "Only concepts that passed the novelty filter and look like past films count.")
    ax.legend(loc="lower right")
    return fig


def plot_concept_map(records: pd.DataFrame, final: pd.DataFrame, threshold: float) -> plt.Figure:
    """Every concept: how close it is to an existing film vs. its score."""
    fig, ax = plt.subplots(figsize=(10, 6))
    rej = records[records["status"] == "rejected_similar"]
    n_repeat = int((records["status"] == "rejected_repeat").sum())
    sc = records[records["status"] == "scored"]
    if len(rej):  # rejected concepts have no score: drawn on the floor of the chart
        ax.scatter(rej["nearest_sim"], np.zeros(len(rej)), s=18, marker="x", color=ORANGE,
                   label=f"Rejected: too close to an existing film ({len(rej)})")
    for arm, color, label in (("no_feedback", CONTEXT, "Control arm"), ("loop", BLUE, "Feedback loop")):
        d = sc[sc["arm"] == arm]
        fam = d["in_distribution"].astype(bool)
        ax.scatter(d.loc[fam, "nearest_sim"], d.loc[fam, "p_aaa_hit"], s=22, color=color, alpha=0.75,
                   lw=0, label=f"{label} ({int(fam.sum())})")
        if (~fam).any():
            ax.scatter(d.loc[~fam, "nearest_sim"], d.loc[~fam, "p_aaa_hit"], s=26, facecolor="none",
                       edgecolor=color, lw=1, label=f"{label}, unfamiliar ({int((~fam).sum())})")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axvline(threshold, color=INK_2, lw=0.9)
    ax.text(threshold, ax.get_ylim()[1], " novelty limit", color=INK_2, fontsize=8, va="top")
    for i, (_, r) in enumerate(final.iterrows(), start=1):
        ax.annotate(f"{i}. {r['title']}", (r["nearest_sim"], r["p_aaa_hit"]), xytext=(6, 6 if i % 2 else -14),
                    textcoords="offset points", fontsize=9, fontweight="bold", color=INK)
    ax.set_xlabel("Similarity to the closest existing film (plot embedding, cosine)")
    ax.set_ylabel("P(AAA hit) from the judge")
    _titles(ax, "All generated concepts: originality vs. predicted chance of a AAA hit",
            "Numbered: the final portfolio (best familiar, original concept per island). "
            f"Not shown: {n_repeat} near-repeats of earlier concepts.")
    ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_llm_vs_judge(preds: pd.DataFrame, table: pd.DataFrame) -> plt.Figure:
    ok = preds["p_llm"].notna()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _pr(ax, preds.loc[ok, "y"], preds.loc[ok, "p_judge"], BLUE,
        f"{table['model'].iloc[1]}: PR-AUC {table['pr_auc'].iloc[1]:.2f}")
    _pr(ax, preds.loc[ok, "y"], preds.loc[ok, "p_llm"], ORANGE,
        f"{table['model'].iloc[0]}: PR-AUC {table['pr_auc'].iloc[0]:.2f}")
    ax.axhline(preds.loc[ok, "y"].mean(), color=INK_2, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    n, k = int(ok.sum()), int(preds.loc[ok, "y"].sum())
    _titles(ax, f"LLM few-shot vs. the judge on films released after {int(table['cutoff_year'].iloc[0])}",
            f"Same {n} films ({k} AAA hits, all included; non-hits sampled), same pre-release information, "
            f"titles hidden from the LLM." + (" Too few hits to conclude anything." if k < 5 else ""))
    ax.legend(loc="upper right", fontsize=8.5)
    return fig
