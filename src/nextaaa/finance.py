"""Money: inflation adjustment (optional), ROI, break-even and budget tiers.

ROI multiple
------------
    roi_multiple = worldwide box office / production budget

We judge it against a break-even of 2.5x, the common industry rule of thumb:
theatres keep roughly half of ticket sales, and marketing (prints and
advertising) costs about as much as production, so a studio needs ~2.5x the
production budget at the box office before it makes money. TMDB revenue is
box office only (no streaming or home video), so 2.5x is conservative.

    profitable            roi_multiple >= 2.5
    above_breakeven_usd   revenue - 2.5 x budget  (dollars above/below break-even)

ROI is a ratio of two same-year dollar figures, so inflation cancels out of it.
Revenue and budget themselves are compared within release year in the
clustering, which also removes inflation. A CPI file is therefore only needed
to SHOW real (base-year) dollars on slides and to define the AAA-scale budget
tier consistently across decades.

Inflation (optional)
--------------------
Put the US CPI series from FRED in data/cpi/:
    https://fred.stlouisfed.org/series/CPIAUCSL  ->  "Download" -> CSV
Both FRED layouts work ("observation_date,CPIAUCSL" and the older "DATE,CPIAUCSL"),
as does any two-column CSV of (year, cpi). Monthly values are averaged per year.
Without the file everything still runs; real-dollar columns are left empty and
the budget tiers fall back to nominal dollars (with a warning).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import CPI_DOWNLOAD_URL, AnalysisConfig, Paths

logger = logging.getLogger(__name__)

BUDGET_TIERS = [  # (label, lower bound in US$); the top tier uses cfg.aaa_budget_usd
    ("< $10M", 0),
    ("$10-40M", 10e6),
    ("$40-100M", 40e6),
]
AAA_TIER = "AAA-scale ($100M+)"


# ---------------------------------------------------------------------------
# CPI
# ---------------------------------------------------------------------------

def parse_cpi(df: pd.DataFrame) -> pd.Series:
    """Turn a FRED-style or (year, cpi) CSV into an annual CPI Series indexed by year."""
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("observation_date") or cols.get("date")
    value_col = cols.get("cpiaucsl") or cols.get("cpi") or cols.get("value")
    if date_col and value_col:
        years = pd.to_datetime(df[date_col], errors="coerce").dt.year
    elif "year" in cols and len(df.columns) == 2:
        years = pd.to_numeric(df[cols["year"]], errors="coerce")
        value_col = [c for c in df.columns if c != cols["year"]][0]
    else:
        raise ValueError(f"Unrecognised CPI file layout: columns {list(df.columns)}")
    values = pd.to_numeric(df[value_col], errors="coerce")
    annual = values.groupby(years).mean().dropna()
    annual.index = annual.index.astype(int)
    return annual.sort_index()


def load_cpi(paths: Paths) -> pd.Series | None:
    """Annual CPI from the first CSV in data/cpi/, or None if absent."""
    files = sorted(paths.cpi_dir.glob("*.csv")) if paths.cpi_dir.exists() else []
    if not files:
        logger.warning("No CPI file in %s: showing nominal dollars only. "
                       "Optional download: %s", paths.cpi_dir, CPI_DOWNLOAD_URL)
        return None
    cpi = parse_cpi(pd.read_csv(files[0]))
    logger.info("CPI loaded from %s: %d-%d", files[0].name, cpi.index.min(), cpi.index.max())
    return cpi


def deflator(cpi: pd.Series, base_year: int) -> tuple[pd.Series, int]:
    """Multipliers that convert each year's dollars into base-year dollars.

    If the base year is not in the CPI file (e.g. an old download), the latest
    available year is used instead, and returned so the slides can say so.
    """
    if base_year not in cpi.index:
        new_base = int(cpi.index.max())
        logger.warning("CPI has no %d; using %d dollars instead.", base_year, new_base)
        base_year = new_base
    return cpi.loc[base_year] / cpi, base_year


# ---------------------------------------------------------------------------
# Finance columns
# ---------------------------------------------------------------------------

def budget_tier(budget: pd.Series, aaa_budget: float) -> pd.Series:
    """Ordered budget tiers: < $10M, $10-40M, $40-100M, AAA-scale."""
    edges = [b for _, b in BUDGET_TIERS] + [aaa_budget, np.inf]
    labels = [lab for lab, _ in BUDGET_TIERS] + [AAA_TIER]
    return pd.cut(budget, bins=edges, labels=labels, right=False)


def add_finance(df: pd.DataFrame, cfg: AnalysisConfig, cpi: pd.Series | None = None) -> pd.DataFrame:
    """Add ROI, break-even and (if CPI is given) real-dollar columns.

    Needs: startYear, budget, revenue (NaN where unknown). Adds:
        has_finance          both figures known, and year >= cfg.finance_min_year
        roi_multiple, log_roi, profitable, above_breakeven_usd
        budget_real, revenue_real   base-year dollars (NaN without CPI)
        budget_for_tier             real if available, else nominal
        budget_tier                 ordered category
    """
    out = df.copy()
    out["has_finance"] = (out["budget"].notna() & out["revenue"].notna()
                          & (out["startYear"] >= cfg.finance_min_year))
    fin = out["has_finance"]
    out["roi_multiple"] = (out["revenue"] / out["budget"]).where(fin)
    out["log_roi"] = np.log10(out["roi_multiple"])
    # 1.0 / 0.0 (NaN without data), so groupby().mean() gives a share directly.
    out["profitable"] = (out["roi_multiple"] >= cfg.breakeven_multiple).astype(float).where(fin)
    out["above_breakeven_usd"] = (out["revenue"] - cfg.breakeven_multiple * out["budget"]).where(fin)

    if cpi is not None:
        factor, base = deflator(cpi, cfg.cpi_base_year)
        f = out["startYear"].map(factor)
        out["budget_real"] = out["budget"] * f
        out["revenue_real"] = out["revenue"] * f
        out.attrs["dollar_basis"] = f"{base} US$"
        missing = fin & f.isna()
        if missing.any():
            logger.warning("%d films fall outside the CPI years; their tier uses nominal $.",
                           int(missing.sum()))
        out["budget_for_tier"] = out["budget_real"].fillna(out["budget"])
    else:
        out["budget_real"] = np.nan
        out["revenue_real"] = np.nan
        out.attrs["dollar_basis"] = "nominal US$"
        out["budget_for_tier"] = out["budget"]
        if fin.any():
            logger.warning("No CPI: the AAA-scale tier (>= $%.0fM) uses nominal dollars, "
                           "so older films are under-counted.", cfg.aaa_budget_usd / 1e6)
    out["budget_tier"] = budget_tier(out["budget_for_tier"].where(fin), cfg.aaa_budget_usd)
    return out


def budget_tier_economics(df: pd.DataFrame, verdict_col: str | None = None) -> pd.DataFrame:
    """Per budget tier: how often films make money, and how much.

    Answers the studio question "does spending more make success more likely?".
    lost_money_pct is the share that did not even recoup the production budget
    at the box office (ROI < 1x), a stricter failure than missing break-even.
    """
    fin = df[df["has_finance"]]
    g = fin.groupby("budget_tier", observed=False)
    out = pd.DataFrame({
        "films": g.size(),
        "median_budget": g["budget_for_tier"].median(),
        "median_roi": g["roi_multiple"].median(),
        "profitable_pct": 100 * g["profitable"].mean(),
        "lost_money_pct": 100 * g["roi_multiple"].apply(lambda s: (s < 1).mean()),
    })
    if verdict_col is not None:
        out["aaa_hit_pct"] = 100 * g[verdict_col].apply(lambda s: (s == "AAA hit").mean())
    return out
