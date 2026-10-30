"""Compare every film with films released the same year.

Why
---
"Is this a hit?" only makes sense relative to its era:
  * a 1985 film has had 40 years to collect IMDb votes, and IMDb had fewer users then;
  * $100M of box office in 1980 is worth far more than in 2020 (inflation);
  * typical ratings and typical ROI drift from decade to decade.

Comparing each film with its own release year removes all of this at once, with
no inflation table needed. That is what makes one model work for both the
by-year hit lists and the all-time list: an all-time hit is a film that
dominated its own year more than any other film dominated theirs.

How
---
For each feature value x of a film released in year y:

    z   = (x - median of the reference films) / (IQR of the reference films / 1.349)
    pct = percentile of x among the reference films (0-100, mid-rank for ties)

The reference films are those released in year y. If year y has too few films
(common for budget data before the 1990s), the window widens to y +/- 1,
y +/- 2, ... up to `max_window` years, until there are at least `min_ref`
films. Years that still have too few are left as NaN (excluded).

z keeps the long right tail (true hits ARE far out), which the outlier detectors
need; pct is bounded and easy to explain ("99th percentile of its year").
Median and IQR rather than mean and standard deviation, so a few
mega-blockbusters cannot distort what "typical" means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IQR_TO_SIGMA = 1.349  # IQR of a standard normal distribution


def year_relative(values: pd.Series, years: pd.Series, min_ref: int,
                  max_window: int) -> pd.DataFrame:
    """Robust z-score and percentile of each value within its (widened) year.

    Returns a DataFrame aligned to `values.index` with columns:
        z, pct        the two scores (NaN if no reference set was big enough)
        ref_window    +/- years used for that film's release year (0 = same year)
        ref_n         number of reference films
    """
    values = pd.to_numeric(values, errors="coerce")
    ok = values.notna() & years.notna()
    v = values[ok].to_numpy(dtype=float)
    y = years[ok].to_numpy(dtype=int)

    z = np.full(len(v), np.nan)
    pct = np.full(len(v), np.nan)
    window = np.full(len(v), np.nan)
    ref_n = np.full(len(v), np.nan)

    for year in np.unique(y):
        for w in range(max_window + 1):
            ref = np.sort(v[np.abs(y - year) <= w])
            if len(ref) >= min_ref:
                break
        else:  # never reached min_ref: leave this year's films as NaN
            continue
        mask = y == year
        med = np.median(ref)
        q75, q25 = np.percentile(ref, [75, 25])
        scale = (q75 - q25) / IQR_TO_SIGMA
        if not scale > 0:  # all identical values: fall back to std, then 1
            scale = ref.std() or 1.0
        z[mask] = (v[mask] - med) / scale
        below = np.searchsorted(ref, v[mask], side="left")
        at_or_below = np.searchsorted(ref, v[mask], side="right")
        pct[mask] = 100 * (below + 0.5 * (at_or_below - below)) / len(ref)
        window[mask] = w
        ref_n[mask] = len(ref)

    out = pd.DataFrame(index=values.index, columns=["z", "pct", "ref_window", "ref_n"], dtype=float)
    out.loc[ok, "z"] = z
    out.loc[ok, "pct"] = pct
    out.loc[ok, "ref_window"] = window
    out.loc[ok, "ref_n"] = ref_n
    return out
