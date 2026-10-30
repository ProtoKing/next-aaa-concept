"""All paths and tunable parameters, in one place.

Why a single config module?
    Every "magic number" in the analysis (minimum votes, the year range, the
    shrinkage strength, the k values to try) lives here with a comment explaining
    the choice. When a stakeholder asks "why 1,000 votes?", there is exactly one
    place to look, and one place to change it.

How paths are resolved (first match wins):
    1. Arguments you pass explicitly (e.g. from the command line).
    2. Environment variables: NEXTAAA_IMDB_DIR, NEXTAAA_MOVIELENS_DIR,
       NEXTAAA_CACHE_DIR, NEXTAAA_OUTPUT_DIR, NEXTAAA_TMDB_DIR, NEXTAAA_CPI_DIR.
    3. Defaults relative to the project root: data/imdb, data/ml-20m,
       data/cache, outputs, data/tmdb, data/cpi.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path

# The project root is two levels above this file: <root>/src/nextaaa/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where to download the data from (shown in error messages if files are missing).
IMDB_DOWNLOAD_URL = "https://datasets.imdbws.com/"
MOVIELENS_DOWNLOAD_URL = "https://grouplens.org/datasets/movielens/20m/"
TMDB_DOWNLOAD_URL = "https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies"
CPI_DOWNLOAD_URL = "https://fred.stlouisfed.org/series/CPIAUCSL"  # "Download" -> CSV
max_parallel_calls: int = 6  # lower to 2-3 if you hit rate limits (429s)

def _env_path(var: str, default: Path) -> Path:
    """Read a path from an environment variable, falling back to a default."""
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Paths:
    """Where input data lives and where outputs are written."""

    imdb_dir: Path
    movielens_dir: Path
    cache_dir: Path
    output_dir: Path
    # v2 inputs (both optional: the code runs without them, with reduced output)
    tmdb_dir: Path = PROJECT_ROOT / "data" / "tmdb"
    cpi_dir: Path = PROJECT_ROOT / "data" / "cpi"

    @classmethod
    def resolve(
        cls,
        imdb_dir: str | Path | None = None,
        movielens_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        tmdb_dir: str | Path | None = None,
        cpi_dir: str | Path | None = None,
    ) -> "Paths":
        """Build Paths from explicit arguments, then env vars, then defaults."""
        return cls(
            tmdb_dir=Path(tmdb_dir) if tmdb_dir else _env_path(
                "NEXTAAA_TMDB_DIR", PROJECT_ROOT / "data" / "tmdb"),
            cpi_dir=Path(cpi_dir) if cpi_dir else _env_path(
                "NEXTAAA_CPI_DIR", PROJECT_ROOT / "data" / "cpi"),
            imdb_dir=Path(imdb_dir) if imdb_dir else _env_path(
                "NEXTAAA_IMDB_DIR", PROJECT_ROOT / "data" / "imdb"),
            movielens_dir=Path(movielens_dir) if movielens_dir else _env_path(
                "NEXTAAA_MOVIELENS_DIR", PROJECT_ROOT / "data" / "ml-20m"),
            cache_dir=Path(cache_dir) if cache_dir else _env_path(
                "NEXTAAA_CACHE_DIR", PROJECT_ROOT / "data" / "cache"),
            output_dir=Path(output_dir) if output_dir else _env_path(
                "NEXTAAA_OUTPUT_DIR", PROJECT_ROOT / "outputs"),
        )

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    def make_output_dirs(self) -> None:
        for d in (self.cache_dir, self.figures_dir, self.tables_dir):
            d.mkdir(parents=True, exist_ok=True)


def _last_complete_year() -> int:
    """The most recent calendar year that has fully ended.

    Films from the current, unfinished year are still collecting votes, so we
    leave them out of the clustering by default.
    """
    return _dt.date.today().year - 1


@dataclass(frozen=True)
class AnalysisConfig:
    """Every tunable number in the analysis, with the reason for its default."""

    # ---- Which titles count as "a movie" ---------------------------------
    # IMDb titleType values include movie, tvMovie, short, tvSeries, videoGame...
    # A studio greenlighting a AAA title cares about theatrical feature films.
    title_types: tuple[str, ...] = ("movie",)
    exclude_adult: bool = True
    # Some features are mislabelled shorts. 60 min is a common feature cut-off.
    min_runtime_minutes: int = 60
    # Many legitimate films have no runtime recorded; keep them by default.
    keep_missing_runtime: bool = True

    # ---- The universe we cluster on ---------------------------------------
    # A rating built on a handful of votes is noise. 1,000 votes keeps tens of
    # thousands of films while making each averageRating reasonably stable.
    min_votes_for_clustering: int = 1_000
    # Earlier decades have too few films per year for stable year-adjustment.
    min_year: int = 1950
    # Leave out the unfinished current year (votes still accumulating).
    max_year: int = field(default_factory=_last_complete_year)
    # Year-adjustment needs enough films per year to estimate a "typical" film.
    min_films_per_year: int = 30

    # ---- Bayesian shrinkage (the "balanced" rating) ----------------------
    # m = this quantile of numVotes within the comparison pool. IMDb's Top 250
    # uses a "minimum votes to be listed" threshold for m; the 90th percentile
    # is the common data-driven equivalent. We also show sensitivity to it.
    shrinkage_quantile: float = 0.90
    # Quantiles of numVotes used for the "does the #1 film change?" chart.
    sensitivity_quantiles: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)

    # ---- The EDA year --------------------------------------------------------
    target_year: int = 2023
    # For "highest rated 2023 film", only consider films with at least this
    # many votes (protects the leaderboard from obscure 5-vote titles; the
    # shrinkage does most of the work, this just trims the long tail).
    min_votes_target_year: int = 100

    # ---- Clustering -----------------------------------------------------------
    k_range: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    # The four-quadrant hypothesis: Hit / Hidden Gem / Loser / Popular but Divisive.
    k_hypothesis: int = 4
    # Silhouette is O(n^2); scoring on a random sample is standard practice.
    silhouette_sample_size: int = 15_000
    # DBSCAN min_samples ("minPts"). Literature guidance: 2 x dimensions for
    # small, clean data (Ester et al. 1996; Schubert et al. 2017); larger for
    # big, noisy data. With tens of thousands of points, 25 avoids lots of
    # tiny accidental clusters.
    dbscan_min_samples: int = 25
    # DBSCAN memory grows with neighbourhood size. Above this many points we fit
    # on a random sample and assign the rest to the nearest core point.
    dbscan_max_points: int = 60_000
    random_state: int = 42

    # ---- Actor popularity --------------------------------------------------
    # Billing weight = 1 / sqrt(billing rank): lead 1.00, 2nd 0.71, 4th 0.50.
    # Softer than 1/rank so ensemble casts are not wiped out.
    billing_weight_power: float = 0.5
    actor_categories: tuple[str, ...] = ("actor", "actress")
    top_n_actors: int = 15

    # ---- Trends --------------------------------------------------------------
    trend_min_year: int = 1970
    # Genres shown individually in trend charts; the rest fold into "Other".
    trend_top_genres: int = 6
    # MovieLens: a movie needs this many ratings for its rating spread to be trusted.
    movielens_min_ratings_for_std: int = 100
    # MovieLens: skip rating-years with fewer ratings than this (1995 is nearly empty).
    movielens_min_ratings_per_year: int = 10_000

    # =========================================================================
    # v2: money (TMDB budget / revenue), five methods, by-year + all-time
    # =========================================================================

    # ---- Cleaning TMDB money fields ----------------------------------------
    # TMDB stores "unknown" as 0, and some entries are typed in millions
    # ("5" meaning $5M). Real micro-budget hits exist (Paranormal Activity:
    # ~$15k), so the floor is low: below it we treat the figure as unknown.
    min_budget_usd: float = 10_000
    min_revenue_usd: float = 10_000

    # ---- ROI -------------------------------------------------------------------
    # ROI multiple = worldwide box office / production budget. Industry rule of
    # thumb: a film needs ~2.5x its budget at the box office to break even,
    # because theatres keep ~half the ticket money and marketing (P&A) costs
    # roughly as much as production.
    breakeven_multiple: float = 2.5
    # "AAA-scale" production: budget at or above this, in base-year dollars when
    # a CPI file is available (nominal dollars otherwise, with a warning).
    aaa_budget_usd: float = 100_000_000
    cpi_base_year: int = 2023

    # ---- Year normalisation ------------------------------------------------------
    # Every clustering feature is compared with films released the SAME year
    # (robust z-score). Years with too few films borrow neighbouring years:
    # the window widens one year at a time, up to +/- max_year_window.
    min_ref_films_audience: int = 50
    min_ref_films_finance: int = 30
    max_year_window: int = 5
    # Budgets are sparse before ~1970; the money stage starts here.
    finance_min_year: int = 1970

    # ---- The five methods ---------------------------------------------------------
    # HDBSCAN (Campello et al. 2013): no eps to choose. min_cluster_size is the
    # smallest group worth calling a cluster: here max(50, 1% of films).
    hdbscan_min_cluster_frac: float = 0.01
    hdbscan_min_cluster_floor: int = 50
    # HDBSCAN's outlier score is GLOSH (Campello et al. 2015). Following the
    # hdbscan documentation, the most outlying 10% (scores above the 90th
    # percentile) count as unusual.
    glosh_quantile: float = 0.90
    # Isolation Forest (Liu et al. 2008). contamination = share flagged as
    # unusual. 0.10 matches GLOSH so the two score-based detectors are equally
    # strict; "auto" (the paper's 0.5 score threshold) flags ~20% on this data.
    iforest_contamination: float | str = 0.10
    iforest_n_estimators: int = 300
    # Votes (out of 5) needed to call a film a hit in a stage (a majority), and
    # for "mega" (unanimous). GLOSH and Isolation Forest agree closely, so 4 votes
    # is barely rarer than 3; unanimity is the meaningful higher bar.
    hit_min_votes: int = 3
    mega_min_votes: int = 5

    # ---- Reporting -------------------------------------------------------------------
    top_n_per_year: int = 5
    top_n_all_time: int = 25
    # Genres: "recent" window for the genre economics table.
    genre_recent_years: int = 15


# =============================================================================
# Part B: generate -> score -> refine (the next AAA title)
# =============================================================================

@dataclass(frozen=True)
class PartBConfig:
    """Every tunable number in Part B, with the reason for its default.

    LLM endpoints and keys are NOT here: they come from environment variables
    (see nextaaa/partb/llm.py and .env.example), so secrets never sit in code.
    """

    # ---- Talent track records ------------------------------------------------
    # Smoothed hit rate = (hits + a * base_rate) / (films + a): a director with
    # 1 film and 1 hit is not treated as "100% hit rate".
    track_record_prior_strength: float = 2.0
    leads_per_film: int = 2                 # top-billed actors used as "leads"
    shortlist_directors: int = 25           # shown to the generator
    shortlist_actors: int = 40
    shortlist_min_prior_films: int = 2      # need some track record to be listed
    shortlist_active_within_years: int = 8  # last film within this many years

    # ---- Text features ---------------------------------------------------------
    text_pca_components: int = 16           # plot-embedding dimensions fed to the scorer

    # ---- Scorer and backtest -----------------------------------------------------
    # Train on films up to test_start - 1, test on test_start .. test_end.
    # Defaults (None) = the last 3 complete years before the most recent one,
    # because the most recent year's votes and box office are still maturing.
    test_start_year: int | None = None
    test_end_year: int | None = None
    validation_years: int = 3               # inside the training period, for model choice
    candidate_models: tuple[str, ...] = ("logistic", "hgb", "tabpfn")
    tabpfn_max_train: int = 10_000           # TabPFN on CPU: subsample (keeping all hits)
    n_bootstrap: int = 5                    # refits for the uncertainty band
    ood_percentile: float = 99.0            # beyond this kNN-distance percentile = unfamiliar
    random_state: int = 42

    # ---- Concepts ----------------------------------------------------------------
    min_budget_usd: float = 100_000_000     # AAA
    max_budget_usd: float = 350_000_000
    min_runtime: int = 90
    max_runtime: int = 175

    # ---- The loop (OPRO-style) -------------------------------------------------------
    n_islands: int = 3                      # separate lines of ideas, one lead genre each
    n_rounds: int = 5
    concepts_per_call: int = 8
    history_top_k: int = 20                 # best concepts shown back to the generator
    generator_temperature: float = 1.0
    # Novelty: reject a concept whose plot is more similar to an existing film than
    # this percentile of "film vs its nearest other film" similarities.
    novelty_percentile: float = 99.0
    run_ablation: bool = True               # same calls without score feedback
    portfolio_size: int = 3

    # ---- LLM-as-predictor baseline (backtest only) -------------------------------------
    run_llm_baseline: bool = True
    llm_baseline_max_films: int = 200       # API calls; all hits are kept
    llm_baseline_shots: int = 20            # retrieved past films per prompt
    llm_baseline_workers: int = 4
    max_parallel_calls:int = 6