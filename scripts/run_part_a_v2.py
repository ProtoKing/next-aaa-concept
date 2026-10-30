"""Command-line entry point for Part A v2 (money, five methods, by-year + all-time).

From the project root:

    # Smoke test on fake data (about a minute):
    python scripts/run_part_a_v2.py --demo

    # Real run (IMDb required; TMDB, CPI and MovieLens optional):
    python scripts/run_part_a_v2.py

    # Different KMeans k per stage, after reading the elbow charts:
    python scripts/run_part_a_v2.py --k-audience 3 --k-commercial 4

Outputs go to outputs/v2/. Run with --help for every option.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextaaa.config import PROJECT_ROOT, AnalysisConfig, Paths  # noqa: E402
from nextaaa.io_imdb import MissingDataError  # noqa: E402
from nextaaa.pipeline_v2 import run_part_a_v2  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part A v2: money-aware hits, by year and all time.")
    p.add_argument("--imdb-dir", type=Path)
    p.add_argument("--movielens-dir", type=Path)
    p.add_argument("--tmdb-dir", type=Path, help="Folder with the TMDB CSV or Kaggle zip")
    p.add_argument("--cpi-dir", type=Path, help="Folder with a FRED CPIAUCSL CSV (optional)")
    p.add_argument("--output-dir", type=Path, help="Base output folder; v2 writes to <it>/v2")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--k-audience", type=int, default=None, help="KMeans k for the audience stage")
    p.add_argument("--k-commercial", type=int, default=None, help="KMeans k for the commercial stage")
    p.add_argument("--year", type=int, default=None, help="EDA year (default 2023)")
    p.add_argument("--refresh", action="store_true", help="Ignore the cache and re-parse all files")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--demo", action="store_true", help="Run on synthetic data (setup check only)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = AnalysisConfig()
    if args.year:
        cfg = replace(cfg, target_year=args.year)

    if args.demo:
        from nextaaa.synthetic import (make_synthetic_cpi, make_synthetic_imdb, make_synthetic_movielens,
                                       make_synthetic_tmdb)
        demo = PROJECT_ROOT / "data" / "synthetic"
        if not (demo / "imdb" / "title.basics.tsv.gz").exists():
            make_synthetic_imdb(demo / "imdb")
            make_synthetic_movielens(demo / "imdb", demo / "ml-20m")
        if not (demo / "tmdb").exists():
            make_synthetic_tmdb(demo / "imdb", demo / "tmdb")
            make_synthetic_cpi(demo / "cpi")
        paths = Paths.resolve(demo / "imdb", demo / "ml-20m", demo / "cache",
                              args.output_dir or PROJECT_ROOT / "outputs_demo", demo / "tmdb", demo / "cpi")
        cfg = replace(cfg, min_votes_for_clustering=200, movielens_min_ratings_per_year=500,
                      movielens_min_ratings_for_std=20, dbscan_min_samples=10)
        logging.warning("DEMO MODE: synthetic data. Numbers are meaningless; do not use in the deck.")
    else:
        paths = Paths.resolve(args.imdb_dir, args.movielens_dir, args.cache_dir, args.output_dir,
                              args.tmdb_dir, args.cpi_dir)

    try:
        summary = run_part_a_v2(paths, cfg, k_audience=args.k_audience, k_commercial=args.k_commercial,
                                refresh=args.refresh, make_plots=not args.no_plots)
    except MissingDataError as exc:
        logging.error("%s", exc)
        return 1
    headline = {k: summary.get(k) for k in ("top_rated", "actors", "audience_hits", "aaa_hits",
                                             "aaa_mega_hits", "all_time_top5", "verdicts", "dollar_basis")}
    print(json.dumps(headline, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
