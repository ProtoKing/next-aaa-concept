"""Command-line entry point for Part A.

Typical use (from the project root):

    # 1) Smoke test on fake data (about a minute), to check your setup:
    python scripts/run_part_a.py --demo

    # 2) The real run (first run parses the big files and caches them):
    python scripts/run_part_a.py --imdb-dir data/imdb --movielens-dir data/ml-20m

    # Try a different number of KMeans clusters after reading the elbow plot:
    python scripts/run_part_a.py --k 3

Run `python scripts/run_part_a.py --help` for every option.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display

# Make `import nextaaa` work even if the package was not pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextaaa.config import PROJECT_ROOT, AnalysisConfig, Paths  # noqa: E402
from nextaaa.io_imdb import MissingDataError  # noqa: E402
from nextaaa.pipeline import run_part_a  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part A: hit metric, clustering and EDA.")
    p.add_argument("--imdb-dir", type=Path, help="Folder with the IMDb .tsv.gz files")
    p.add_argument("--movielens-dir", type=Path, help="Folder with MovieLens 20M CSVs (optional)")
    p.add_argument("--output-dir", type=Path, help="Where tables and figures go (default: outputs/)")
    p.add_argument("--cache-dir", type=Path, help="Where parsed data is cached (default: data/cache/)")
    p.add_argument("--k", type=int, default=None, help="KMeans clusters (default: 4, the hypothesis)")
    p.add_argument("--year", type=int, default=None, help="EDA year (default: 2023)")
    p.add_argument("--refresh", action="store_true", help="Ignore the cache and re-parse all files")
    p.add_argument("--no-plots", action="store_true", help="Skip figures (faster)")
    p.add_argument("--demo", action="store_true",
                   help="Generate small synthetic data and run on it (setup check only)")
    p.add_argument("-v", "--verbose", action="store_true", help="More detailed logging")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = AnalysisConfig()
    if args.year:
        cfg = replace(cfg, target_year=args.year)

    if args.demo:
        from nextaaa.synthetic import make_synthetic_imdb, make_synthetic_movielens
        demo = PROJECT_ROOT / "data" / "synthetic"
        if not (demo / "imdb" / "title.basics.tsv.gz").exists():
            make_synthetic_imdb(demo / "imdb")
            make_synthetic_movielens(demo / "imdb", demo / "ml-20m")
        paths = Paths.resolve(demo / "imdb", demo / "ml-20m", demo / "cache",
                              args.output_dir or PROJECT_ROOT / "outputs_demo")
        # The fake data is small, so relax thresholds meant for the real thing.
        cfg = replace(cfg, min_votes_for_clustering=200, movielens_min_ratings_per_year=500,
                      movielens_min_ratings_for_std=20, dbscan_min_samples=10)
        logging.warning("DEMO MODE: synthetic data. Numbers are meaningless; do not use in the deck.")
    else:
        paths = Paths.resolve(args.imdb_dir, args.movielens_dir, args.cache_dir, args.output_dir)

    try:
        summary = run_part_a(paths, cfg, k=args.k, refresh=args.refresh, make_plots=not args.no_plots)
    except MissingDataError as exc:
        logging.error("%s", exc)
        return 1

    headline = {k: summary[k] for k in ("top_rated", "actors", "clustering") if k in summary}
    print(json.dumps(headline, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
