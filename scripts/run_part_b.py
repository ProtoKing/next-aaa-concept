"""Command-line entry point for Part B: design the next AAA title.

From the project root:

    # Offline smoke test: synthetic data + mock LLM (about 2-3 minutes):
    python scripts/run_part_b.py --demo

    # Real run. Needs Part A v2 labels (run automatically if missing) and,
    # for real concepts, LLM settings in .env (see .env.example):
    python scripts/run_part_b.py

    # Force offline mode on real data (to test the judge and backtest only):
    python scripts/run_part_b.py --mock

    # Cheaper first live run:
    python scripts/run_part_b.py --rounds 3 --islands 2 --no-ablation --no-llm-baseline

Outputs go to outputs/partb/ (pitch_cards.md is the headline).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextaaa.config import PROJECT_ROOT, AnalysisConfig, PartBConfig, Paths  # noqa: E402
from nextaaa.io_imdb import MissingDataError  # noqa: E402
from nextaaa.partb.pipeline import run_part_b  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part B: generate -> score -> refine the next AAA title.")
    p.add_argument("--imdb-dir", type=Path)
    p.add_argument("--movielens-dir", type=Path)
    p.add_argument("--tmdb-dir", type=Path)
    p.add_argument("--cpi-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--mock", action="store_true", help="Use the offline mock LLM even if .env is set")
    p.add_argument("--rounds", type=int, help="Rounds per island (default 5)")
    p.add_argument("--islands", type=int, help="Islands / lead genres (default 3)")
    p.add_argument("--per-call", type=int, help="Concepts per generator call (default 8)")
    p.add_argument("--no-ablation", action="store_true", help="Skip the no-feedback control arm")
    p.add_argument("--no-llm-baseline", action="store_true", help="Skip the LLM-as-predictor backtest")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--demo", action="store_true", help="Synthetic data + mock LLM (setup check)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("nextaaa.clustering").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg, bcfg = AnalysisConfig(), PartBConfig()
    overrides = {k: v for k, v in (("n_rounds", args.rounds), ("n_islands", args.islands),
                                   ("concepts_per_call", args.per_call)) if v}
    if args.no_ablation:
        overrides["run_ablation"] = False
    if args.no_llm_baseline:
        overrides["run_llm_baseline"] = False
    bcfg = replace(bcfg, **overrides)
    mode = "mock" if (args.mock or args.demo) else None

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
        logging.warning("DEMO MODE: synthetic data and a mock LLM. Numbers and concepts are meaningless.")
    else:
        paths = Paths.resolve(args.imdb_dir, args.movielens_dir, args.cache_dir, args.output_dir,
                              args.tmdb_dir, args.cpi_dir)
    try:
        summary = run_part_b(paths, cfg, bcfg, mode=mode, refresh=args.refresh, make_plots=not args.no_plots)
    except (MissingDataError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    print(json.dumps({k: summary.get(k) for k in ("clients", "backtest", "portfolio", "llm_baseline", "usage")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
