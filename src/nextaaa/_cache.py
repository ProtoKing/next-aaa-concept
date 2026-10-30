"""A tiny on-disk cache for expensive DataFrame builds.

The raw IMDb files are large (title.principals alone is ~100 million rows), and
re-parsing them on every run wastes minutes. The first run parses and filters
each file once and saves the result; later runs read the small cached copy.

A cache entry is reused only if it is newer than every source file it was built
from. Download fresh IMDb data and the cache rebuilds itself automatically.

Format: Parquet when `pyarrow` is installed (fast, compact), otherwise pickle.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

try:  # pyarrow is optional
    import pyarrow  # noqa: F401

    _HAS_PYARROW = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_PYARROW = False


def _params_hash(params: dict | None) -> str:
    """Short, stable fingerprint of the parameters that shaped a cache entry."""
    if not params:
        return "noparams"
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def cached_frame(
    name: str,
    builder: Callable[[], pd.DataFrame],
    cache_dir: Path,
    sources: Iterable[Path] = (),
    params: dict | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return `builder()`, reading from / writing to an on-disk cache.

    Args:
        name: Human-readable cache name, e.g. "title_basics".
        builder: Zero-argument function that builds the DataFrame from scratch.
        cache_dir: Folder for cache files.
        sources: Input files; the cache is stale if any is newer than it.
        params: Settings that change the result (e.g. filters). Different
            params produce a different cache file, so they never collide.
        refresh: Ignore any existing cache and rebuild.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".parquet" if _HAS_PYARROW else ".pkl"
    path = cache_dir / f"{name}__{_params_hash(params)}{suffix}"

    sources = [Path(s) for s in sources]
    newest_source = max((s.stat().st_mtime for s in sources if s.exists()), default=0.0)
    is_fresh = path.exists() and path.stat().st_mtime >= newest_source

    if is_fresh and not refresh:
        logger.info("Cache hit: %s", path.name)
        return pd.read_parquet(path) if _HAS_PYARROW else pd.read_pickle(path)

    logger.info("Building %s (this can take a few minutes the first time)...", name)
    df = builder()
    try:
        if _HAS_PYARROW:
            df.to_parquet(path, index=False)
        else:
            df.to_pickle(path)
        logger.info("Cached %s rows to %s", f"{len(df):,}", path.name)
    except Exception as exc:  # caching is a convenience; never fail the run on it
        logger.warning("Could not write cache %s (%s); continuing without it.", path, exc)
    return df
