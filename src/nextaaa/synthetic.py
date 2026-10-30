"""Generate small fake IMDb, MovieLens, TMDB and CPI files in the EXACT real formats.

Why: the real files are large and slow to download. With this you can check
that the whole pipeline runs end to end in under a minute, and the automated
tests use it so they never depend on the real data.

The fake data is shaped like the real thing (more films in recent years,
heavy-tailed vote counts, ratings centred near 6, "\\N" for missing values,
quote characters inside titles, a film literally titled "NA", non-movie titles
and adult titles that must be filtered out). The numbers it produces mean
nothing; never put them in the deck.

    python -m nextaaa.synthetic --out data/synthetic
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["Drama", "Comedy", "Action", "Thriller", "Romance", "Horror", "Crime",
          "Adventure", "Sci-Fi", "Animation", "Documentary", "Fantasy", "Family"]
WORDS = ["Night", "Shadow", "River", "Last", "Iron", "Golden", "Silent", "Broken",
         "City", "Storm", "Echo", "Wild", "Hidden", "Crimson", "Lost", "Northern"]


def _write_imdb_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write like IMDb does: tab-separated, gzipped, missing values as \\N, raw quotes."""
    text = df.astype(object).where(df.notna(), "\\N").astype(str)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("\t".join(df.columns) + "\n")
        for row in text.itertuples(index=False):
            f.write("\t".join(row) + "\n")


def make_synthetic_imdb(out_dir: Path, n_movies: int = 12_000, seed: int = 7) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- titles -----------------------------------------------------------
    years = np.clip(np.round(2025 - rng.exponential(18, n_movies)), 1950, 2025).astype(int)
    years[:400] = 2023  # make sure the target year is well populated
    quality = rng.normal(0, 1, n_movies)
    popularity = 0.35 * quality + rng.normal(0, 1, n_movies)
    titles = [f"The {rng.choice(WORDS)} {rng.choice(WORDS)} {i}" for i in range(n_movies)]
    titles[1] = 'The "Quoted" Picture'   # quote characters (trap 2)
    titles[2] = "NA"                     # a film literally titled NA (trap 3)

    genres = []
    for _ in range(n_movies):
        k = rng.integers(1, 4)
        genres.append(",".join(sorted(rng.choice(GENRES, size=k, replace=False))))
    genres = pd.Series(genres)
    genres[rng.random(n_movies) < 0.02] = None  # some missing genres

    runtime = pd.Series(rng.normal(105, 18, n_movies).round().astype(int)).astype("Int64")
    runtime[rng.random(n_movies) < 0.08] = pd.NA  # missing runtimes
    runtime[rng.random(n_movies) < 0.02] = 45      # mislabelled shorts

    movies = pd.DataFrame({
        "tconst": [f"tt{1000000 + i:07d}" for i in range(n_movies)],
        "titleType": "movie", "primaryTitle": titles, "originalTitle": titles,
        "isAdult": (rng.random(n_movies) < 0.01).astype(int), "startYear": years,
        "endYear": None, "runtimeMinutes": runtime, "genres": genres,
    })
    n_other = n_movies // 4
    other = pd.DataFrame({
        "tconst": [f"tt{3000000 + i:07d}" for i in range(n_other)],
        "titleType": rng.choice(["short", "tvSeries", "tvEpisode", "videoGame"], n_other),
        "primaryTitle": "Other title", "originalTitle": "Other title", "isAdult": 0,
        "startYear": rng.integers(1950, 2026, n_other), "endYear": None,
        "runtimeMinutes": pd.array([20] * n_other, dtype="Int64"), "genres": "Drama",
    })
    _write_imdb_tsv(pd.concat([movies, other]), out_dir / "title.basics.tsv.gz")

    # --- ratings: rating from quality; votes from popularity, age and IMDb growth
    age = 2026 - years
    log_votes = (2.2 + 0.9 * popularity + 0.25 * quality + 0.012 * np.minimum(age, 25)
                 + 0.015 * (years - 1950) + rng.normal(0, 0.35, n_movies))
    votes = np.maximum(5, (10 ** log_votes)).astype(int)
    rating = np.clip(6.2 + 0.85 * quality + rng.normal(0, 0.35, n_movies), 1, 10).round(1)
    ratings = pd.DataFrame({"tconst": movies["tconst"], "averageRating": rating, "numVotes": votes})
    keep = rng.random(n_movies) > 0.03  # a few films have no rating at all
    other_r = pd.DataFrame({"tconst": other["tconst"], "averageRating": 7.0, "numVotes": 50})
    _write_imdb_tsv(pd.concat([ratings[keep], other_r]), out_dir / "title.ratings.tsv.gz")

    # --- people and credits: stars land in popular films more often ---------
    n_people = 4_000
    star_power = rng.pareto(2.5, n_people)
    p = star_power / star_power.sum()
    rows = []
    for i, t in enumerate(movies["tconst"]):
        n_cast = rng.integers(3, 8)
        boost = np.exp(0.6 * popularity[i])
        cast = rng.choice(n_people, size=n_cast, replace=False,
                          p=(p ** (1 / boost)) / (p ** (1 / boost)).sum())
        cats = ["director", "writer"] + list(rng.choice(["actor", "actress"], n_cast)) + ["composer"]
        people = [rng.integers(n_people), rng.integers(n_people)] + list(cast) + [rng.integers(n_people)]
        for order, (nc, cat) in enumerate(zip(people, cats), start=1):
            rows.append((t, order, f"nm{1000000 + int(nc):07d}", cat, None, None))
    principals = pd.DataFrame(rows, columns=["tconst", "ordering", "nconst", "category", "job", "characters"])
    _write_imdb_tsv(principals, out_dir / "title.principals.tsv.gz")

    first = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn"]
    last = ["Reyes", "Okafor", "Lindqvist", "Tanaka", "Moreau", "Kowalski", "Haddad", "Silva"]
    names = pd.DataFrame({
        "nconst": [f"nm{1000000 + i:07d}" for i in range(n_people)],
        "primaryName": [f"{rng.choice(first)} {rng.choice(last)} {i}" for i in range(n_people)],
        "birthYear": None, "deathYear": None, "primaryProfession": "actor", "knownForTitles": None,
    })
    _write_imdb_tsv(names, out_dir / "name.basics.tsv.gz")


def make_synthetic_movielens(imdb_dir: Path, out_dir: Path, n_users: int = 1_500,
                             n_films: int = 900, seed: int = 11) -> None:
    """A MovieLens-format sample of films released up to 2015, linked to the fake IMDb."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    basics = pd.read_csv(imdb_dir / "title.basics.tsv.gz", sep="\t", na_values="\\N",
                         keep_default_na=False, quoting=csv.QUOTE_NONE, dtype=str)
    ratings = pd.read_csv(imdb_dir / "title.ratings.tsv.gz", sep="\t", dtype=str)
    pool = basics[(basics["titleType"] == "movie")].merge(ratings, on="tconst")
    pool = pool[pd.to_numeric(pool["startYear"]) <= 2015]
    pool = pool.sort_values("numVotes", key=lambda s: pd.to_numeric(s), ascending=False).head(n_films)

    movies = pd.DataFrame({
        "movieId": np.arange(1, len(pool) + 1),
        "title": [f"{t} ({y})" for t, y in zip(pool["primaryTitle"], pool["startYear"])],
        "genres": pool["genres"].fillna("(no genres listed)").str.replace(",", "|").to_numpy(),
    })
    movies.to_csv(out_dir / "movies.csv", index=False)
    pd.DataFrame({"movieId": movies["movieId"],
                  "imdbId": pool["tconst"].str[2:].to_numpy(),
                  "tmdbId": ""}).to_csv(out_dir / "links.csv", index=False)

    base = pd.to_numeric(pool["averageRating"]).to_numpy() / 2  # 1-10 -> 0.5-5
    weight = pd.to_numeric(pool["numVotes"]).to_numpy().astype(float) ** 0.6
    weight /= weight.sum()
    rows = []
    for u in range(1, n_users + 1):
        n = rng.integers(20, 160)
        films = rng.choice(len(pool), size=n, replace=False, p=weight)
        spread = rng.uniform(0.5, 1.4)
        r = np.clip(np.round((base[films] + rng.normal(0, spread, n)) * 2) / 2, 0.5, 5)
        ts = rng.integers(820454400, 1427846400, n)  # 1996 .. 2015
        rows.append(pd.DataFrame({"userId": u, "movieId": movies["movieId"].to_numpy()[films],
                                  "rating": r, "timestamp": ts}))
    pd.concat(rows).to_csv(out_dir / "ratings.csv", index=False)


TMDB_COLUMNS = ["id", "title", "vote_average", "vote_count", "status", "release_date", "revenue",
                "runtime", "adult", "backdrop_path", "budget", "homepage", "imdb_id",
                "original_language", "original_title", "overview", "popularity", "poster_path",
                "tagline", "genres", "production_companies", "production_countries",
                "spoken_languages", "keywords"]


def make_synthetic_tmdb(imdb_dir: Path, out_dir: Path, seed: int = 23, as_zip: bool = False) -> Path:
    """A TMDB-format CSV (asaniczka layout) for the fake IMDb films, with every trap:
    zero budgets, budgets typed in millions, revenue copied from budget, duplicate
    and missing imdb_ids, unreleased films, TMDB-only films, and plot text with
    commas, quotes and line breaks."""
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    basics = pd.read_csv(imdb_dir / "title.basics.tsv.gz", sep="\t", na_values="\\N",
                         keep_default_na=False, quoting=csv.QUOTE_NONE, dtype=str)
    ratings = pd.read_csv(imdb_dir / "title.ratings.tsv.gz", sep="\t", dtype=str)
    films = basics[basics["titleType"] == "movie"].merge(ratings, on="tconst")
    year = pd.to_numeric(films["startYear"]).to_numpy()
    rating = pd.to_numeric(films["averageRating"]).to_numpy()
    log_votes = np.log10(pd.to_numeric(films["numVotes"]).to_numpy())
    n = len(films)
    q = (rating - 6.2) / 0.9
    pop = (log_votes - np.median(log_votes)) / log_votes.std()

    # Popular, recent films are more likely to have money figures (as in real TMDB).
    p_money = 1 / (1 + np.exp(-(1.2 * pop + 0.04 * (year - 1995) - 0.6)))
    has_money = rng.random(n) < p_money
    inflation = 1.035 ** (year - 2000)
    budget = np.exp(rng.normal(np.log(15e6) + 0.7 * pop, 0.9)) * inflation
    log_roi = rng.normal(0.25 + 0.18 * q + 0.22 * pop, 0.38)
    revenue = budget * 10 ** log_roi
    budget = np.where(has_money, budget.round(-3), 0)
    revenue = np.where(has_money & (rng.random(n) > 0.12), revenue.round(-3), 0)  # some revenue unknown

    unit_err = has_money & (rng.random(n) < 0.02)
    budget = np.where(unit_err, np.maximum(1, (budget / 1e6).round()), budget)  # "5" meaning $5M
    copy_err = has_money & (rng.random(n) < 0.01)
    revenue = np.where(copy_err, budget, revenue)

    words = ["heist", "war", "family", "space", "revenge", "love", "storm", "secret", "border", "island",
             "election", "plague", "kingdom", "robot", "ocean", "prison", "circus", "desert", "train"]
    people = ["a detective", "two sisters", "a pilot", "a teacher", "an android", "a chef", "a soldier",
              "a hacker", "a widow", "a boxer", "a monk", "a journalist", "a smuggler", "a nurse"]
    overview = [f"A {rng.choice(words)} story: {rng.choice(people)} and {rng.choice(people)} face a "
                f"{rng.choice(words)} and a {rng.choice(words)}, with \"twists\", around year {y}."
                + ("\nSecond paragraph, after a line break." if i % 50 == 0 else "")
                for i, y in enumerate(year)]
    status = np.where(rng.random(n) < 0.01, "Post Production", "Released")
    imdb_id = films["tconst"].to_numpy().astype(object)
    imdb_id[rng.random(n) < 0.02] = None  # missing ids cannot be joined

    df = pd.DataFrame({
        "id": np.arange(1, n + 1), "title": films["primaryTitle"].to_numpy(),
        "vote_average": (rating * 0.95).round(3), "vote_count": (10 ** (log_votes - 1.5)).astype(int),
        "status": status, "release_date": [f"{y}-06-15" for y in year],
        "revenue": revenue, "runtime": films["runtimeMinutes"].to_numpy(), "adult": "False",
        "backdrop_path": "", "budget": budget, "homepage": "", "imdb_id": imdb_id,
        "original_language": rng.choice(["en", "en", "en", "fr", "ja", "ko", "es"], n),
        "original_title": films["primaryTitle"].to_numpy(), "overview": overview,
        "popularity": (10 ** pop).round(3), "poster_path": "", "tagline": "",
        "genres": films["genres"].fillna("").str.replace(",", ", ").to_numpy(),
        "production_companies": rng.choice(["Studio A", "Studio B, Studio C", ""], n),
        "production_countries": "United States of America", "spoken_languages": "English",
        "keywords": rng.choice(["heist, crime", "space, future", "family", "sequel, superhero",
                                "based on novel or book", "revenge"], n),
    })[TMDB_COLUMNS]
    # Duplicate rows for 1% of films, without money (the loader must keep the richer row).
    dup = df.sample(frac=0.01, random_state=seed).assign(budget=0, revenue=0, vote_count=1)
    # TMDB-only films that are not in IMDb's movie list.
    extra = df.sample(n=min(500, n), random_state=seed + 1).assign(
        imdb_id=[f"tt{9000000 + i:07d}" for i in range(min(500, n))])
    df = pd.concat([df, dup, extra]).sample(frac=1, random_state=seed)

    csv_path = out_dir / "TMDB_movie_dataset_v11.csv"
    df.to_csv(csv_path, index=False)
    if not as_zip:
        return csv_path
    zip_path = out_dir / "archive.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
    csv_path.unlink()
    return zip_path


def make_synthetic_cpi(out_dir: Path) -> Path:
    """A monthly CPI file in FRED's current layout (observation_date, CPIAUCSL).
    About 3.5% a year: NOT real CPI. Download the real one for the deck."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("1947-01-01", "2025-12-01", freq="MS")
    t = np.arange(len(dates)) / 12
    cpi = pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"),
                        "CPIAUCSL": (21.5 * 1.035 ** t).round(3)})
    path = out_dir / "CPIAUCSL_synthetic.csv"
    cpi.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()
    make_synthetic_imdb(args.out / "imdb")
    make_synthetic_movielens(args.out / "imdb", args.out / "ml-20m")
    make_synthetic_tmdb(args.out / "imdb", args.out / "tmdb")
    make_synthetic_cpi(args.out / "cpi")
    print(f"Synthetic data written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
