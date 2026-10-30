"""nextaaa: code for "Give Me The Next AAA Title" (Deliverable #1, Part A).

Module map (read them in this order if you are new to the code):

    config.py      Where the data lives and every tunable number, in one place.
    io_imdb.py     Loads the IMDb .tsv.gz files safely and caches them.
    movielens.py   Loads MovieLens 20M (optional: everything works without it).
    ratings.py     The Bayesian "shrunk" rating that balances rating vs. votes.
    features.py    Builds the two clustering axes: quality and year-adjusted reach.
    clustering.py  Elbow, silhouette, KMeans, DBSCAN, and the final hit labels.
    popularity.py  "Most popular actor of 2023" under two definitions.
    trends.py      Trends in audience preferences over the years.
    plots.py       Every figure for the deck, in one consistent style.
    pipeline.py    Runs all of the above in order and saves outputs.

v2 (money, five methods, by-year and all-time hits):

    io_tmdb.py     Loads TMDB budget, revenue and plot text; joins on imdb_id.
    finance.py     ROI, 2.5x break-even, budget tiers, optional CPI (real dollars).
    normalize.py   Compares every film with films released the same year.
    ensemble.py    Five methods, one vote each: KMeans, median split, DBSCAN,
                   HDBSCAN (GLOSH), Isolation Forest. Runs on any pair of axes.
    hits.py        Two-stage gate -> verdicts; hits by year; all-time hits.
    plots_v2.py    The v2 figures.
    pipeline_v2.py Runs v2 end to end (outputs in outputs/v2/).

Part B (design the next AAA title): see the nextaaa.partb package.
"""

__version__ = "0.3.0"
