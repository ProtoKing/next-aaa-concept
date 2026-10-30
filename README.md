# Give Me The Next AAA Title: Part A (Deliverable #1)

Code for the first half of Deliverable #1: answering the exploratory questions and **defining what a "hit" is**. Part B (the model that proposes the next AAA title) builds on the hit labels this produces.

There are two versions. **Use v2 for the deck**; v1 is kept for comparison.

| | v1 | v2 (current) |
|---|---|---|
| Notebook | `notebooks/01_part_a_walkthrough.ipynb` | `notebooks/02_part_a_v2_money_and_hits.ipynb` |
| Script | `scripts/run_part_a.py` | `scripts/run_part_a_v2.py` |
| Hit = | audience only (rating × votes) | audience hit **and** commercial hit (ROI × box office, ≥ 2.5× break-even) |
| Methods | KMeans, DBSCAN, median split | + HDBSCAN (GLOSH) and Isolation Forest: five voters |
| Lists | one label | hits **by year** and **all time** |
| Outputs | `outputs/` | `outputs/v2/` |

v2 is described in [section 9](#9-v2-money-five-methods-by-year-and-all-time-hits). Sections 1 to 8 cover the shared setup and v1.

**Part B**, the model that designs the next AAA title (LLM generator + calibrated judge + feedback loop), is in [section 10](#10-part-b-designing-the-next-aaa-title): `notebooks/03_part_b_next_aaa_title.ipynb`, `scripts/run_part_b.py`.

| Question from the brief | Where it is answered (v2) |
|---|---|
| Highest-rated movie of 2023? How to balance rating vs. votes? | `ratings.py`, figures 01 and 02 |
| Most popular actor of 2023? | `popularity.py`, figure 03 (three definitions) |
| A metric that captures a "hit movie" | `normalize.py`, `ensemble.py`, `hits.py`, figures 04 to 16 |
| Trends in user-movie preferences over the years | `hits.py`, `trends.py`, figures 14, 16 to 19 |

---

## 1. Quick start

```bash
# 1. Install (Python 3.10+). A virtual environment is recommended.
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Check your setup on small synthetic data (about a minute, no downloads needed)
python scripts/run_part_a_v2.py --demo
pytest                                                   # 53 tests, all should pass

# 3. Download the real data (see sections 2 and 9.1), then run for real
python scripts/run_part_a_v2.py
```

Or open `notebooks/02_part_a_v2_money_and_hits.ipynb`, which runs the same steps one at a time and explains how to read every table and chart. **Start with the notebook if you want to understand the analysis; use the script when you just want the outputs.**

> Demo mode uses fake data. Its numbers are meaningless, so never put them in the deck.

---

## 2. Getting the data

Put the files in these folders (or point elsewhere, see section 5):

```
data/
  imdb/                        from https://datasets.imdbws.com/  (keep them gzipped)
    title.basics.tsv.gz
    title.ratings.tsv.gz
    title.principals.tsv.gz    the biggest file; only needed for the actor question
    name.basics.tsv.gz
  ml-20m/                      from https://grouplens.org/datasets/movielens/20m/ (unzip)
    ratings.csv
    movies.csv
    links.csv
```

- **IMDb is required.** If a file is missing, you get a clear message saying which one and where to get it.
- **MovieLens is optional for Part A.** Without it, the MovieLens trend charts and the "is Divisive really divisive?" check are skipped; everything else runs.
- **First run is slow, later runs are fast.** The first run streams the big files in chunks, keeps only what it needs, and caches the result in `data/cache/`. Expect several minutes the first time, seconds afterwards. Download newer IMDb files and the cache rebuilds automatically. Use `--refresh` to force it.
- **Memory:** 8 GB RAM is comfortable. Files are filtered while streaming, so the 100-million-row principals file is never fully in memory.

---

## 3. How the code is organised

```
src/nextaaa/
  config.py       Every path and every tunable number, each with the reason for its default.
  io_imdb.py      Loads IMDb safely (handles \N, quotes in titles, films titled "NA") and caches.
  movielens.py    Loads MovieLens 20M; returns None if absent so callers can skip gracefully.
  ratings.py      The Bayesian "shrunk" rating, the 2023 leaderboard, sensitivity to m.
  features.py     The two clustering axes: quality and year-adjusted reach.
  clustering.py   Elbow + silhouette, KMeans, k-distance + DBSCAN, median split, majority vote.
  popularity.py   Most popular actor: Reach vs. Net goodwill.
  trends.py       Genre supply vs. demand (IMDb) and user behaviour over time (MovieLens).
  plots.py        All figures, one consistent colour-blind-aware style.
  pipeline.py     Runs every step in order and saves all outputs.
  synthetic.py    Generates fake data in the exact real formats (for --demo and the tests).
  _cache.py       Small on-disk cache so the big files are parsed only once.
scripts/run_part_a.py              Command-line entry point.
notebooks/01_part_a_walkthrough.ipynb   Guided, explained walkthrough.
tests/                              Unit tests + an end-to-end test on synthetic data.
```

**Suggested reading order:** `config.py` → `ratings.py` → `features.py` → `clustering.py`. Each file opens with a docstring explaining the idea in plain language before any code.

**Design principles:**
- *Pure functions where possible.* Most functions take DataFrames and return DataFrames, with no file access, so they are easy to test and reuse in Part B.
- *One place for decisions.* Thresholds live in `AnalysisConfig`, never hard-coded inside functions.
- *Fail loudly on required data, degrade gracefully on optional data.*
- *Reproducible.* Every random step uses `random_state` from the config.

---

## 4. The key decisions, and why

### Balancing rating and votes: shrinkage
IMDb's `averageRating` is *weighted* (to resist vote-stuffing) but not *shrunk* by vote count, so a 12-vote film can show 9.4. We use IMDb's own Top 250 formula:

```
shrunk = v/(v+m) × R  +  m/(v+m) × C
```

With few votes, a film is pulled to the average C; with many, it keeps its own rating R. We set **m = the 90th percentile of votes** in the comparison pool, and figure 02 shows how the ranking changes from the 50th to the 99th percentile, so the choice is transparent. This is the posterior mean of a Normal-Normal Bayesian model, the same idea as actuarial credibility weighting (Bühlmann, 1967).

### Most popular actor: two definitions side by side
- **Reach** = Σ billing weight × votes. Who brings an audience?
- **Net goodwill** = Σ billing weight × votes × (shrunk rating − average). Who brings an audience *that likes the film*?

The billing weight is 1/√rank (lead 1.00, 4th billed 0.50). We use the rating *above average* because multiplying by the raw rating barely changes the order: ratings vary from about 5 to 8.5 while votes vary 1,000-fold.

### The hit metric: three methods, majority vote
Axes: **quality** = raw `averageRating` (films with ≥ 1,000 votes), and **reach** = log-votes relative to other films *released the same year* (`reach_z`). We do not cluster on the shrunk rating, because it would erase the Hidden Gem quadrant. Year adjustment stops "hit" from simply meaning "old enough to have collected lots of votes".

| Method | Role | How its setting is chosen |
|---|---|---|
| KMeans | Partitions films into k groups, named by matching to quadrant corners | Elbow (Kneedle) + silhouette; default k = 4 from the hypothesis |
| DBSCAN | Outlier detector: dense = Typical, sparse = Exceptional | eps at the knee of the k-distance plot |
| Median split | No-algorithm baseline | Medians of each axis |

`is_hit` = at least 2 of 3 agree. **Mega-hit** = a hit DBSCAN also flags as exceptional (the AAA tier). Agreement is reported with Cohen's kappa rather than "% agree", because hits are rare and "% agree" flatters methods that say "no" to everything.

**Be ready for this question:** if the elbow does not land on 4, that is a finding, not a failure. It means the data is one continuous cloud and the four quadrants are a business lens we impose. Present both the elbow's answer and the hypothesis, and use `--k` to compare.

### Trends
- **IMDb:** per genre, share of films made vs. share of audience votes (supply vs. demand). Multi-genre films are split evenly so shares sum to 100%.
- **MovieLens:** share of ratings by genre, plus yearly activity, average rating, age of films being rated, and how concentrated attention is on the top 1% of films. The partial final year (MovieLens 20M stops in March 2015) is dropped.

---

## 5. Running options

```bash
python scripts/run_part_a.py --help
python scripts/run_part_a.py --imdb-dir /path/to/imdb --movielens-dir /path/to/ml-20m
python scripts/run_part_a.py --k 3          # try a different number of KMeans clusters
python scripts/run_part_a.py --year 2024    # same EDA for another year
python scripts/run_part_a.py --no-plots     # tables only, faster
python scripts/run_part_a.py --refresh      # ignore the cache and re-parse everything
```

Paths can also come from environment variables: `NEXTAAA_IMDB_DIR`, `NEXTAAA_MOVIELENS_DIR`, `NEXTAAA_CACHE_DIR`, `NEXTAAA_OUTPUT_DIR`. To change a threshold (e.g. minimum votes), edit `AnalysisConfig` in `config.py`, or in the notebook use `cfg = replace(cfg, min_votes_for_clustering=5000)`.

---

## 6. Outputs

```
outputs/
  summary.json                      headline numbers for the deck
  figures/01_rating_vs_votes_2023.png    leaderboard with lines of equal shrunk rating
  figures/02_m_sensitivity_2023.png      is the #1 robust to the choice of m?
  figures/03_actors_2023.png             Reach vs. Net goodwill
  figures/04_elbow_silhouette.png        choosing k
  figures/05_k_distance.png              choosing DBSCAN's eps
  figures/06_cluster_maps.png            KMeans vs. DBSCAN vs. median split
  figures/07_hit_tiers.png               the final hit label
  figures/08_imdb_genre_trends.png       supply vs. demand by genre
  figures/09_movielens_genre_trends.png  (MovieLens) what users rated
  figures/10_movielens_yearly.png        (MovieLens) behaviour over time
  figures/11_divisiveness_check.png      (MovieLens) is "Divisive" really divisive?
  tables/*.csv                      every number behind every chart
  tables/hit_labels.csv             one row per film with all labels (input to Part B)
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not find 'title.basics.tsv.gz'` | Put the file in `data/imdb/` or pass `--imdb-dir`. |
| First run seems stuck | It is parsing the big files; watch the log. Later runs use the cache. |
| `MovieLens 20M not found` warning | Optional. Add the files to `data/ml-20m/` to enable those sections. |
| Out of memory | Close other programs; `dbscan_max_points` in `config.py` can be lowered. |
| Results look odd after changing code | Run with `--refresh` to rebuild the cache. |

---

## 8. References

- Bühlmann, H. (1967). Experience rating and credibility. *ASTIN Bulletin*, 4(3).
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20.
- Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a "Kneedle" in a haystack: detecting knee points in system behavior. *ICDCS Workshops*.
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3).
- IMDb. Weighted average ratings; IMDb non-commercial datasets (developer.imdb.com/non-commercial-datasets).
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: history and context. *ACM TiiS*, 5(4).

---

## 9. v2: money, five methods, by-year and all-time hits

### 9.1 Extra data (both optional; the code runs without them)

```
data/
  tmdb/     the Kaggle download, untouched (archive.zip) OR the extracted CSV (any *.csv name)
            https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
  cpi/      US CPI from FRED: https://fred.stlouisfed.org/series/CPIAUCSL -> Download -> CSV
```

- **TMDB** gives budget, revenue and plot text, joined on `imdb_id` = IMDb `tconst`. Without it, v2 runs the audience stage only and says so.
- **CPI** is only for *showing* 2023 dollars and for a consistent AAA-scale budget tier ($100M+ in 2023 $). The clustering does not need it (see 9.3). Without it, slides show nominal dollars.
- **`hdbscan` package** (in `requirements.txt`) provides the exact GLOSH outlier score. Without it, v2 uses a documented approximation and logs a warning.

The TMDB loader handles the dataset's known problems: 0 meaning "unknown" (most films), figures typed in millions, revenue copied from budget, duplicate and missing `imdb_id`s, and plot text with commas and line breaks. It writes a data-quality report (`summary.json` → `tmdb`) worth an appendix slide.

### 9.2 The hit definition: a two-stage gate

| Stage | Axes (each vs. same-year films) | Hit corner |
|---|---|---|
| Audience | quality (rating) × reach (votes) | Hit |
| Commercial | ROI multiple × box office | Blockbuster, **and** ROI ≥ 2.5× |

|  | commercial hit | not a commercial hit |
|---|---|---|
| **audience hit** | **AAA hit** | Loved, unprofitable |
| **not an audience hit** | Profitable, not loved | Neither |

- **ROI multiple** = worldwide box office ÷ production budget. **Break-even = 2.5×**, the industry rule of thumb: theatres keep about half of ticket sales, and marketing costs about as much as production.
- The absolute 2.5× gate is applied on top of the relative vote, because in a weak year "above the year's median" can still mean losing money.
- **AAA mega-hit** = unanimous (5 of 5) in both stages.
- **Hit score** = mean of the four within-year percentiles. It ranks films inside each year and all time.

### 9.3 One model for "by year" and "all time"

Every axis becomes a **robust z-score against films released the same year**: (x − year median) / (year IQR / 1.349). Thin years borrow neighbours, widening to ±1 … ±5 years until there are enough reference films. This removes inflation, IMDb's user growth and rating drift in one step. One fit then serves both lists:

- **by year**: the AAA hits of each year, ranked by hit score (`aaa_hits_by_year.csv`, `yearly_summary.csv` with the "hit of the year");
- **all time**: films that dominated their own year most completely (`aaa_hits_all_time.csv`, figure 15), plus a real-dollar box-office cross-check when CPI is present.

### 9.4 The five voters

| Method | Type | A "hit vote" means | Setting |
|---|---|---|---|
| KMeans | partition | in the top-right cluster | k from elbow + silhouette (default 4) |
| Median split | partition | above its year's median on both axes | none |
| DBSCAN | outlier | outside the main dense mass **and** top-right | eps at the k-distance knee |
| HDBSCAN | outlier | GLOSH score in the top 10% **and** top-right | min_cluster_size = max(50, 1%) |
| Isolation Forest | outlier | anomaly (top 10%) **and** top-right | 300 trees, contamination 0.10 |

Hit = **≥ 3 of 5**; mega = **5 of 5**. Agreement is reported as Cohen's kappa (figure 12).

**Why GLOSH instead of HDBSCAN's "noise" label?** On one continuous cloud of films, HDBSCAN either forces artificial splits or labels most films noise. On the synthetic test data (shaped like the real data) we saw about 40 to 95% noise depending on settings, and a kappa of 0.81 with KMeans, so it was acting as a second partitioner rather than an outlier detector. The method summary table (`aud_method_summary.csv`, column `noise_pct`) reports HDBSCAN's noise share, so you can confirm this on the real data. GLOSH is the HDBSCAN authors' own outlier measure for this purpose: 1 − λₓ / λ_max(C), comparing the density at which a film leaves its cluster with the cluster's densest point.

### 9.5 Running options

```bash
python scripts/run_part_a_v2.py --help
python scripts/run_part_a_v2.py --tmdb-dir /path/to/tmdb --cpi-dir /path/to/cpi
python scripts/run_part_a_v2.py --k-audience 3 --k-commercial 4   # after reading figures 06 and 09
```

Key settings in `config.py` (section "v2"): `breakeven_multiple`, `aaa_budget_usd`, `min_budget_usd`, `finance_min_year`, `max_year_window`, `glosh_quantile`, `iforest_contamination`, `hit_min_votes`, `mega_min_votes`.

### 9.6 v2 outputs (`outputs/v2/`)

| Figure | Shows |
|---|---|
| 01, 02 | 2023 leaderboard by shrunk rating; robustness to m |
| 03 | Most popular 2023 actor: Reach, Net goodwill, Box office |
| 04 | Budget vs. box office with break-even lines, by verdict |
| 05 | Budget tiers: break-even rate, AAA-hit rate, median ROI |
| 06 to 08 | Audience stage: choosing k, choosing eps, each method's votes |
| 09 to 11 | Commercial stage: the same |
| 12 | Method agreement (Cohen's kappa) |
| 13 | Verdict map: audience score vs. commercial score |
| 14 | Hit rates by release year |
| 15 | All-time AAA hits (percentile heatmap) |
| 16 | Genre economics: AAA-hit rate and median ROI |
| 17 to 19 | Trends: IMDb genres; MovieLens behaviour; divisiveness check |

`data/cache/hit_labels_v2.pkl` holds every film with its verdict, scores, money and **TMDB plot text**. It is Part B's input, and it replaces the need for the Wikipedia plots dataset.

### 9.7 Extra references (v2)

- Campello, R., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD*.
- Campello, R., Moulavi, D., Zimek, A., & Sander, J. (2015). Hierarchical density estimates for data clustering, visualization, and outlier detection. *ACM TKDD*, 10(1). (GLOSH)
- McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. *Journal of Open Source Software*, 2(11).
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *IEEE ICDM*. Extended in *ACM TKDD* 6(1), 2012.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1).
- Ramaswamy, S., Rastogi, R., & Shim, K. (2000). Efficient algorithms for mining outliers from large data sets. *SIGMOD*. (kNN-distance fallback)
- Break-even rule of thumb: "Blockbuster movie economics 101", Brandstofans (Substack), on studio rental share (~45%) and P&A ≈ production cost.

---

## 10. Part B: designing the next AAA title

`notebooks/03_part_b_next_aaa_title.ipynb` · `scripts/run_part_b.py` · package `src/nextaaa/partb/` · outputs in `outputs/partb/`

**What it does.** A studio deciding how to make its next AAA title needs a *film to make*, not a forecast for films that already exist. A classifier can only score a film you give it, so Part B puts one inside a search:

```
LLM generator --> concepts (JSON) --> validate --> novelty filter --> judge --> scores + drivers
      ^                                                                              |
      +----------------- scored history, lowest to highest (OPRO) -------------------+
```

Agreed scope: a **new** film · **original** stories only · **real** directors and actors, chosen by id from track-record tables.

### 10.1 Set up the LLMs

```bash
cp .env.example .env       # then paste your OpenAI key into OPENAI_API_KEY
python scripts/run_part_b.py --demo                  # offline: synthetic data + mock LLM (about 1 minute)
python scripts/run_part_b.py --rounds 3 --islands 2 --no-ablation --no-llm-baseline   # cheap first live run
python scripts/run_part_b.py                         # full run
```

With an OpenAI key, the whole setup is one key and three model names:

| Setting | Default in `.env.example` | Used for |
|---|---|---|
| `OPENAI_API_KEY` | (your key) | shared by every role |
| `GENERATOR_MODEL` | `gpt-6-sol` | writing concepts |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | novelty filter, plot features (empty = local TF-IDF fallback) |
| `BASELINE_MODEL` + `BASELINE_KNOWLEDGE_CUTOFF` | `gpt-5-mini`, `2024-05` | backtest only: can an LLM predict hits? (empty = off) |
| `LLM_MODE` | `auto` | `auto` / `mock` / `live` |

A role is switched on by naming its model. `OPENAI_BASE_URL` and `OPENAI_API_VERSION` (the OpenAI SDK's own names) point every role at another provider; `<ROLE>_API_KEY`, `<ROLE>_BASE_URL` and `<ROLE>_API_VERSION` override one role, so providers can be mixed. `.env.example` has ready snippets for:

- **Azure OpenAI:** `OPENAI_BASE_URL=https://<resource>.openai.azure.com` (Azure's v1 API, no api-version needed; set `OPENAI_API_VERSION` only for the classic style). Model names are your deployment names.
- **Claude API:** `GENERATOR_BASE_URL=https://api.anthropic.com/v1`, through Anthropic's OpenAI-compatible layer. It ignores JSON mode (we parse JSON from the text), has no embeddings, and rejects temperature below 1 on newer models (the client then drops temperature). Replies are capped at 16000 tokens by default (`GENERATOR_MAX_TOKENS`). `ANTHROPIC_WORKSPACE_ID` is only for keys not tied to a workspace. Claude 5.5 models were trained to June 2026, so they cannot be the baseline yet.
- **Any OpenAI-compatible server** (vLLM, Ollama, OpenRouter): a base URL ending in `/v1`.

**With no key or no model names, everything runs in mock mode**, and the outputs say so in bold. Every prompt and reply is logged to `outputs/partb/llm_log.jsonl` (keys never are); the run log prints which model each role uses.

The client handles the usual model differences: models that reject `temperature`, deployments without JSON mode, replies wrapped in code fences, rate limits (exponential backoff) and malformed JSON (one repair request). These paths are tested against a fake OpenAI-compatible server.

### 10.2 The judge

- **Target:** Part A v2's AAA hit (films with budget and revenue).
- **Features, pre-release only:** genres, runtime, budget (2023 $ with CPI), English-language, source (original / sequel / adaptation, from TMDB keywords and title patterns), director and lead-actor **track records as of release** (strictly earlier years; smoothed rates), and the plot as a 16-dimension PCA of its embedding. Release year is deliberately excluded.
- **Candidates:** logistic regression and gradient boosting (the brief's "typical" models, kept as baselines), plus **TabPFN** (*Nature* 2025) when installed (`pip install tabpfn`). Each is Platt-calibrated.
- **Honest backtest:** choose the model on validation years, report every candidate on the last three complete years, then refit on all years. Metrics: PR-AUC and its lift over the base rate, ROC-AUC, Brier, R-precision. Figures 01 to 03.
- **Per concept:** P(AAA hit), a bootstrap range, **drivers** (score change when one feature group is replaced by an average film's), and **familiarity** (see 10.3).

### 10.3 Guardrails against gaming a learned judge

| Guardrail | Rule |
|---|---|
| Real people only | director and lead ids must come from the shortlists (alive, active in the last 8 years, at least 2 films) |
| Novelty | reject if the plot is closer to an existing film than 99% of real films are to their nearest neighbour (relative, so it works with any embedding model; exact duplicates are ignored when setting it). Repeats of earlier concepts are rejected the same way |
| Familiarity | the *invented* parts (genre mix, budget, runtime, story) are compared, on percentile ranks with equal weight per group, with past **AAA-scale** films. Beyond their 99th percentile, a concept is flagged, counts half in the loop, and can't enter the portfolio |
| Control arm | the same number of calls with no feedback; figure 04 shows whether feedback helps |

### 10.4 Outputs

| File | What |
|---|---|
| `pitch_cards.md` | the final portfolio: one card per concept, with premise, talent records, P(AAA hit) and range, drivers, closest existing film |
| `figures/01`–`03` | backtest PR curves, calibration, what the judge relies on |
| `figures/04`, `05` | loop vs. control; every concept (originality vs. score) |
| `figures/06` | LLM few-shot vs. the judge on films released after the LLM's cutoff |
| `tables/*.csv` | every number, including every concept generated |

### 10.5 Settings (`PartBConfig` in `config.py`)

`n_islands` (3), `n_rounds` (5), `concepts_per_call` (8), `history_top_k` (20), `generator_temperature` (1.0), `novelty_percentile` (99), `ood_percentile` (99), `min_budget_usd` / `max_budget_usd` ($100M–$350M), `shortlist_*`, `n_bootstrap` (5), `candidate_models`, `llm_baseline_max_films` (200). A full live run is 30 generator calls (two arms × 3 islands × 5 rounds) plus up to 200 baseline calls.

### 10.6 References (Part B)

- Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., & Chen, X. (2024). Large language models as optimizers. *ICLR*. (OPRO)
- Romera-Paredes, B., et al. (2024). Mathematical discoveries from program search with large language models. *Nature*, 625. (FunSearch: LLM + evaluator, islands)
- Hollmann, N., et al. (2025). Accurate predictions on small data with a tabular foundation model. *Nature*, 637. (TabPFN)
- Platt, J. (1999). Probabilistic outputs for support vector machines. (sigmoid calibration)
- Predicting Movie Hits Before They Happen with LLMs (2025). *UMAP*. (LLM baseline)
- Luan, Y., & Kim, Y. J. (2022). *PLOS ONE*. (audiences rate moderate novelty highest; ratings, not box office)
