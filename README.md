# GHit! · Optimized Movie Concept Generator (OMCG)

GHit! proposes original film concepts with a high chance of becoming an **AAA hit**: a film that audiences love *and* that pays off big at the box office. It has three parts:

1. **AAA label:** a data-driven definition of a hit (Part A).
2. **Judge:** a calibrated classifier that scores a concept's chance of becoming an AAA hit (Part B).
3. **Generator:** an LLM whose prompt is optimised with the judge's scores, OPRO and FunSearch style (Part B).

The design, the reasoning and the results are in the technical document, **GHit! · OMCG Solution Design** (PDF). This README covers how to run the code.

| Question from the brief | Where it is answered |
|---|---|
| Highest-rated film of 2023 (rating vs. votes) | `ratings.py` · `outputs/v2/figures/01`, `02` |
| Most popular actor of 2023 | `popularity.py` · figure `03` |
| A metric for a "hit" film | `normalize.py`, `ensemble.py`, `hits.py` · figures `04`–`16` |
| Trends in audience preferences | `hits.py`, `trends.py` · figures `14`, `16`–`19` |
| The next AAA title | `partb/` · `outputs/partb/pitch_cards.md` |

---

## 1. Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Check the setup on synthetic data (about a minute, no downloads, no API key)
python scripts/run_part_a_v2.py --demo
python scripts/run_part_b.py --demo
pytest                                                  # all tests should pass

# Real run: add the data (section 2) and an LLM key (section 4), then
python scripts/run_part_a_v2.py
python scripts/run_part_b.py
```

If you want to follow each step with explanations, use the notebooks instead: `notebooks/02_part_a_v2_money_and_hits.ipynb`, then `notebooks/03_part_b_next_aaa_title.ipynb`.

> Demo mode uses fake data and a mock LLM. Its numbers are meaningless; never present them.

---

## 2. Data

```
data/
  imdb/     title.basics, title.ratings, title.principals, name.basics (.tsv.gz)   https://datasets.imdbws.com/
  tmdb/     the Kaggle TMDB movies dataset (zip or CSV)                             kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
  cpi/      US CPI (CPIAUCSL) as CSV                                                https://fred.stlouisfed.org/series/CPIAUCSL
  ml-20m/   ratings.csv, movies.csv, links.csv                                      https://grouplens.org/datasets/movielens/20m/
```

- **IMDb is required.** If a file is missing, the run stops and says which file and where to get it.
- **TMDB** adds budget, revenue and plot text, joined on `imdb_id`. The commercial gate and Part B need it.
- **CPI** (optional) converts dollars to 2023 dollars. **MovieLens** (optional) powers the trend and divisiveness charts.
- The first run streams the large files and caches them in `data/cache/`. That takes several minutes; later runs take seconds. Use `--refresh` to rebuild the cache. 8 GB of RAM is enough.

---

## 3. Part A: the AAA label

```bash
python scripts/run_part_a_v2.py                                   # all outputs to outputs/v2/
python scripts/run_part_a_v2.py --k-audience 3 --k-commercial 4   # after reading figures 06 and 09
python scripts/run_part_a_v2.py --year 2024 --no-plots
```

**How a film becomes an AAA hit:**

1. Score every film against films from its own release year (percentiles; thin years borrow neighbouring years).
2. **Audience gate:** quality (rating) × reach (votes). **Commercial gate:** ROI × box office, plus ROI ≥ 2.5× (break-even).
3. Five methods vote at each gate: KMeans, median split, DBSCAN, HDBSCAN (GLOSH), Isolation Forest. A film passes a gate with at least 3 of 5 votes.
4. **AAA hit** = passes both gates. **Mega-hit** = all five methods agree at both gates.

Key outputs: `summary.json`, `tables/aaa_hits_by_year.csv`, `tables/aaa_hits_all_time.csv`, `tables/top_rated_2023.csv`, and figures `01`–`19`. `data/cache/hit_labels_v2.pkl` is the input to Part B.

The original audience-only version (v1: `run_part_a.py`, notebook `01`) is kept for comparison only.

---

## 4. Part B: the judge and the generator

### Connect an LLM

Copy `.env.example` to `.env` (git-ignored) and fill in:

| Setting | Default | Used for |
|---|---|---|
| `OPENAI_API_KEY` | your key | every role |
| `GENERATOR_MODEL` | `gpt-6-sol` | writing concepts |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | novelty check and plot features; empty = free local TF-IDF |
| `BASELINE_MODEL`, `BASELINE_KNOWLEDGE_CUTOFF` | `gpt-5-mini`, `2024-05` | optional backtest of an LLM on its own; empty = off |
| `LLM_MODE` | `auto` | `auto`, `mock` or `live` |

Other providers work through `OPENAI_BASE_URL`, or `<ROLE>_BASE_URL` / `<ROLE>_API_KEY` for a single role: Azure OpenAI, the Claude API, or any OpenAI-compatible server. `.env.example` has ready-made snippets. With no key, everything runs in mock mode and the outputs say so.

### Run

```bash
python scripts/run_part_b.py --rounds 3 --islands 2 --no-ablation --no-llm-baseline   # cheap first live run
python scripts/run_part_b.py              # full run: 3 islands × 5 rounds × 2 arms = 30 generator calls
python scripts/run_part_b.py --rescore    # re-judge the last run's concepts, no LLM calls
```

Run `--rescore` after any change to the judge. In the notebook, setting `RESCORE = True` in the first cell does the same.

**What happens:**

1. **Judge:** trained on pre-release features only: genres, budget, runtime, source, director and lead track records as of release, and the plot embedding. The model is chosen on 2019–21 and tested once on 2022–24. Each score is the median of 20 calibrated refits on 80% of films, with a 10th–90th percentile range.
2. **Generator loop, per genre island:**
   1. Build the meta-prompt: task, evidence, the 20 best scored concepts so far, and rejections.
   2. GPT-6 Sol writes 8 concepts.
   3. Validate them, then run the novelty and familiarity checks.
   4. Score them with the judge and store the results.
   5. Repeat for 5 rounds.
3. **Control arm:** the same number of calls without feedback, to measure what the feedback adds.
4. **Portfolio:** the best familiar concept from each island.

Outputs in `outputs/partb/`:
- `pitch_cards.md`: the portfolio.
- `tables/concepts_all.csv`: every concept.
- figures `01`–`05`.
- `llm_log.jsonl`: every prompt and reply (keys are never logged).

---

## 5. Code layout

```
src/nextaaa/
  config.py                          every path and tunable number, with the reason for each default
  io_imdb.py, io_tmdb.py, movielens.py   data loaders (streamed and cached)
  finance.py                         budget, revenue, ROI and CPI
  ratings.py, popularity.py, trends.py   the 2023 questions and trends
  normalize.py, ensemble.py, hits.py     within-year scores, five voters, AAA label
  pipeline_v2.py                     Part A end to end
  partb/                             llm.py (clients), talent.py, dataset.py, scorer.py (judge),
                                     prompts.py, loop.py (OPRO × FunSearch), pipeline.py (Part B end to end)
scripts/                             command-line entry points
notebooks/                           guided walkthroughs
tests/                               unit tests and end-to-end tests on synthetic data
```

All thresholds live in `config.py` (`AnalysisConfig`, `PartBConfig`). Every random step uses `random_state`, so runs are reproducible.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not find 'title.basics.tsv.gz'` | Put the file in `data/imdb/` or pass `--imdb-dir`. |
| First run seems stuck | It is parsing the large files; later runs use the cache. |
| Part B says MOCK mode | `.env` is missing, or `OPENAI_API_KEY` / `GENERATOR_MODEL` is empty. The run log shows which model each role uses. |
| A reply was cut off | Raise `GENERATOR_MAX_TOKENS` in `.env`. |
| Results look stale after a code change | Run with `--refresh` (data) or `--rescore` (judge). |

---

## 7. Key references

- Yang et al. (2024). Large Language Models as Optimizers (OPRO). *ICLR*.
- Romera-Paredes et al. (2024). Mathematical discoveries from program search with large language models (FunSearch). *Nature* 625.
- Trabucco, Kumar, Geng & Levine (2021). Conservative Objective Models for Effective Offline Model-Based Optimization. *ICML*.
- Campello et al. (2015). Hierarchical density estimates for data clustering, visualization, and outlier detection (GLOSH). *ACM TKDD* 10(1).
- Liu, Ting & Zhou (2008). Isolation Forest. *IEEE ICDM*.
- Hollmann et al. (2025). Accurate predictions on small data with a tabular foundation model (TabPFN). *Nature* 637.

Full method details and the remaining references are in the technical document and in each module's docstring.
