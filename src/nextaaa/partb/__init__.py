"""Part B: design the next AAA title with a generate -> score -> refine loop.

Module map (read in this order):

    llm.py           LLM and embedding clients, configured from environment variables.
                     Mock mode runs everything offline for testing.
    talent.py        Director and lead-actor track records, as of each film's release.
    dataset.py       Pre-release features for past films and for new concepts.
    scorer.py        The judge: a calibrated classifier, backtested on later years.
    prompts.py       The prompts, the concept schema and its validation.
    loop.py          The loop: islands, OPRO-style refinement, novelty filter, ablation.
    llm_baseline.py  Backtest of an LLM as the predictor (few-shot), vs. the judge.
    plots.py         Part B figures.
    pipeline.py      Runs everything in order and saves outputs to outputs/partb/.
"""
