"""The judge: P(AAA hit | pre-release features), backtested on later years.

Candidates (the brief asks to go beyond logistic regression / XGBoost, so those
are the BASELINES the novel parts must beat):
    logistic   standardised logistic regression, class-balanced
    hgb        histogram gradient boosting (scikit-learn's XGBoost-style model)
    tabpfn     TabPFN (Hollmann et al., Nature 2025): a transformer pre-trained on
               millions of synthetic tables that does approximate Bayesian
               prediction "in context". Optional (pip install tabpfn); on CPU we
               subsample to `tabpfn_max_train` rows, keeping every hit.

Every candidate is wrapped in sigmoid (Platt) calibration with internal CV, so
"0.30" means roughly "30% of concepts like this became AAA hits". Hits are rare
(a few %), so models are trained class-balanced and calibrated afterwards.

Honest model selection (no peeking at the test years)
    train  : films up to (test_start - validation_years - 1)
    valid  : the next `validation_years` years   -> choose the model (PR-AUC)
    test   : test_start .. test_end              -> report every candidate
Metrics: PR-AUC (average precision; the right headline when positives are
rare), its lift over the base rate, ROC-AUC, Brier score, and R-precision
(precision among the top-k ranked films, k = number of real hits).

For new concepts the chosen model is refit on all years, and each score gets:
    p_lo / p_hi       10th-90th percentile over bootstrap refits (uncertainty)
    ood_pct           how unusual the concept is: its mean distance to its 10
                      nearest past films, as a percentile of the same distance for
                      past AAA-scale films (budget >= $100M), the population a
                      concept belongs to. Design choices:
                      * only the INVENTED parts count (genre mix, budget, runtime,
                        story). That is where a generator could game the judge with
                        combinations no film ever had. Talent track records are real
                        data about real people, and the shortlist is by design the
                        top of the talent distribution, so they stay in the score
                        but are left out of this check;
                      * percentile-rank features, not z-scores, so skewed columns
                        cannot dominate;
                      * each group weighs the same, so the 16 plot columns do not
                        drown out the budget;
                      * compared with blockbusters, not all films, because every
                        concept is a blockbuster by design.
    drivers           occlusion: score change when one feature group is replaced
                      by an average film's values ("+ director track record")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from ..config import PartBConfig

logger = logging.getLogger(__name__)

MODEL_LABELS = {"logistic": "Logistic regression", "hgb": "Gradient boosting (HGB)", "tabpfn": "TabPFN"}


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

class TabPFNSubsample(ClassifierMixin, BaseEstimator):
    """TabPFN with a row cap: keeps every positive, samples negatives."""

    def __init__(self, max_train: int = 10000, random_state: int = 42):
        self.max_train = max_train
        self.random_state = random_state

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier  # optional dependency
        from tabpfn.constants import ModelVersion

        classifier = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST)

        X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_neg = max(0, min(len(neg), self.max_train - len(pos)))
        idx = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
        try:
            self.model_ = TabPFNClassifier(random_state=self.random_state)
        except TypeError:
            self.model_ = TabPFNClassifier()
        self.model_.fit(X[idx], y[idx])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))


def make_model(name: str, bcfg: PartBConfig):
    rs = bcfg.random_state
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))
    if name == "hgb":
        return HistGradientBoostingClassifier(learning_rate=0.01, max_iter=300, max_leaf_nodes=15,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              early_stopping=True, n_iter_no_change=20,
                                              validation_fraction=0.15, class_weight="balanced",
                                              random_state=rs)
    if name == "tabpfn":
        return TabPFNSubsample(max_train=bcfg.tabpfn_max_train, random_state=rs)
    raise ValueError(f"Unknown model {name!r}")


def calibrated(name: str, bcfg: PartBConfig, n_pos: int):
    folds = int(max(2, min(3, n_pos // 5)))
    return CalibratedClassifierCV(make_model(name, bcfg), method="sigmoid",
                                  cv=StratifiedKFold(folds, shuffle=True, random_state=bcfg.random_state))


def available_models(bcfg: PartBConfig) -> list[str]:
    names = list(bcfg.candidate_models)
    if "tabpfn" in names:
        try:
            import tabpfn  # noqa: F401
        except ImportError:
            logger.info("TabPFN not installed (pip install tabpfn): skipping that candidate.")
            names.remove("tabpfn")
    return names


# ---------------------------------------------------------------------------
# Metrics and backtest
# ---------------------------------------------------------------------------

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    out = {"films": len(y), "hits": int(y.sum()), "base_rate": float(y.mean()) if len(y) else np.nan}
    if y.sum() == 0 or y.sum() == len(y):
        return {**out, "pr_auc": np.nan, "lift": np.nan, "roc_auc": np.nan,
                "brier": float(brier_score_loss(y, p)) if len(y) else np.nan, "r_precision": np.nan}
    k = int(y.sum())
    top = np.argsort(-p)[:k]
    pr = float(average_precision_score(y, p))
    return {**out, "pr_auc": pr, "lift": pr / out["base_rate"], "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "r_precision": float(y[top].mean())}


def split_years(max_year: int, bcfg: PartBConfig) -> dict:
    test_end = bcfg.test_end_year or max_year - 1
    test_start = bcfg.test_start_year or test_end - 2
    train_end = test_start - 1
    return {"select_train_end": train_end - bcfg.validation_years, "valid": (train_end - bcfg.validation_years + 1, train_end),
            "train_end": train_end, "test": (test_start, test_end)}


@dataclass
class Backtest:
    years: dict
    selection: pd.DataFrame          # validation metrics per candidate
    test: pd.DataFrame               # test metrics per candidate
    chosen: str
    predictions: pd.DataFrame        # test films: tconst, year, y, p_<model>
    group_importance: pd.DataFrame   # PR-AUC drop when a feature group is shuffled (chosen model)


def _fit_predict(name, bcfg, Xtr, ytr, Xte):
    model = calibrated(name, bcfg, int(ytr.sum())).fit(Xtr, ytr)
    return model, model.predict_proba(Xte)[:, 1]


def grouped_permutation_importance(model, X: pd.DataFrame, y: np.ndarray, groups: dict,
                                   n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Drop in test PR-AUC when all columns of a group are shuffled together."""
    rng = np.random.default_rng(random_state)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for gname, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"group": gname, "pr_auc_drop": float(np.mean(drops)), "sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False).reset_index(drop=True)


def run_backtest(X: pd.DataFrame, y: pd.Series, years: pd.Series, max_year: int,
                 bcfg: PartBConfig, groups: dict) -> Backtest:
    yrs = split_years(max_year, bcfg)
    names = available_models(bcfg)
    y = y.astype(int).to_numpy()
    yv = years.to_numpy()
    sel_tr = yv <= yrs["select_train_end"]
    sel_va = (yv >= yrs["valid"][0]) & (yv <= yrs["valid"][1])
    tr = yv <= yrs["train_end"]
    te = (yv >= yrs["test"][0]) & (yv <= yrs["test"][1])
    logger.info("Backtest: select on %d-%d, train <= %d (%d films, %d hits), test %d-%d (%d films, %d hits)",
                *yrs["valid"], yrs["train_end"], tr.sum(), y[tr].sum(), *yrs["test"], te.sum(), y[te].sum())
    if y[tr].sum() < 4 or y[te].sum() < 1:
        raise ValueError("Too few AAA hits to backtest; widen the year range or check Part A labels.")

    selection, test, preds, models = [], [], {}, {}
    for name in list(names):
        try:
            if y[sel_tr].sum() >= 4 and y[sel_va].sum() >= 1:
                _, pv = _fit_predict(name, bcfg, X[sel_tr], y[sel_tr], X[sel_va])
                selection.append({"model": name, **metrics(y[sel_va], pv)})
            model, pt = _fit_predict(name, bcfg, X[tr], y[tr], X[te])
        except Exception as exc:  # e.g. TabPFN weights unavailable offline
            logger.warning("Model %s failed (%s: %s); dropping it.", name, type(exc).__name__, exc)
            names.remove(name)
            continue
        models[name], preds[name] = model, pt
        test.append({"model": name, **metrics(y[te], pt)})
    selection, test = pd.DataFrame(selection), pd.DataFrame(test)
    if selection.empty or selection["pr_auc"].isna().all():
        chosen = test.sort_values("pr_auc", ascending=False)["model"].iloc[0]
        logger.warning("Validation years had no hits: chose %s on test PR-AUC (optimistic).", chosen)
    else:
        chosen = selection.sort_values("pr_auc", ascending=False)["model"].iloc[0]
    logger.info("Chosen judge: %s (validation PR-AUC). Test results:\n%s", chosen,
                test[["model", "pr_auc", "lift", "roc_auc", "brier", "r_precision"]].round(3).to_string(index=False))

    pred_df = pd.DataFrame({"tconst": X.index[te], "year": yv[te], "y": y[te]})
    for name, p in preds.items():
        pred_df[f"p_{name}"] = p
    imp = grouped_permutation_importance(models[chosen], X[te], y[te], groups,
                                         random_state=bcfg.random_state)
    return Backtest(years=yrs, selection=selection, test=test, chosen=chosen,
                    predictions=pred_df, group_importance=imp)


# ---------------------------------------------------------------------------
# The final judge for new concepts
# ---------------------------------------------------------------------------

@dataclass
class Judge:
    name: str
    model: object
    boot: list
    X_ref: pd.DataFrame                    # training features (for OOD and drivers)
    groups: dict
    ref_mask: np.ndarray | None = None     # which training films set the OOD scale
    ood_groups: tuple = ("genre mix", "budget", "runtime", "story (plot text)")
    reference: pd.Series = field(init=False)
    _nn: NearestNeighbors = field(init=False)
    _ranks: QuantileTransformer = field(init=False)
    _imputer: SimpleImputer = field(init=False)
    _train_dist: np.ndarray = field(init=False)

    def __post_init__(self):
        self.reference = self.X_ref.mean()
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(self.X_ref)
        Xi = self._imputer.transform(self.X_ref)
        self._ranks = QuantileTransformer(n_quantiles=min(1000, len(Xi)), output_distribution="uniform",
                                          random_state=0).fit(Xi)
        # Equal weight per checked group (1/sqrt(group size)); unchecked groups weigh 0.
        cols = list(self.X_ref.columns)
        self._w = np.zeros(len(cols))
        for gname, gcols in self.groups.items():
            idx = [cols.index(c) for c in gcols if c in cols]
            if idx and gname in self.ood_groups:
                self._w[idx] = 1 / np.sqrt(len(idx))
        Z = self._ranks.transform(Xi) * self._w
        self._nn = NearestNeighbors(n_neighbors=11).fit(Z)
        d, _ = self._nn.kneighbors(Z)
        dist = d[:, 1:].mean(axis=1)  # exclude self
        ref = self.ref_mask if self.ref_mask is not None and self.ref_mask.sum() >= 30 else np.ones(len(dist), bool)
        self._train_dist = np.sort(dist[ref])

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.model.predict_proba(X)[:, 1]
        out = pd.DataFrame({"p_aaa_hit": p}, index=X.index)
        if self.boot:
            B = np.column_stack([m.predict_proba(X)[:, 1] for m in self.boot])
            out["p_lo"], out["p_hi"] = np.percentile(B, 10, axis=1), np.percentile(B, 90, axis=1)
        else:
            out["p_lo"] = out["p_hi"] = np.nan
        Z = self._ranks.transform(self._imputer.transform(X)) * self._w
        d, _ = self._nn.kneighbors(Z, n_neighbors=10)
        out["ood_pct"] = 100 * np.searchsorted(self._train_dist, d.mean(axis=1)) / len(self._train_dist)
        return out

    def drivers(self, X: pd.DataFrame, top: int = 3) -> list[list[tuple[str, float]]]:
        """Per row: the feature groups that move its score most vs. an average film."""
        base = self.model.predict_proba(X)[:, 1]
        deltas = {}
        for gname, cols in self.groups.items():
            cols = [c for c in cols if c in X.columns]
            if not cols:
                continue
            Xm = X.copy()
            Xm[cols] = self.reference[cols].to_numpy()
            deltas[gname] = base - self.model.predict_proba(Xm)[:, 1]
        D = pd.DataFrame(deltas)
        out = []
        for i in range(len(X)):
            row = D.iloc[i].sort_values(key=np.abs, ascending=False).head(top)
            out.append([(g, float(v)) for g, v in row.items()])
        return out


def fit_judge(X: pd.DataFrame, y: pd.Series, name: str, bcfg: PartBConfig, groups: dict,
              ref_mask: np.ndarray | None = None) -> Judge:
    """Refit the chosen model on all labelled years, plus bootstrap copies for the band."""
    y = y.astype(int).to_numpy()
    model = calibrated(name, bcfg, int(y.sum())).fit(X, y)
    rng = np.random.default_rng(bcfg.random_state)
    n_boot = bcfg.n_bootstrap if name != "tabpfn" else min(bcfg.n_bootstrap, 3)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boot = []
    for _ in range(n_boot):  # stratified bootstrap keeps the hit count stable
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boot.append(calibrated(name, bcfg, int(y[idx].sum())).fit(X.iloc[idx], y[idx]))
    logger.info("Judge refit on %d films (%d hits) with %d bootstrap copies.", len(y), y.sum(), n_boot)
    return Judge(name=name, model=model, boot=boot, X_ref=X, groups=groups, ref_mask=ref_mask)
