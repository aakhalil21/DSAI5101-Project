"""Reusable modelling helpers for Member 3 (Q1: which factors best predict heart disease).

All preprocessing is fitted inside a scikit-learn Pipeline, so it only ever learns
from the rows passed to ``fit`` (the training split). The test split is never read here.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .preprocessing import make_preprocessor
except ImportError:  # imported with src/ on sys.path
    from preprocessing import make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
OBSERVED_DIR = ROOT / "data/processed/observed"
RESULTS_DIR = ROOT / "results/modelling"
SEED = 42
TARGET = "heart_disease"
ID_COLUMNS = ["row_id", "duplicate_group"]
TARGET_SOURCES = ["HadHeartAttack", "HadAngina"]
# Healthcare-use variables that can be triggered by the diagnosis itself
# (timing relative to diagnosis unknown -> reverse causality / temporal leakage risk).
POST_DIAGNOSIS_FEATURES = ["ChestScan", "PneumoVaxEver"]
NEVER_PREDICTORS = [TARGET, *ID_COLUMNS, *TARGET_SOURCES]


def load_split(name: str) -> pd.DataFrame:
    """Load the observed training or validation split. The test split is refused on purpose."""
    if name not in {"training", "validation"}:
        raise ValueError("Only 'training' and 'validation' may be loaded during model development.")
    return pd.read_csv(OBSERVED_DIR / f"{name}.csv")


def split_xy(frame: pd.DataFrame, drop: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the target and remove IDs and excluded features."""
    drop = list(drop or [])
    X = frame.drop(columns=[c for c in NEVER_PREDICTORS + drop if c in frame.columns])
    return X, frame[TARGET].astype(int)


def check_no_forbidden_columns(X: pd.DataFrame, forbidden: list[str]) -> None:
    """Fail loudly if an ID, the target, its sources or an excluded feature is still in X."""
    present = sorted(set(forbidden) & set(X.columns))
    if present:
        raise AssertionError(f"Forbidden columns present in X: {present}")


def make_scaled_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Shared preprocessor plus standardisation of the numeric block (for logistic regression).

    The numeric block becomes: median imputation (+ missing flags) -> StandardScaler.
    Categorical handling is unchanged (``Missing`` category + one-hot encoding).
    """
    prep = make_preprocessor(X)
    name, imputer, columns = prep.transformers[0]
    prep.transformers[0] = (name, Pipeline([("impute", imputer), ("scale", StandardScaler())]), columns)
    return prep


def make_variant_preprocessor(X: pd.DataFrame, scale: bool, missing: str = "indicators",
                              keep_height_weight: bool = False) -> ColumnTransformer:
    """Preprocessor for the preparation checks (same steps as the shared one, one thing changed).

    missing="indicators":    numeric median + missing flags; categorical "Missing" category (primary rule)
    missing="no_indicators": numeric median, categorical most frequent, no missing flags (alternative)
    keep_height_weight=True: HeightInMeters and WeightInKilograms are used alongside BMI.
    """
    if missing not in {"indicators", "no_indicators"}:
        raise ValueError("missing must be 'indicators' or 'no_indicators'")
    drop = [] if keep_height_weight else ["HeightInMeters", "WeightInKilograms"]
    features = X.drop(columns=[c for c in drop if c in X.columns])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.columns.difference(numeric, sort=False).tolist()
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=(missing == "indicators")))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    if missing == "indicators":
        cat_impute = SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)
    else:
        cat_impute = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([("missing", cat_impute),
                                  ("one_hot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")


def make_pipeline_for(estimator, X: pd.DataFrame, scale: bool, preprocessor=None,
                      dense: bool = False) -> Pipeline:
    """Wrap a fresh, unfitted preprocessor and the estimator in one Pipeline.

    dense=True makes the preprocessor return a dense matrix; random forests train several
    times faster on dense input and give identical results.
    """
    if preprocessor is None:
        preprocessor = make_scaled_preprocessor(X) if scale else make_preprocessor(X)
    if dense:
        preprocessor.set_params(sparse_threshold=0)
    return Pipeline([("prep", preprocessor), ("model", clone(estimator))])


def fit_and_score(pipe: Pipeline, X_train, y_train, X_val, y_val) -> tuple[Pipeline, np.ndarray, dict]:
    """Fit on training rows, predict validation probabilities, record runtime and warnings."""
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X_train, y_train)
    runtime = time.perf_counter() - start
    proba = pipe.predict_proba(X_val)[:, 1]
    info = {"fit_seconds": round(runtime, 1),
            "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            "n_features_after_encoding": int(pipe.named_steps["prep"].transform(X_val.head(1)).shape[1])}
    return pipe, proba, info


def metrics_at_threshold(y_true, proba, threshold: float = 0.5) -> dict:
    """Threshold-free (AP, ROC-AUC) and threshold-based metrics for one model."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(proba)) > 1 else 0.5,
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 on the given (validation) data; ties go to higher recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = np.flatnonzero(np.isclose(f1, f1.max()))
    return float(thresholds[best[np.argmax(recall[best])]])


def add_sleep_groups(frame: pd.DataFrame, bins=(0, 6, 9, 25),
                     labels=("Less than 6", "6 to 8", "More than 8")) -> pd.DataFrame:
    """Replace SleepHours with Member 2's EDA sleep groups (left-closed bins; missing stays missing)."""
    out = frame.copy()
    groups = pd.cut(out["SleepHours"], bins=list(bins), labels=list(labels), right=False)
    out["SleepGroup"] = groups.astype("object")
    return out.drop(columns="SleepHours")
