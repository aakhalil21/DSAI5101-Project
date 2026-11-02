"""Unfitted primary-model preprocessing; place inside an estimator Pipeline for CV."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

EXCLUDED = {"heart_disease", "HadHeartAttack", "HadAngina", "row_id",
            "duplicate_group", "HeightInMeters", "WeightInKilograms"}
NUMERIC_IMPUTATION_STRATEGY = "median"
CATEGORICAL_FILL_VALUE = "Missing"


def impute_split_copies(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Fill predictor copies using training-only statistics, retaining IDs and labels."""
    training = splits["train"]
    metadata = ["row_id", "duplicate_group", "heart_disease"]
    features = training.drop(columns=metadata)
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.columns.difference(numeric, sort=False).tolist()
    numeric_imputer = SimpleImputer(strategy=NUMERIC_IMPUTATION_STRATEGY)
    categorical_imputer = SimpleImputer(strategy="constant", fill_value=CATEGORICAL_FILL_VALUE,
                                        keep_empty_features=True)
    if numeric:
        numeric_imputer.fit(features[numeric])
    if categorical:
        categorical_imputer.fit(features[categorical])
    copies = {}
    for split, frame in splits.items():
        filled = frame.copy()
        if numeric:
            filled[numeric] = numeric_imputer.transform(frame[numeric])
        if categorical:
            filled[categorical] = categorical_imputer.transform(frame[categorical])
        if filled.isna().any().any():
            raise ValueError(f"Imputed {split} copy still has missing values")
        if not filled[metadata].equals(frame[metadata]):
            raise ValueError(f"Imputation changed {split} IDs or targets")
        copies[split] = filled
    return copies


def make_preprocessor(training_columns: pd.DataFrame) -> ColumnTransformer:
    """Return a fresh, unfitted transformer based on training-frame dtypes.

    Pass a training frame to select columns, then use this transformer inside an
    estimator Pipeline for cross-validation. Never fit it on validation or test.
    BMI stays in the primary model; height and weight, including their missingness
    indicators, are excluded.
    """
    if "BMI" not in training_columns:
        raise ValueError("Primary-model input must include BMI")
    features = training_columns.drop(columns=list(EXCLUDED & set(training_columns.columns)))
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.columns.difference(numeric, sort=False).tolist()
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy=NUMERIC_IMPUTATION_STRATEGY, add_indicator=True), numeric),
            ("categorical", Pipeline([
                ("missing", SimpleImputer(strategy="constant", fill_value=CATEGORICAL_FILL_VALUE,
                                           keep_empty_features=True)),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
    )
