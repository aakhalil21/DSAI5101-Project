"""Unfitted primary-model preprocessing; place inside an estimator Pipeline for CV."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

EXCLUDED = {"heart_disease", "HadHeartAttack", "HadAngina", "row_id",
            "duplicate_group", "HeightInMeters", "WeightInKilograms"}


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
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            ("categorical", Pipeline([
                ("missing", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
    )
