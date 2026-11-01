from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(csv_path, seed=42):
    """Create known-target data and duplicate-aware split assignments."""
    raw = pd.read_csv(csv_path)

    for column in ["HadHeartAttack", "HadAngina"]:
        unexpected = set(raw[column].dropna().unique()) - {"Yes", "No"}
        if unexpected:
            raise ValueError(
                f"Unexpected values in {column}: {unexpected}"
            )

    group_ids = raw.groupby(
        raw.columns.tolist(),
        dropna=False,
        sort=False,
        observed=True
    ).ngroup()

    data = raw.copy()
    data.insert(0, "row_id", range(len(data)))
    data.insert(1, "duplicate_group", group_ids.to_numpy())

    positive = (
        data["HadHeartAttack"].eq("Yes")
        | data["HadAngina"].eq("Yes")
    )
    negative = (
        data["HadHeartAttack"].eq("No")
        & data["HadAngina"].eq("No")
    )

    data["heart_disease"] = pd.Series(
        pd.NA, index=data.index, dtype="Int64"
    )
    data.loc[positive, "heart_disease"] = 1
    data.loc[negative, "heart_disease"] = 0

    excluded = data.loc[
        data["heart_disease"].isna(), ["row_id"]
    ].copy()
    excluded["reason"] = "Unknown heart disease target"

    known = data.loc[data["heart_disease"].notna()].copy()
    known["heart_disease"] = known["heart_disease"].astype(int)

    assert known.groupby(
        "duplicate_group"
    )["heart_disease"].nunique().eq(1).all()

    groups = known[
        ["duplicate_group", "heart_disease"]
    ].drop_duplicates("duplicate_group")

    train_groups, remaining_groups = train_test_split(
        groups,
        test_size=0.30,
        random_state=seed,
        stratify=groups["heart_disease"]
    )

    validation_groups, test_groups = train_test_split(
        remaining_groups,
        test_size=0.50,
        random_state=seed,
        stratify=remaining_groups["heart_disease"]
    )

    assignments = pd.concat([
        train_groups.assign(split="train"),
        validation_groups.assign(split="validation"),
        test_groups.assign(split="test")
    ]).set_index("duplicate_group")["split"]

    manifest = known[["row_id", "duplicate_group"]].copy()
    manifest["split"] = manifest["duplicate_group"].map(assignments)

    assert manifest["row_id"].is_unique
    assert manifest["split"].notna().all()
    assert manifest.groupby(
        "duplicate_group"
    )["split"].nunique().eq(1).all()

    X = known.drop(columns=[
        "row_id", "duplicate_group", "heart_disease",
        "HadHeartAttack", "HadAngina"
    ])
    y = known["heart_disease"].copy()

    return X, y, manifest, excluded


def save_assignments(manifest, excluded, output_dir):
    """Save row assignments and exclusion reasons."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    excluded.to_csv(output_dir / "excluded_rows.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    X, y, manifest, excluded = prepare_data(args.csv_path)
    save_assignments(manifest, excluded, args.output_dir)

    print("Retained rows:", len(X))
    print("Excluded rows:", len(excluded))
    print("Preparation completed successfully.")
