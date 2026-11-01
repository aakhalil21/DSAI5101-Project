"""Regenerate raw-valued handoff CSVs using the frozen split manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/heart_2022_with_nans.zip"
MANIFEST_DIR = ROOT / "data/manifests"
OUTPUT_DIR = ROOT / "data/processed"
AUDIT_DIR = ROOT / "results/audits"
TARGET_SOURCES = ("HadHeartAttack", "HadAngina")
SPLITS = ("train", "validation", "test")
OUTPUT_NAMES = {"train": "training.csv", "validation": "validation.csv", "test": "test.csv"}
EXPECTED_COUNTS = {"train": 308080, "validation": 66016, "test": 66015}
EXPECTED_EXCLUDED = 5021


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(source_path: Path | str = SOURCE) -> pd.DataFrame:
    """Read a source ZIP or its CSV member after checking frozen provenance."""
    source_path = Path(source_path).resolve()
    checksums = json.loads((MANIFEST_DIR / "source_checksums.json").read_text(encoding="utf-8"))
    if not source_path.is_file():
        raise FileNotFoundError(f"Source dataset is missing: {source_path}")
    key = {".zip": "raw_zip_sha256", ".csv": "raw_csv_member_sha256"}.get(source_path.suffix.lower())
    if key is None or sha256(source_path) != checksums[key]:
        raise ValueError(f"Source checksum or type differs from frozen provenance: {source_path}")
    if key == "raw_zip_sha256":
        with ZipFile(source_path) as archive:
            if archive.namelist() != ["heart_2022_with_nans.csv"]:
                raise ValueError("Unexpected raw ZIP members")
    raw = pd.read_csv(source_path)
    if len(raw) != sum(EXPECTED_COUNTS.values()) + EXPECTED_EXCLUDED or len(raw.columns) != 40:
        raise ValueError("Unexpected source row or column count")
    for column in TARGET_SOURCES:
        if column not in raw:
            raise ValueError(f"Missing target source field: {column}")
        unexpected = set(raw[column].dropna().unique()) - {"Yes", "No"}
        if unexpected:
            raise ValueError(f"Unexpected {column} categories: {sorted(unexpected)}")
    return raw


def construct_records(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach source-order IDs and target without changing predictor values."""
    groups = raw.groupby(raw.columns.tolist(), dropna=False, sort=False, observed=True).ngroup()
    data = raw.copy()
    data.insert(0, "row_id", range(len(data)))
    data.insert(1, "duplicate_group", groups.to_numpy())
    positive = data.HadHeartAttack.eq("Yes") | data.HadAngina.eq("Yes")
    negative = data.HadHeartAttack.eq("No") & data.HadAngina.eq("No")
    target = pd.Series(pd.NA, index=data.index, dtype="Int64")
    target.loc[positive] = 1
    target.loc[negative] = 0
    data["heart_disease"] = target
    excluded = data.loc[target.isna(), ["row_id"]].copy()
    excluded["reason"] = "Unknown heart disease target"
    known = data.loc[target.notna()].drop(columns=list(TARGET_SOURCES)).copy()
    known["heart_disease"] = known.heart_disease.astype(int)
    if not known.groupby("duplicate_group").heart_disease.nunique().eq(1).all():
        raise ValueError("A duplicate group has conflicting target values")
    return known, excluded


def load_frozen_assignments(known: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    """Validate exact frozen rows, groups, assignments, and exclusion reasons."""
    checksums = json.loads((MANIFEST_DIR / "source_checksums.json").read_text(encoding="utf-8"))
    paths = {"split_manifest_sha256": MANIFEST_DIR / "split_manifest.csv",
             "excluded_rows_sha256": MANIFEST_DIR / "excluded_rows.csv"}
    for key, path in paths.items():
        if not path.is_file() or sha256(path) != checksums[key]:
            raise ValueError(f"Frozen artifact missing or changed: {path}")
    manifest = pd.read_csv(paths["split_manifest_sha256"])
    frozen_excluded = pd.read_csv(paths["excluded_rows_sha256"])
    if list(manifest.columns) != ["row_id", "duplicate_group", "split"]:
        raise ValueError("Frozen manifest schema changed")
    if list(frozen_excluded.columns) != ["row_id", "reason"] or not frozen_excluded.equals(excluded.reset_index(drop=True)):
        raise ValueError("Frozen exclusions differ from source target rule")
    if not manifest[["row_id", "duplicate_group"]].equals(known[["row_id", "duplicate_group"]].reset_index(drop=True)):
        raise ValueError("Frozen IDs or duplicate groups differ from source")
    if not manifest.row_id.is_unique or manifest.split.isna().any() or set(manifest.split) != set(SPLITS):
        raise ValueError("Frozen manifest has invalid assignments")
    if not manifest.groupby("duplicate_group").split.nunique().eq(1).all():
        raise ValueError("A duplicate group crosses split boundaries")
    if manifest.split.value_counts().to_dict() != EXPECTED_COUNTS or len(excluded) != EXPECTED_EXCLUDED:
        raise ValueError("Frozen split or exclusion counts changed")
    return manifest


def verify_seed_42(known: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Check historical grouping algorithm without writing assignments."""
    groups = known[["duplicate_group", "heart_disease"]].drop_duplicates("duplicate_group")
    train, remainder = train_test_split(groups, test_size=0.30, random_state=42, stratify=groups.heart_disease)
    validation, test = train_test_split(remainder, test_size=0.50, random_state=42, stratify=remainder.heart_disease)
    lookup = pd.concat([train.assign(split="train"), validation.assign(split="validation"),
                        test.assign(split="test")]).set_index("duplicate_group").split
    if not manifest.split.equals(manifest.duplicate_group.map(lookup).rename("split")):
        raise ValueError("Seed-42 algorithm does not reproduce frozen assignments")


def write_audits(outputs: dict[str, pd.DataFrame], audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{"split": split, "rows": len(outputs[split]),
                             "positive_cases": int(outputs[split].heart_disease.sum()),
                             "negative_cases": int(len(outputs[split]) - outputs[split].heart_disease.sum())}
                            for split in SPLITS])
    summary.to_csv(audit_dir / "split_summary.csv", index=False)
    predictors = outputs["train"].drop(columns=["row_id", "duplicate_group", "heart_disease"])
    missing = pd.DataFrame({"column": predictors.columns, "missing_count": predictors.isna().sum().to_numpy()})
    missing["missing_pct"] = 100 * missing.missing_count / len(predictors)
    missing.to_csv(audit_dir / "training_missing_summary.csv", index=False)
    bounds = {"PhysicalHealthDays": (0, 30), "MentalHealthDays": (0, 30), "SleepHours": (1, 24),
              "HeightInMeters": (0, None), "WeightInKilograms": (0, None), "BMI": (0, None)}
    flags = []
    for column, (low, high) in bounds.items():
        values = predictors[column]
        flagged = values.notna() & ((values < low) | ((values > high) if high is not None else False))
        flags.append({"column": column, "review_flag_count": int(flagged.sum()),
                      "lower_bound": low, "upper_bound": high})
    pd.DataFrame(flags).to_csv(audit_dir / "training_range_flags.csv", index=False)


def prepare_data(source_path: Path | str = SOURCE, output_dir: Path | str = OUTPUT_DIR,
                 audit_dir: Path | str = AUDIT_DIR, check_seed_42: bool = False) -> dict[str, pd.DataFrame]:
    """Regenerate handoff files and audits; never write frozen artifacts."""
    known, excluded = construct_records(load_source(source_path))
    manifest = load_frozen_assignments(known, excluded)
    if check_seed_42:
        verify_seed_42(known, manifest)
        print("Seed-42 duplicate-group split matches the frozen manifest.")
    known = known.copy()
    known["split"] = manifest.split.to_numpy()
    output_dir, audit_dir = Path(output_dir), Path(audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in SPLITS:
        frame = known.loc[known.split.eq(split)].drop(columns="split")
        if len(frame) != EXPECTED_COUNTS[split]:
            raise ValueError(f"Unexpected {split} count")
        path = output_dir / OUTPUT_NAMES[split]
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        outputs[split] = frame
        print(f"{split}: {len(frame):,} raw-valued rows -> {path}")
    write_audits(outputs, audit_dir)
    print(f"Unknown-target exclusions: {len(excluded):,}; frozen artifacts unchanged.")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Verified raw ZIP or its CSV member")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--verify-seed-42", action="store_true")
    args = parser.parse_args()
    prepare_data(args.source, args.output_dir, args.audit_dir, args.verify_seed_42)


if __name__ == "__main__":
    main()
