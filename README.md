# DSAI5101 Project

Group study of predictors of self-reported heart disease and associated demographic, health, and lifestyle factors. The research question and frozen data splits are unchanged.

## Setup and preparation

Use Python 3.12 and install the versions used in local verification:

```powershell
py -m pip install -r requirements.txt
py src/prepare_data.py --verify-seed-42
```

The command works from the repository root or another directory. It reads `data/raw/heart_2022_with_nans.zip` directly, checks `data/manifests/source_checksums.json`, validates the frozen `split_manifest.csv` and `excluded_rows.csv`, and writes `data/processed/training.csv`, `validation.csv`, and `test.csv`. To use an explicitly supplied copy of the verified source, pass `--source PATH`. The optional `--verify-seed-42` checks the original grouping algorithm against the manifest; it never creates a new split. `--output-dir` and `--audit-dir` allow temporary verification outputs. The command fails if provenance, categories, assignments, counts, or duplicate groups differ. Generated datasets are kept locally and ignored by Git.

Expected rows: training **308,080**, validation **66,016**, test **66,015**, unknown-target exclusions **5,021**. The 41-column handoff contains 38 unmodified predictors, `heart_disease`, `row_id`, and `duplicate_group`. A target is 1 when `HadHeartAttack` or `HadAngina` is Yes; it is 0 only when both are No. Other combinations are excluded with reasons in `data/manifests/excluded_rows.csv`. The two target-source fields are absent from predictor exports. Missing predictor values are preserved.

## Team handoff

Member 2 should open **`data/processed/training.csv`** for EDA and observed-value associations. Keep missing values visible and report denominators. Use validation only for model decisions and test only for final evaluation. `heart_disease` stays in each saved dataset, but separate it as `y`; do not include it in `X`. `row_id` and `duplicate_group` are metadata, never predictors.

For a primary model, call `src.preprocessing.make_preprocessor(training_frame)` to get a fresh, unfitted transformer, and place it inside the estimator `Pipeline` during cross-validation. It keeps BMI, excludes height and weight, and fits imputation and encoding within each training fold. The raw handoff CSVs are the inputs for EDA and cross-validation. `data/archive/legacy_imputed_splits.zip` is a local historical artifact, not an input for either task. `data/archive/supplied_raw_valued_splits.zip` is preserved locally for provenance.

The maintained notebook is `notebooks/02_data_preparation_copy.ipynb`. The original `notebooks/02_data_preparation.ipynb` is preserved byte for byte for comparison; its old path references are historical and should not be run as the active workflow. See [preparation changes](docs/preparation_changes.md), [decision log](docs/decision_log.md), and [data dictionary](docs/data_dictionary.csv). The proposal remains in `reports/`.
