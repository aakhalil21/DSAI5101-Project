# DSAI5101 Project

Group study of predictors of self-reported heart disease and associated demographic, health, and lifestyle factors. The research question and frozen data splits are unchanged.

## Setup and preparation

Use Python 3.12 and install the versions used in local verification:

```powershell
py -m pip install -r requirements.txt
py src/prepare_data.py --verify-seed-42
```

To view the results dashboard, open `dashboard/index.html` in a browser.

The command works from the repository root or another directory. It reads `data/raw/heart_2022_with_nans.zip` directly, checks `data/manifests/source_checksums.json`, validates the frozen `split_manifest.csv` and `excluded_rows.csv`, and writes `training.csv`, `validation.csv`, and `test.csv` under `data/processed/observed/`. To use an explicitly supplied copy of the verified source, pass `--source PATH`. The optional `--verify-seed-42` checks the original grouping algorithm against the manifest; it never creates a new split. `--output-dir` and `--audit-dir` allow temporary verification outputs. The command fails if provenance, categories, assignments, counts, or duplicate groups differ. Manifest files use LF endings across Git checkouts, and byte-level SHA-256 checks remain enabled. Generated CSVs are ignored by Git.

Expected rows: training **308,080**, validation **66,016**, test **66,015**, unknown-target exclusions **5,021**. The 41-column handoff contains 38 unmodified predictors, `heart_disease`, `row_id`, and `duplicate_group`. A target is 1 when `HadHeartAttack` or `HadAngina` is Yes; it is 0 only when both are No. Other combinations are excluded with reasons in `data/manifests/excluded_rows.csv`. The two target-source fields are absent from predictor exports. Missing predictor values are preserved.

## Data layout and team handoff

| Directory | Purpose |
| --- | --- |
| `data/raw/` | Unchanged source ZIP |
| `data/manifests/` | Frozen row assignments, exclusions, and checksums |
| `data/processed/observed/` | Authoritative splits with original predictor values and missingness |
| `data/processed/imputed/` | Optional filled copies of the same splits |
| `data/archive/` | Historical files only |
| `results/tables/` | EDA summary tables written by `notebooks/03_eda.ipynb` |
| `results/figures/` | EDA figures written by `notebooks/03_eda.ipynb` (regenerated copies of figures 10 and 13 in `results/figures/_regenerated/`) |
| `results/output_manifest.csv` | Every saved table and figure, the notebook cell that writes it, and whether it is reproduced and cited in the EDA report |
| `Reports/` | Project proposal and the Member 2 EDA summary report |
| `dashboard/` | Interactive results dashboard (single HTML file; see `dashboard/README.md`) |

Member 2 should open **`data/processed/observed/training.csv`** for EDA and observed-value associations. Member 3 should use the observed splits for modeling, fitting preprocessing within each training fold during cross-validation. Keep missing values visible and report denominators. Use validation only for model decisions and test only for final evaluation. `heart_disease` stays in each saved dataset, but separate it as `y`; do not include it in `X`. `row_id` and `duplicate_group` are metadata, never predictors.

For a primary model, call `src.preprocessing.make_preprocessor(training_frame)` to get a fresh, unfitted transformer, and place it inside the estimator `Pipeline` during cross-validation. It keeps BMI, excludes height and weight, and fits imputation and encoding within each training fold. If filled copies are needed for a separate downstream use, run `py src/prepare_data.py --write-imputed`. This fits numeric medians and the categorical `Missing` rule on the entire observed training split, then applies those fitted rules to validation and test. It fills predictors only and preserves IDs and target labels. **Do not use `data/processed/imputed/` for observed-data associations or cross-validation**; whole-training imputation would leak information across CV folds. `data/archive/legacy_imputed_splits.zip` and `data/archive/supplied_raw_valued_splits.zip` are historical artifacts.

The maintained notebook is `notebooks/02_data_preparation.ipynb`. The original is preserved byte for byte as `notebooks/archive/02_data_preparation_original.ipynb` for comparison; its old path references are historical and should not be run as the active workflow. See [decision log](docs/decision_log.md) and [data dictionary](docs/data_dictionary.csv). The proposal and the EDA summary report are in `Reports/`.

## EDA outputs

`notebooks/03_eda.ipynb` uses paths relative to `notebooks/`; `py -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb` runs it there from the repository root. It reads only the observed training split and writes tables to `results/tables/` and figures to `results/figures/`. `results/output_manifest.csv` lists each output with its generating cell.
