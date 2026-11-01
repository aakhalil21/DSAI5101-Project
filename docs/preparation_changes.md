# Preparation workflow changes and handoff

## What changed

Preparation now reads the canonical raw ZIP directly, validates its checksum and the frozen manifest/exclusions, then writes raw-valued handoff files and small audits. It does not split anew, impute, encode, or remove flagged predictor rows. The historical seed-42 duplicate-aware split can be checked with `--verify-seed-42` and reproduced the frozen assignments. All exact-record duplicate groups remain in one split.

The original `notebooks/02_data_preparation.ipynb` is preserved byte for byte, including its source, outputs, metadata, and historical paths. Open it beside `notebooks/02_data_preparation_copy.ipynb` to compare. The corrected copy starts with an explanation, imports maintained functions from `src`, resolves the root from either the root or `notebooks/` directory, uses canonical paths, and runs from a fresh kernel. The old notebook is retained for comparison rather than as the active workflow. The only change to another existing notebook was one outdated manifest-path note in `01_data_understanding.ipynb`; its analysis cells were not changed. The three empty future notebook files are now valid title-and-scope notebooks.

## File moves

| Old location | Current location or disposition |
| --- | --- |
| Root `heart_2022_with_nans.zip` | Removed after SHA-256 matched `data/raw/heart_2022_with_nans.zip` |
| Root `prepare_data.py` | `src/prepare_data.py` |
| `Data Split/split_manifest.csv`, `excluded_rows.csv` | `data/manifests/` (frozen bytes unchanged) |
| `Data Split/Data Split_with_missing_value.zip` | `data/archive/supplied_raw_valued_splits.zip` (local provenance; ignored by Git) |
| `Data Split/Data Split.zip` | `data/archive/legacy_imputed_splits.zip` (local legacy; ignored by Git) |
| `Data Split/split_summary.csv`, `training_missing_summary.csv` | `results/audits/legacy_*.csv` |
| `Data dictionary/data_dictionary.csv`, `.xlsx`, `Missing_Value_Decisions.pdf`, `decision_log.md` | `docs/` |
| `Data dictionary/suggestions_feature_names.csv` | `results/audits/legacy_encoded_feature_names.csv` |
| `Reports/DSAI5101_Proposal.pdf` | `reports/DSAI5101_Proposal.pdf` |

The older imputed archive has 47 columns, including six derived missingness indicators. It is **not** an input for EDA or cross-validation. The supplied raw-valued archive has 41 columns and was retained only after comparison with the canonical outputs. Its CSV bytes differ because the supplied files have a UTF-8 BOM and LF line endings, while regenerated files use the platform CSV format; parsed columns, row order, values, targets, and missing positions match exactly. `data/archive/` and `data/processed/` are local and Git-ignored. Frozen manifest CSVs and small audits remain trackable.

The completed dictionary CSV and the workbook's **Data dictionary** sheet match across all 41 entries. The workbook's **Member 2 guide** sheet was retained. `docs/prepared_dataset_schema.csv` and the new workbook **Prepared dataset schema** sheet define `row_id` and `duplicate_group` as metadata and `heart_disease` as a derived target, separate from the survey source variables. `docs/legacy_features.md` explains the old encoded feature names.

## Regeneration and team use

From the repository root or another working directory:

```powershell
py -m pip install -r requirements.txt
py src/prepare_data.py --verify-seed-42
```

Use an absolute path to the script if running from elsewhere. The command reads `data/raw/heart_2022_with_nans.zip` and `data/manifests/{source_checksums.json,split_manifest.csv,excluded_rows.csv}`, then writes `data/processed/{training.csv,validation.csv,test.csv}` and audit CSVs in `results/audits/`. `--source PATH` accepts an identical verified ZIP or its CSV member. `--output-dir` and `--audit-dir` support isolated regeneration.

Member 2 should open **`data/processed/training.csv`** for EDA and observed-value associations. Its 38 predictors retain raw values and missingness. The target `heart_disease` remains in each saved file for handoff but must be separated as `y`; `row_id`, `duplicate_group`, and the target are excluded from `X`. `HadHeartAttack` and `HadAngina` define the target and are not exported as predictors. Validation is for model decisions; test is for final evaluation. `src/preprocessing.py` returns an unfitted transformer to place inside an estimator pipeline for each cross-validation fold. It keeps BMI and excludes height and weight, including their missingness indicators.

## Offline verification performed

- Original notebook SHA-256 `687773ece7b05d2865b6fe3898989587ad59e30c0891dd7cdaf1049938a45909` and raw ZIP SHA-256 `8bc95f1a36fa96c9105da335df66086041b07da7c9e17d02e522fe09e03ac4a2` remained unchanged. The manifest, exclusions, and supplied raw-valued archive also matched their recorded SHA-256 values in `data/manifests/source_checksums.json` before and after regeneration.
- Frozen assignments have 308,080 training, 66,016 validation, and 66,015 test rows. The 5,021 exclusions exactly match the target rule and recorded reason. No row or duplicate group crosses boundaries. The optional seed-42 check reproduced the manifest.
- For every exported row, all 38 predictor values and missing positions matched the original source row, every target matched the source rule, and all 41 columns and row orders matched the supplied raw-valued splits after CSV parsing.
- Training has 135,348 rows with at least one missing predictor and 33,328 missing BMI values. The SleepHours 1–24 review flag count was 0; no audit flag deletes records.
- Repeated regeneration into a temporary directory produced byte-identical output CSVs and did not alter frozen artifacts.
- The corrected notebook passed format validation and executed from a fresh kernel in `notebooks/` with three code cells and zero errors.
- The preprocessing transformer was unfitted on construction, included BMI, excluded target/identifiers/target sources/height/weight, fit on a training sample, and transformed validation rows without fitting on them.
- `.gitignore` ignores generated datasets and both local archives; `git check-ignore` confirmed the frozen manifest CSVs remain trackable. No commits or pushes were made.

## Remaining limitations

No EDA, model training, associated-factor analysis, or final test evaluation was performed. The existing source dictionary and Member 2 guide were preserved; future scientific interpretation and any alternative preprocessing comparisons remain team work. Local archives and generated datasets are ignored by Git, so teammates must regenerate the handoff files from the tracked canonical source in their own workspace.
