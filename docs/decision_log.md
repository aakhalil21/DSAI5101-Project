# Data Preparation Decision Log

## Decisions implemented

1. Preserve the original CSV unchanged.
   Reason: retain a reproducible source dataset.

2. Create a stable row_id from the original CSV row order.
   Reason: track records and share split assignments.
   The row_id is not a model predictor.

3. Define heart_disease as 1 when HadHeartAttack or HadAngina
   is Yes, and 0 only when both are No.
   Reason: follow the project target definition.

4. Exclude 5,021 records with unknown targets.
   Reason: their outcome cannot be determined.
   Do not impute target labels.

5. Exclude HadHeartAttack and HadAngina from predictors and Q2.
   Reason: both fields define the target.

6. Retain identical records initially.
   Reason: identical survey answers do not prove duplicate people.
   Keep each exact-record group within one split.
   Group identifiers are not model predictors.

7. Split record groups using stratification and seed 42.
   Assign approximately 70% training, 15% validation and 15% test.
   Frozen row assignments are in `data/manifests/split_manifest.csv`.
   Normal regeneration validates and reads this file; it never rewrites it.

8. Retain rows with missing predictor values.
   Reason: 135,348 training rows have at least one missing predictor.
   Complete-case deletion would discard about 43.9% of training rows.

9. Keep the handoff CSVs raw valued with missingness intact.
   `src/preprocessing.py` provides an unfitted training-median numeric
   transformer with missingness indicators and a categorical Missing
   category with one-hot encoding. Fit it inside each model CV fold.

10. Keep BMI and exclude height and weight from the primary model.
    Reason: reduce overlap between BMI and its component measurements.
    Retain original measurements for EDA and sensitivity analysis.

11. Do not automatically remove extreme numeric values.
    The implemented training range checks are review flags, including
    SleepHours outside the dictionary's 1-24 range.
    These checks do not establish that every extreme value is valid.

## Remaining work

- Revisit variable definitions, units, or special responses if a new source issue is found; the current dictionary is completed.
- Investigate training missingness by age, sex and state.
- Inspect fields that tend to be missing together.
- Compare records with known and unknown targets for exclusion context.
- Review extreme body measurements against source documentation.
- Coordinate numeric scaling for logistic regression with Member 3.
- Fit preprocessing within each cross-validation training fold when models are built.
- Compare alternative preprocessing on validation data with Member 3.
- Reserve test evaluation until model decisions are frozen.
