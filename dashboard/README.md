# Results dashboard

`index.html` is a single, self-contained page with five tabs: Overview, Leading predictors, Young adults, Stories and Models.

## How to open it

Double-click `index.html`. It opens in your browser. No install or server is needed. Fonts load from Google Fonts. When you are offline, the page falls back to system fonts and still works.

## How to edit it

Every number on the page is in the `DATA` block near the top of the `<script>`, marked `EDIT CONTENT HERE`. Change a number there and it updates everywhere on the page. Do not edit numbers anywhere else.

Where each part of `DATA` comes from:

| `DATA` field | Source |
| --- | --- |
| `tiles` (440,111 rows) | Training + validation row counts, plus the test count in the main README (308,080 + 66,016 + 66,015) |
| `tiles` (9.03%), `ageRates` | Training split (`data/processed/observed/training.csv`), heart disease rate overall and by age band |
| `tiles` (34 predictors) | 38 predictors minus ChestScan, PneumoVaxEver, height and weight (`results/modelling/frozen_settings.json`) |
| `tiles` (AP 0.361, baseline 0.090), `models`, `metrics`, `totalCases` | `results/modelling/validation_metrics_three_models.csv` |
| `thresholds` | `results/modelling/validation_metrics.csv` (default 0.5 and chosen threshold rows for logistic regression) |
| `importance` | `results/modelling/predictor_importance.csv`, logistic regression, seed 42 |
| `agreement` | `results/modelling/predictor_importance.csv`, rank 1–5 for each model |
| `oddsRatios` | `results/modelling/odds_ratios.csv` |
| `factors` (stroke, kidney disease, teeth, sleep, smoking) | `results/stories/age_interactions.csv`, Model A, training split (`rate_with`, `rate_without`) |
| `factors` (poor general health, COPD, depressive disorder) and all "People (18–39)" counts | Not in a results file yet. Recomputed from the training split with the same rules as `notebooks/06_age_interactions.ipynb` |
| `stories` | `results/stories/age_interactions.csv`, Model B, `odds_ratio` for 18-39 and 65+ (training and validation) |

## Placeholders still to fill

- **Q2 associations** (Leading predictors tab): will come from `results/q2/q2_associations.csv`.
- **Story deep dives** (Stories tab): whether each story holds within smokers and non-smokers, by sex, and by diabetes status.
- **Final test results** (Models tab): test-split metrics, once all choices are frozen.

## Note

The numbers are training and validation results. The test split is not shown yet. All findings are associations from a single survey. They do not show cause.
