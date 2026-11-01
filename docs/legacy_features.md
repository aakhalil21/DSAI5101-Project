# Legacy encoded feature names

`results/audits/legacy_encoded_feature_names.csv` lists encoded/model feature names from an earlier preparation attempt. Entries such as `numeric__missingindicator_BMI` are derived legacy features, not BRFSS source variables and not columns in the current 41-column handoff. The file is retained for provenance only. Current preprocessing feature names are determined by fitting the transformer within a training fold.
