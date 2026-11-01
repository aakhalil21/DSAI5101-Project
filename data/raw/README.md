# Original dataset

Source: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

File: heart_2022_with_nans.zip
Contains: heart_2022_with_nans.csv
Size: 445,132 rows and 40 columns.

This is the original 2022 dataset with missing values retained. Keep it unchanged. `src/prepare_data.py` reads the ZIP directly and verifies its SHA-256 using `data/manifests/source_checksums.json`; extraction is optional. Original row order defines `row_id`. The redundant root ZIP was removed after its checksum matched this copy.
