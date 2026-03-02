# Data

Place sensitive datasets in `data/private/`.
Anything in `data/private/` is ignored by git and stays local.
Do not put private data in other folders unless you also add ignore rules.
Supported input formats include `.csv`, `.xlsx`, and `.xls`.

Public safe sample data can be stored in `data/public/`.
`data/public/public_testbench_dataset_20k_minmax.csv` is anonymized:
- per-column min-max normalization to [0, 1]
- first column renamed to `time`; remaining signals use generic letter headers (`B`, `D`, `F`, ...)
- no signal descriptions or original names
- duplicate time columns removed from the public export
- ~5% missing values injected only in `B`, `D`, and `F` for missing-data workflow testing

Additional synthetic public datasets:
- `data/public/public_binary_classification_dataset.csv`
  - balanced binary classification example
  - safe synthetic features (`feature_a`, `feature_b`, `feature_c`) and target `class_label`
  - useful for Agent 1 task detection and stratified steady-state split checks
- `data/public/public_fraud_screening_dataset.csv`
  - imbalanced fraud-like label example
  - safe synthetic transaction-risk style features and target `fraud_flag`
  - useful for Agent 1 fraud detection and future classifier workflow testing
