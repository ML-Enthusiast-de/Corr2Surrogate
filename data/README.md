# Data

Place sensitive datasets in `data/private/`.
Anything in `data/private/` is ignored by git and stays local.
Do not put private data in other folders unless you also add ignore rules.
Supported input formats include `.csv`, `.xlsx`, and `.xls`.

Public safe sample data can be stored in `data/public/`.
`data/public/public_testbench_dataset_20k_minmax.csv` is anonymized:
- per-column min-max normalization to [0, 1]
- generic letter headers (`A`, `B`, ..., `AA`, ...)
- no signal descriptions or original names
