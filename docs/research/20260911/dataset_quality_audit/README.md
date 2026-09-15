# NF-v3 suite data-quality audit (2026-09-11, read-only)

Question: before moving experiments off cic2018, do bot_iot / ton_iot / unsw_nb15 carry the same
pathologies (byte-identical feature vectors with conflicting labels, single-vector scenarios,
memorized chronological test rows)? Source: `data/nfv3_energy_suite_uncapped_scenarios.pkl`
(66,935,021 x 46). Row hash = `pd.util.hash_pandas_object` over the nan_to_num float32 row.

Files
- `audit_summary.csv` per dataset: rows, unique vectors, duplicate rate, fraction of rows in
  mixed-label hash groups, hash-majority macro-F1 ceiling, chronological-test (per-family
  60/20/20 by time) memorization and conflict rates.
- `audit_classes.csv` per class: twin fractions (any other label / benign), hash-majority F1,
  time quantiles, chrono-test memorization/conflict.
- `audit_scenarios.csv` per scenario: unique vectors, top-vector share, rows sharing vectors
  with other scenarios.
- `audit_ceilings.csv` per class: clean-vector share, majority-assignment and rarest-class-
  assignment F1 ceilings (all rows and chrono-test rows).
- `unsw_xgb_plain.csv`, `unsw_xgb_sqrt_balanced.csv`: XGBoost (300 trees, depth 8, lr 0.05,
  seed 42) on unsw_nb15 per-family chronological 60/20/20; F1 on clean vectors only; ceilings.
- `unsw_probe.log`: TTL-only benign/attack shortcut check and XGB confusion matrices with and
  without MIN_TTL/MAX_TTL.
Scripts are the exact code that produced the CSVs; they write nothing under the repo.
- `report_data_twins.py` → `report_twins.json`: per dataset the top-10 conflicting hash groups
  (labels, scenarios, time spans, the full 46-feature vector) and per-class clean / benign-twin /
  other-twin decomposition; `report_data_time.py` → `report_time.json`: class share over 60
  time-ordered bins, per-class time spans with the per-family 60/20/20 boundaries, and the
  hash-lookup classifier under random 80/20 vs chronological split.
- `build_dataset_report.py`: renders `lablog/html_report/dataset_quality_0911.html` from the above.

