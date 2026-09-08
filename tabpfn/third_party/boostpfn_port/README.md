# BoostPFN port (what `nfv3_v3_exp35_boostpfn.py` runs)

`tabpfn/scripts/nfv3_v3_exp35_boostpfn.py` does **not** contain the BoostPFN algorithm.
At runtime (`import_boostpfn()`, ~line 199) it puts `--boostpfn-root` (default
`tabpfn/third_party/BoostPFN`, vendored 2026-09-08 -- previously `~/Desktop/SOTA/BoostPFN`) on
`sys.path` and imports the upstream classes unchanged:

| symbol used by exp35 | upstream file | role |
|---|---|---|
| `gradient_boost_tabpfn.SamplingGradientboost` | `gradient_boost_tabpfn.py` | the boosting loop; `get_weak_learner()` (line 102) draws the 500-row weighted sample, fits TabPFN-v1 in-context, predicts train+test |
| `boost_tabpfn.SamplingAdaboost` | `boost_tabpfn.py` | AdaBoost variant |
| `utils.splitting_predict_proba` | `utils.py` | batched predict |
| `scripts.transformer_prediction_interface.TabPFNClassifier` | `scripts/transformer_prediction_interface.py` | the TabPFN-v1 weak learner (upstream's copy of the v1 interface) |

exp35 only (a) builds the IDS train/test matrices with the project's loaders, (b) calls
`SamplingGradientboost(TabPFNClassifier(device, N_ensemble_configurations=1), T, sampling_size,
args, split_test=True, max_samples=500, batch_test=50000, version=2).fit(X, y, X_test)` and
`.predict_proba(X_test)` exactly as `largedataset_boostpfn.py:250-258` does, and (c) scores the
result with the project's per-class table.  Static analysers flag the imports inside
`import_boostpfn()` as unresolved because the clone is outside this repo; at runtime
`inspect.getsourcefile(SamplingGradientboost)` is `.../SOTA/BoostPFN/gradient_boost_tabpfn.py`.

The vendored source in `tabpfn/third_party/BoostPFN/` is tracked in git (its `.git` was detached;
provenance in `UPSTREAM.txt`); only the 103 MB checkpoint is not -- fetch it with
`bash tabpfn/third_party/fetch_checkpoints.sh`.  Because the copy carries local compatibility
patches (upstream targets TabPFN v1 / sklearn 0.24 / torch 1.9 / python 3.7), the port is ALSO
recorded here as patches so it can be rebuilt from upstream anywhere:

```
bash tabpfn/third_party/boostpfn_port/setup_boostpfn_port.sh            # -> tabpfn/third_party/BoostPFN (refuses if present)
bash tabpfn/third_party/boostpfn_port/setup_boostpfn_port.sh /some/dir   # -> custom root; pass --boostpfn-root to exp35
```

Files: `boostpfn_tracked.patch` (diff of upstream-tracked files vs commit c957ac2),
`tabpfn_v1_vendored.patch` (diff of the vendored TabPFN v1 package vs the pristine `tabpfn==0.1.9`
wheel), `gb_losses_compat.py`, `IMBALCIC_PORT_NOTES.md` (what/why of every patch), `checkpoint.sha256`.

Validation of the port against upstream (2026-09-08, see `lablog/report/0904.md` §8): the
unmodified upstream driver `largedataset_boostpfn.py`, run with the README command on its own
OpenML dataset BNG(glass,nominal,137781) at T=1000, gives AUC 0.9011 vs 0.900 in the paper's Table 13.
