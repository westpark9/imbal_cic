# CLAUDE.md

Working manual for `imbalcic/`. Read before touching code. Last rewritten 2026-08-14.

This folder is a lean copy of `imbal_cic/` holding only the **currently active** work. Everything
older lives in the original `imbal_cic/` (unchanged), and the 44 archived experiment scripts are
summarized in **`../SRC_HISTORY.md`** — read that before proposing any method change.

## What this project is

Open-set / OOD-aware classification under severe class imbalance (IR > 1000 for tail attack
classes), for network intrusion detection on NetFlow-v3 datasets. Paper draft: `docs/main.tex`.

Two active tracks:

| Track | Files | What it is |
|---|---|---|
| **NF-v3 expert pipeline** (paper) | `scripts/s43_*.py`, `scripts/s44_*.py` | independent per-expert Energy gates + a resolution step |
| **TabPFN check** (standalone) | `tabpfn/nfv3_multiclass_test_v2.py` | TabPFN-v3 vs XGBoost sanity check; *not* part of the paper's experiment record |

### Current frontier state — both tracks are losing

**s44** (28-config `--unseen` sweep, 2026-08-04, `results/scripts_analysis/s44_unseen_tradeoff_*.csv`):
the expert pipeline beats the Global XGBoost baseline on known macro-F1 in only **3 of 28**
configurations; median delta **−0.184**. Unseen resolved accuracy median **0.125**. The XGBoost
baseline is the bar, and it is not being cleared.

**TabPFN** (`manuscript/report/0812.md`): no configuration where TabPFN-v3 clearly beats XGBoost;
cic2018-chronological is the only partial win. Cost gap is 50–331×
(2026-08-13: 24,649 s vs 74 s). Memory analysis: `../docs/tabpfn_v2_memory_rootcause.html`.

Any proposed change should say which of these two numbers it attacks.

## Running

```bash
pip install -r requirements.txt      # xgboost>=2.0, torch==2.6.0+cu124, sklearn, imbalanced-learn
pip install tabpfn                   # or use the local editable install — see Data section below
```

**Always run from this folder's root** — `scripts/s4x` scripts resolve `scripts/` via `REPO_ROOT` and
data paths CWD-relative.

```bash
# s44: resolved expert pipeline (the paper track)
python scripts/s44_nfv3_resolved_expert_pipeline.py --target cse_cic_ids2018 --unseen bot
python scripts/s43_nfv3_independent_expert_energy.py --target ton_iot     # gates only, no resolution

# sweep summary over existing s44 runs
python scripts/s44_unseen_tradeoff_summary.py

# TabPFN track (defaults resolve to this folder's data/ and the ckpt symlink)
python tabpfn/nfv3_multiclass_test_v2.py --target-dataset cic2018_capped
```

`tabpfn/tabpfn-v3-...ckpt` is a **symlink** into `../tabpfn/`. The `tabpfn` package itself is a
pip editable install pointing at `../tabpfn/src/tabpfn`, so `import tabpfn` resolves regardless
of CWD — there is no local `tabpfn_src` copy in this folder.

## Data

`data/` holds real copies (17 GB, gitignored) of the five pkls the current code needs:

| File | Used by | Notes |
|---|---|---|
| `nfv3_energy_suite_uncapped_scenarios.pkl` | s43, s44, tabpfn | 14.7 GB, 66.9M rows × 46 feat; 4 datasets in one schema |
| `nfv3_energy_suite_cic2018_scenarios.pkl` | tabpfn `*_capped` targets | per-family-capped slice of all four |
| `nfv3_energy_suite.pkl` | `preprocess_nfv3_cic2018_scenarios.py` (`--base-suite`) | |
| `cic2017_full_raw.pkl` | tabpfn `cic2017_full` | 15 classes, synthetic `time_proxy` |
| `cic2017_chrono_v2.pkl` | `analyze_nfv3_embedding.py` | legacy CIC open-set split |

The suite pkls key on `dataset_names ∈ {cse_cic_ids2018, bot_iot, ton_iot, unsw_nb15}` plus
`families`, `attack_scenarios`, `timestamps`. Only cic2018 has real per-scenario chronological
splitting; the others split at family level.

**Splits are chronological, never random.** Random stratified splits on CIC data leak temporally —
this project already paid for that lesson (`s05`/`s06` in SRC_HISTORY). If validation looks
near-perfect, suspect leakage, not success.

## Conventions (follow exactly)

- **One script = one frozen experiment.** New idea → copy the nearest predecessor to a new file
  with a new suffix. Never retrofit a script that already has runs in `results/` — those are the
  reproducibility record.
- **`scripts/exp_utils.py` is the shared library** for s43/s44 and the tabpfn scripts. Changing a
  helper changes past experiments' code paths — add a new helper alongside, don't edit in place.
- **`configs/nfv3_experts.json` is required** by s43/s44 (`--expert-config`): it defines the
  per-dataset expert → family grouping.
- **Results dirs are created by the script**, never by hand:
  `results/<ts>_<pid>_<dataset>_<tag>/`, with the tag hardcoded per script.
- **First log line is the full `Args: {...}` dump.** That is the config record — there are no
  config files. Every hyperparameter must be an argparse arg so it lands there.
- **Artifacts**: numbered CSV + PNG pairs (`0a…5c`, plus s44's `6a/6b`), per-class comparison CSVs
  with macro/weighted footers. No checkpoints — `s43`/`s44` save no weights; reproducing a
  number means re-running. (The tabpfn script is the exception: it saves fitted models and
  resume checkpoints.)
- **`--seed 42` everywhere**, threaded into every RNG; sub-models get deterministic offsets.
  Single-seed by design — the workflow is one-knob-at-a-time structural ablation.
- **One knob per run.** Consecutive runs differ by one structural/threshold change so causality
  stays readable in the run sequence. If a change needs two knobs, say so in the log.
- **Energy convention**: `E = -T * logsumexp(logits / T)`, thresholded at an ID-validation
  quantile, "accept if score ≥ threshold". ⚠ `--energy_margin_in/out` sign conventions differ
  between older scripts — read the loss line before transplanting a value.
- **Lab log per work session** in `manuscript/report/MMDD.md` (Korean/English mix). Findings go in
  the run dir and the lab log, not only in chat.

## What counts as a result

A result is the **tuple**: known macro-F1 including tail classes, OOD/unseen detection, and
benign/ID retention — versus the baseline computed **in the same run, same split**. ID accuracy
alone is meaningless here; so is an OOD number without its ID cost. Improving one leg while
silently dropping another is a regression.

Before citing any number, read the run's `Args:` line. `results/` contains subsampled and
smoke runs that look identical to real ones.

## Before proposing a method change

Read `../SRC_HISTORY.md` first and state which recorded failure mode your idea is nearest to and
why it escapes it. The short list of what is already buried:

1. Routing error compounds imbalance — routers misclassify exactly the tail classes the MoE exists to save.
2. Anything fitted to validation predictions (confusion-matrix experts, competence models, val-tuned
   thresholds) looks perfect on val and does not transfer under chrono splits.
3. XGBoost's boosting is already an implicit MoE; explicit routing on top adds error, not specialization.
4. TTA/perturbation stability cannot identify the owning expert (proven by `s23_code_tta_diag.py`) —
   non-owning experts are just as confident and stable as owners.
5. Global test-time weight learning is head-dominated.
6. Energy gates degrade toward chance under long-tail training.

A "novel suggestion" here is usually one of these in disguise.
