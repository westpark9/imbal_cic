# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **network intrusion detection** with severely imbalanced multi-class classification (IR > 1000+ for tail attack classes). The goal is to beat a single XGBoost baseline using a Mixture-of-Experts (MoE) ensemble, with special focus on improving tail-class recall without sacrificing majority-class precision.

Supported datasets: CIC-IDS2017, CIC-IDS2018, UNSW-NB15, NF-UNSW-NB15.

## Development History and What Was Tried

Scripts were developed in this order, each addressing the failures of the previous:

| File | Approach | Outcome |
|---|---|---|
| `cic_6.py` | TAILGUARD: taxonomy-aware MoE, router routes samples to attack-family experts | No gain over baseline |
| `cic_xg_19.py` | XGBoost router with soft probabilities, SMOTE per expert | No gain over baseline |
| `code_final.py` | Dropped explicit routing; uses OOD-filtered competence-weighted ensemble | Still not better |
| `code_final_nf.py` | Same as code_final but on NF-UNSW-NB15 for cross-dataset comparison | — |
| `code_chrono.py` | Chronological split to prevent data leakage from random splits | Performance degraded |
| `code_file.py` | Per-file split (same motivation as chrono) | Performance degraded |
| `code_mat.py` | FAR-MoE: confusion-matrix-derived expert design + OOD-gated routing | Val near-perfect, test not improved |

**Known failure modes (do not repeat these):**

1. **Routing error compounds imbalance**: The router misclassifies tail classes (rare → hard to learn), so those samples get sent to the wrong expert and perform worse than the baseline.
2. **Expert composition imbalance**: Assigning classes to experts still leaves tail classes with too few samples per expert; specialists are not actually specialized.
3. **Confusion matrix leakage in code_mat.py**: The confusion matrix is derived from predictions on the validation split of the *same* dataset. Confusion patterns on val don't transfer to test, especially with chrono/file splits. Val looks nearly perfect because the probe model sees a similar distribution as its training set.
4. **XGBoost's boosting already does implicit MoE**: Gradient boosting iteratively reweights hard examples, so adding explicit routing on top offers little additional specialization.

## Environment Setup

```bash
pip install -r requirements.txt
# For CUDA 12.x GPU acceleration (optional):
pip install cupy-cuda12x
# For other CUDA versions: cupy-cuda11x etc.
```

PyTorch is used only for CUDA device detection. XGBoost handles training. CuPy is optional GPU acceleration.

## Data Preparation

Raw CSV data: `data/cic2017/`, `data/cic2018/`, `data/unswnb15/`, `data/nf-unswnb15/`.

```bash
python data/3_preprocessed.py --data_dir data/cic2017 --output cic2017_proc.pkl
python data/3_preprocessed.py --data_dir data/unswnb15 --output unswnb15_proc.pkl
```

Preprocessing removes leakage columns (IPs, ports, flow IDs, timestamps), cleans repeated headers and NaN rows, frequency-encodes categoricals, and encodes labels with `LabelEncoder`. Dataset type inferred from folder name (must contain `2017`, `2018`, `unsw`, or `nb15`).

Pre-processed files: `cic2017_proc.pkl`, `unswnb15_proc.pkl`, `nfunswnb15_prep.pkl`.

## Running Models

### `cic_xg_19.py`

```bash
python cic_xg_19.py --data cic2017_proc.pkl \
  --num_experts 4 --n_estimators 200 --router_n_estimators 200 \
  --models baseline ensemble --smote_threshold 1000 \
  --assignment_mode type --router_target expert --seed 42
```

### `cic_6.py`

```bash
python cic_6.py --data cic2017_proc.pkl --model 2  # 0=baseline only, 1=MoE only, 2=both
```

### `code_mat.py` (FAR-MoE)

```bash
python code_mat.py --data cic2017_proc.pkl --model 2 --seed 42 --batch_size 20000
```

All scripts use `--data <pkl_file>` and write results to `results/<timestamp>/experiment.log`.

## Architecture of `code_mat.py` (most recent)

`code_mat.py` implements **FAR-MoE** (Family-Aware Reliability-routed MoE):

```
Train split
  ├── Baseline probe → predict on val → confusion matrix
  │       └── Top confused class pairs → ConfMat_Focus expert
  └── Full MoE training:
        ├── Anchor expert (all classes, always selected)
        ├── Taxonomy family experts (Group1–4, each with a feature view subset)
        ├── ConfMat_Focus expert (confused classes from val)
        ├── Tail-class expert (< threshold samples)
        └── Random specialists (random class subsets, random feature views)

Val split → fit_router():
  Each expert gets a LogisticRegression competence model
  Features: family router P(family), margin, entropy, OOD z-score
  Labels: was this expert correct on this val sample?

Test split → predict():
  1. Always include anchor + generalist experts
  2. For each specialist: include if OOD z-score ≤ threshold
  3. Rank remaining by competence score → take top_k
  4. Weighted avg of selected expert global probabilities → argmax
```

**Expert feature views**: `volume` (byte/packet counts), `timing` (IAT, duration), `packet` (flag/size features), `tcp` (TCP-specific), `all`.

**OOD detection**: Per-expert Mahalanobis-style distance using `LedoitWolf` covariance on training features. High OOD z-score → expert not competent on this sample.

## Data Flow

`.pkl` keys: `X` (float32), `y` (int), `label_encoder`, `feature_names`, `dataset_type`.

Split strategy varies by script:
- Random stratified (code_final, cic_xg_19, cic_6): 60/20/20 — known to have temporal leakage
- Chronological (code_chrono): episodes split in time order within each class
- Per-file (code_file): whole capture files assigned to train/val/test
- `code_mat.py` uses the chrono-style `apply_attack_episode_ratio_split`

## Dataset-Specific Notes

Label columns: `"Label"` (CIC-IDS), `"attack_cat"` (UNSW-NB15), `"Attack"` (NF-UNSW-NB15). CIC labels may have leading/trailing spaces — preprocessing strips them.

## Open Research Problem

The core unsolved problem: **no MoE variant has beaten the single XGBoost baseline on tail-class F1, particularly under non-random (chrono/file) splits.** See "Known failure modes" above. New approaches should directly address either (a) the unbiased expert design problem or (b) the routing error compounding problem.
