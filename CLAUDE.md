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

## Visualization Standards

All main scripts must produce a **colored PNG comparison table** (saved to `results/<timestamp>/`) in addition to the plain-text log. Rules:

### Table layout
- Columns interleave Baseline and MoE for each metric in fixed order:
  `prec(B) | prec(M) | recall(B) | recall(M) | f1(B) | f1(M) | support`
- Rows are **descending by support** (majority class first, tail classes last).
- Append `macro avg` and `weighted avg` footer rows.

### Cell color coding (MoE cell vs. corresponding Baseline cell)
| Condition | Color |
|---|---|
| MoE > Baseline + ε (ε = 0.001) | Blue (`#cce5ff`) |
| MoE < Baseline − ε | Red (`#ffcccc`) |
| Within ε (neutral) | Yellow (`#fff9cc`) |
| Baseline column cell | White (no tint) |
| support / avg rows | Light gray (`#f2f2f2`) |

Only **MoE metric cells** are colored; Baseline cells stay white.

### Implementation helper
Use `matplotlib` + `matplotlib.table` or `pandas` + `matplotlib` to render the table as a PNG. A shared helper function `save_colored_table(df, path)` should be placed near the top of `results/` or inlined in each script. Minimum font size 9pt; save at 150 dpi. Example skeleton:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BLUE   = "#cce5ff"
RED    = "#ffcccc"
YELLOW = "#fff9cc"
GRAY   = "#f2f2f2"
WHITE  = "#ffffff"
EPS    = 0.001

def save_colored_table(rows, col_headers, path, title=""):
    """
    rows  : list of dicts with keys matching col_headers.
    Baseline metric cols end in '(B)', MoE cols in '(M)'.
    """
    import numpy as np
    n_rows = len(rows)
    n_cols = len(col_headers)
    cell_text = []
    cell_colors = []
    for row in rows:
        texts, colors = [], []
        for col in col_headers:
            val = row.get(col, "")
            texts.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            # determine color
            if col.endswith("(M)"):
                base_col = col.replace("(M)", "(B)")
                b_val = row.get(base_col, None)
                m_val = row.get(col, None)
                if isinstance(b_val, float) and isinstance(m_val, float):
                    if m_val > b_val + EPS:
                        colors.append(BLUE)
                    elif m_val < b_val - EPS:
                        colors.append(RED)
                    else:
                        colors.append(YELLOW)
                else:
                    colors.append(WHITE)
            elif col in ("macro avg", "weighted avg", "support") or row.get("_footer"):
                colors.append(GRAY)
            else:
                colors.append(WHITE)
        cell_text.append(texts)
        cell_colors.append(colors)

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.2), max(4, n_rows * 0.35)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_headers,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(n_cols)))
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

Also save the same data as a CSV (`baseline_vs_moe_per_class.csv`) with columns:
`class_name, support, prec(B), prec(M), recall(B), recall(M), f1(B), f1(M), delta_f1`

## Open Research Problem

The core unsolved problem: **no MoE variant has beaten the single XGBoost baseline on tail-class F1, particularly under non-random (chrono/file) splits.** See "Known failure modes" above. New approaches should directly address either (a) the unbiased expert design problem or (b) the routing error compounding problem.
