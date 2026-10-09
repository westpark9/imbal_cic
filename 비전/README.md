# Computer Vision — Homework 1

From-scratch implementation of classic image-processing operations (no built-in
functions that directly perform the operation; only `numpy`, `numpy.fft`, `PIL`
for loading, and `matplotlib` for display are used).

The assignment has three problems:

- **Part A — Point Processing & Histogram-Based Enhancement** (RGB→YUV, histogram,
  HE, contrast stretching, gamma, AHE, CLAHE, bit-plane slicing). **Complete.**
- **Part B — Spatial- & Frequency-Domain Filtering.** *(planned)*
- **Part C — Wiener Filter.** *(planned)*

## Setup (any PC)

```bash
# Python 3.10+ recommended (developed on 3.13)
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Input images

Required inputs live in `images/` (already included):

| file | used by |
|---|---|
| `point_processing_input_rgb.png` | Part A (color RGB) |
| `spatial_frequency_filtering_input.png` | Part B (grayscale) |
| `wiener_filter_input.png` (+ alternates) | Part C (grayscale) |

To use different images, replace the files in `images/` (keep the same names) and re-run.

## How to run

```bash
# 1) Part A code: runs all experiments, writes figures to output/, prints metrics
python partA_point_processing.py

# 2) Word report (English, full-assignment format; Part A filled in):
#    runs Part A internally for live numbers/figures -> Homework1_Report.docx
python build_report.py

# 3) (Re)build the full solution notebook, then execute it:
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace Homework1_CV_solution.ipynb

# 4) (Optional) CLAHE clipping exploration notebook (not part of the report):
python build_explore.py
jupyter nbconvert --to notebook --execute --inplace clahe_clip_explore.ipynb
```

Or simply open `Homework1_CV_solution.ipynb` in Jupyter / VS Code and run all cells.

## Files

| file | description |
|---|---|
| `partA_point_processing.py` | Part A code (from scratch), saves figures + prints metrics |
| `build_report.py` | builds `Homework1_Report.docx` (English) |
| `build_notebook.py` | builds `Homework1_CV_solution.ipynb` (bilingual KR/EN explanations) |
| `build_explore.py` | builds `clahe_clip_explore.ipynb` (exploration, not submitted) |
| `images/` | input images |
| `output/` | generated figures (A1…A9); re-created by the scripts |
| `Homework1_CV.pdf` | assignment instructions |
| `*.pdf` (lecture notes), `강의스크립트/` | reference material |

## Notes

- All figure/axis text in the submitted scripts and report is in English.
- `build_explore.py` renders some Korean labels and sets the `Malgun Gothic` font
  (available on Windows). On a non-Windows PC without that font, those labels may
  fall back to the default font; the rest is unaffected.
- `output/` is committed so figures display without running, but any run regenerates it.
