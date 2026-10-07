# October 2 lab meeting · English edition

English edition of the latest 10-slide Korean deck, including the completed CIC2018 and ToN EXP70 results. Figures, numeric cells, class order and experimental scope are preserved. Text, matrix annotations and speaker notes are translated; line breaks and font sizes are adjusted for English.

| Page | Content |
|---|---|
| 1 | TabPFN for operational intrusion detection |
| 2 | Updating an IDS as new attack labels arrive |
| 3 | CIC2018 · New attack performance and cost |
| 4 | ToN · New attack performance and cost |
| 5 | Data review: remove rows with conflicting labels |
| 6 | Class distribution after cleaning and the fixed test set |
| 7 | Global and expert models built from contexts |
| 8 | Current static performance · SOTA comparison |
| 9 | SOTA cost on cleaned data |
| 10 | Next steps · An IDS that incorporates new attacks |

Outputs: `1002_labmeeting_en_draft.pptx` and `1002_labmeeting_en_draft.pdf`.

Rebuild from the current Korean PPTX with `python docs/slides/translate_1002_en.py`. Translation content is in `docs/slides/1002_english_translation.json`. The source PPTX hash is recorded in `1002_operational_qa/validation.json`; the prior English deck is archived under `../archive/20261002_before_english_update/`.

Validation: all 156 numeric table cells unchanged; numeric columns right aligned; no Korean text in slides or notes; four data-review matrices redrawn from the original CSVs; all 10 pages rendered and inspected; no missing text or text outside page bounds. Korean PPTX and PDF hashes are unchanged.

The static proposed-model row remains EXP63 K=4 with S/V. The arrival experiment is EXP70 Global TabPFN versus XGBoost and is not substituted for the full model. This distinction is retained in the slides and speaker notes.
