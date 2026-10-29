#!/usr/bin/env python3
"""EXP6 combination lab -- offline, no GPU.  Pre-registration: 0820.md SS9/SS9a.

Protocol (honest two-split): for each candidate ensemble, the mixing knob
(gamma, shared scalar; per-expert partial prior correction p/pi_k^gamma before
renormalize+average) and the combo verdict are SELECTED ON VAL (near-future
slice) and CONFIRMED ON TEST exactly once.  Test tables printed here for
combos are only the confirmations of val-selected configs.

Inputs (run dirs):
  exp5 D1 test:  results/*_exp5_viewmoe/        probs_tabpfn_v{a100,a050,a000}.npz
  exp5 E2  val:  results/*_exp5_viewmoe_evval/  (same file names)
  exp6 round1:   results/*_exp6_{recipe}/ and *_exp6_{recipe}_evval/
                 probs_{tabpfn,xgb}_{recipe}.npz (context prior inside)

Candidate ensembles (registered):
  base           = D1 three views (tabpfn), uniform average
  base + <r>     = base plus one exp6 tabpfn recipe        (4 experts)
Exploratory (labeled, not part of the registered gates):
  base + xgb_<r> = hybrid backbone extension
Per-combo target class: web_ctrl/web_hn -> web_attacks; inf_hn -> infiltration;
dos_pair -> dos (ddos reported); bal50k -> macro.  Gate vs confirmed base:
web +0.05 / inf +0.05 / dos +0.03, tuple held (macro >= base-0.010 legs on
val at selection; reported on test).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

GAMMAS = [-1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
RECIPES = ["bal50k", "web_hn", "web_ctrl", "inf_hn", "dos_pair"]
TARGET = {"bal50k": "macro_avg", "web_hn": "web_attacks", "web_ctrl": "web_attacks",
          "inf_hn": "infiltration", "dos_pair": "dos"}
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _latest(pattern):
    hits = sorted(glob.glob(os.path.join(RES, pattern)))
    if not hits:
        raise SystemExit(f"missing run dir: {pattern}")
    return hits[-1]


def load_exp5(split):
    d = _latest("*_exp5_viewmoe" + ("_evval" if split == "val" else ""))
    comp = pd.read_csv(os.path.join(d, "view_composition.csv"))
    out = {}
    y = classes = None
    for atag in ("a100", "a050", "a000"):
        z = np.load(os.path.join(d, f"probs_tabpfn_v{atag}.npz"))
        classes = list(z["class_names"])
        y = z["y_true"]
        rows = comp[comp["view"] == atag].set_index("class")["rows"]
        prior = np.array([rows.get(c, 0) for c in classes], float)
        out[f"view_{atag}"] = (z["probs"].astype(np.float64), prior / prior.sum())
    return out, y, classes


def load_exp6(split):
    out = {}
    for r in RECIPES:
        d = _latest(f"*_exp6_{r}" + ("_evval" if split == "val" else ""))
        for bk in ("tabpfn", "xgb"):
            z = np.load(os.path.join(d, f"probs_{bk}_{r}.npz"))
            prior = z["context_class_counts"].astype(float)
            out[f"{bk}_{r}"] = (z["probs"].astype(np.float64), prior / prior.sum())
    return out


def combine(experts, names, gamma):
    parts = []
    for n in names:
        p, prior = experts[n]
        w = p / np.maximum(prior, 1e-12)[None, :] ** gamma
        parts.append(w / w.sum(axis=1, keepdims=True))
    return np.mean(parts, axis=0)


def metrics(pred, y, classes):
    f1 = {}
    for c, cn in enumerate(classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1[cn] = 2 * tp / max(2 * tp + fp + fn, 1)
    f1["macro_avg"] = float(np.mean([f1[c] for c in classes]))
    return f1


def select_on_val(experts_val, y_val, classes, names, target, base_macro_val):
    """Pick gamma maximizing target F1 on VAL under the tuple legs."""
    best = None
    for g in GAMMAS:
        f1 = metrics(np.argmax(combine(experts_val, names, g), axis=1), y_val, classes)
        ok = f1["macro_avg"] >= base_macro_val - 0.010 and f1["benign"] >= 0.990
        score = f1[target] if target in f1 else f1["macro_avg"]
        if ok and (best is None or score > best[1]):
            best = (g, score, f1)
    if best is None:  # no gamma satisfies the legs; fall back to max-macro gamma, flagged
        for g in GAMMAS:
            f1 = metrics(np.argmax(combine(experts_val, names, g), axis=1), y_val, classes)
            if best is None or f1["macro_avg"] > best[1]:
                best = (g, f1["macro_avg"], f1)
        return best[0], best[2], False
    return best[0], best[2], True


def main():
    ev_val_experts, y_val, classes = load_exp5("val")
    ev_val_experts.update(load_exp6("val"))
    ev_test_experts, y_test, _ = load_exp5("test")
    ev_test_experts.update(load_exp6("test"))

    base_names = ["view_a100", "view_a050", "view_a000"]
    rows = []

    def run_combo(label, names, target):
        # base macro reference on val at its own best gamma (legs use it)
        g0, f1v0, _ = select_on_val(ev_val_experts, y_val, classes, base_names,
                                    "web_attacks", -1.0)
        base_macro_val = f1v0["macro_avg"]
        g, f1v, legs_ok = select_on_val(ev_val_experts, y_val, classes, names,
                                        target, base_macro_val)
        f1t = metrics(np.argmax(combine(ev_test_experts, names, g), axis=1),
                      y_test, classes)
        rows.append({
            "combo": label, "target": target, "gamma_val": g,
            "legs_ok_val": legs_ok,
            "target_F1_test": round(f1t.get(target, f1t["macro_avg"]), 4),
            "web": round(f1t["web_attacks"], 4),
            "infiltration": round(f1t["infiltration"], 4),
            "dos": round(f1t["dos"], 4), "ddos": round(f1t["ddos"], 4),
            "macro": round(f1t["macro_avg"], 4),
            "benign": round(f1t["benign"], 4),
        })
        return f1t

    base_test = run_combo("base(3view)", base_names, "web_attacks")
    for r in RECIPES:
        run_combo(f"base+{r}", base_names + [f"tabpfn_{r}"], TARGET[r])
    for r in RECIPES:  # exploratory hybrid, labeled
        run_combo(f"base+XGB:{r} [hybrid]", base_names + [f"xgb_{r}"], TARGET[r])

    df = pd.DataFrame(rows)
    print("=== combination lab (gamma selected on VAL, numbers below are the "
          "one-shot TEST confirmations) ===")
    print(df.to_string(index=False))
    print("\ngates vs confirmed base: web +0.05 / infiltration +0.05 / dos +0.03, "
          f"base = web {base_test['web_attacks']:.4f} / inf "
          f"{base_test['infiltration']:.4f} / dos {base_test['dos']:.4f} / "
          f"macro {base_test['macro_avg']:.4f}")
    out = os.path.join(RES, "exp6_combination_lab.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
