#!/usr/bin/env python3
"""E-0: offline evaluation battery for RACE-PFN runs. CPU only, no GPU, no refit.

Recomputes the non-per-class evaluation axes (0831.md §3) from artifacts that
every `--dense-eval` run already writes:

    probs_racepfn_system.npz   probs (N,C) float32, y_true (N,)
    probs_tabpfn_global.npz    probs (N,C) float32, y_true (N,)
    system_dump.npz            proposal, accepted, g_lower, final, y_glob, ...
    route_gain.npz             gain (M,K), b_oof (M,K), y (M,)

Writes, into each run directory (CSV; PNGs via scripts/render_run_tables.py):

    9a_selective.csv     risk-coverage curve + AURC (balanced risk on y-axis:
                         macro-F1 is NOT monotone in coverage -- 0831 §3)
    9b_intervention.csv  per-class correctness transitions + Wilson CIs +
                         conditional accuracy on proposed / accepted subsets
    9c_cost.csv          proposal / call / override / wasted-call accounting
    9d_bank_utility.csv  per-expert solo oracle recovery, per-class utility
                         share, gain-correlation variance decomposition

and one cross-run table:

    <out>/9e_frontier.csv   harm-gain frontier over all runs given

Usage (from repo root):

    python scripts/race_offline_eval.py --runs tabpfn/results/2026082*_nfv3_cic2018_*
    python scripts/race_offline_eval.py --runs tabpfn/results/<one_run>   # single

Notes
-----
* npz files are gitignored, so run this on the box that produced the runs and
  commit only the resulting 9*.csv files.
* Nothing here refits a model; every number is a re-read of stored predictions.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

COVERAGES = (1.0, 0.99, 0.97, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30)


# --------------------------------------------------------------------------
# small stats helpers (copied here on purpose: shared modules stay untouched)
# --------------------------------------------------------------------------
def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def macro_f1(y_true, y_pred, n_classes):
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        f1[c] = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    return float(f1.mean()), f1


def balanced_error(y_true, y_pred, n_classes):
    """Mean over PRESENT classes of (1 - recall). Also returns #classes present."""
    errs, present = [], 0
    for c in range(n_classes):
        m = y_true == c
        n = int(m.sum())
        if n == 0:
            continue
        present += 1
        errs.append(1.0 - float((y_pred[m] == c).sum()) / n)
    return (float(np.mean(errs)) if errs else float("nan")), present


# --------------------------------------------------------------------------
# 9a -- selective prediction / risk-coverage
# --------------------------------------------------------------------------
def selective_table(run, probs_sys, probs_glob, y, g_lower, class_names):
    n_classes = len(class_names)
    rows = []
    scores = {
        "system_maxprob": (probs_sys.max(axis=1), probs_sys.argmax(axis=1)),
        "global_maxprob": (probs_glob.max(axis=1), probs_glob.argmax(axis=1)),
    }
    if g_lower is not None:
        # abstain on rows the verifier is least sure about, keep system preds
        scores["system_glower"] = (g_lower, probs_sys.argmax(axis=1))

    for name, (conf, pred) in scores.items():
        order = np.argsort(-conf, kind="stable")
        prev_cov, prev_risk, aurc = None, None, 0.0
        for cov in COVERAGES:
            k = max(int(round(len(y) * cov)), 1)
            idx = order[:k]
            berr, present = balanced_error(y[idx], pred[idx], n_classes)
            mf1, _ = macro_f1(y[idx], pred[idx], n_classes)
            err = float((pred[idx] != y[idx]).mean())
            rows.append({
                "run": run, "score": name, "coverage": cov, "rows": k,
                "balanced_error": round(berr, 6),
                "classes_present": present,
                "error_rate": round(err, 6),
                "macro_f1": round(mf1, 6),
            })
            if prev_cov is not None:                     # trapezoid on coverage
                aurc += 0.5 * (berr + prev_risk) * (prev_cov - cov)
            prev_cov, prev_risk = cov, berr
        rows.append({"run": run, "score": name, "coverage": "AURC_balanced",
                     "rows": len(y), "balanced_error": round(aurc, 6),
                     "classes_present": n_classes, "error_rate": "",
                     "macro_f1": ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 9b -- intervention quality
# --------------------------------------------------------------------------
def intervention_table(run, y, y_glob, y_final, proposal, accepted, class_names):
    rows = []
    prop = proposal > 0
    for c in range(len(class_names)):
        m = y == c
        n = int(m.sum())
        if n == 0:
            continue
        g_ok = y_glob[m] == c
        f_ok = y_final[m] == c
        helpful = int((~g_ok & f_ok).sum())
        harmful = int((g_ok & ~f_ok).sum())
        ww = int((~g_ok & ~f_ok).sum())
        ww_changed = int((~g_ok & ~f_ok & (y_glob[m] != y_final[m])).sum())
        acc_m = accepted[m]
        n_acc = int(acc_m.sum())
        lo, hi = wilson(harmful, max(int(g_ok.sum()), 1))
        prec_lo, prec_hi = wilson(helpful, max(helpful + harmful, 1))
        rows.append({
            "run": run, "class": class_names[c], "rows": n,
            "proposed": int(prop[m].sum()), "accepted": n_acc,
            "g_ok_f_ok": int((g_ok & f_ok).sum()),
            "helpful": helpful, "harmful": harmful,
            "wrong_wrong": ww, "wrong_wrong_changed": ww_changed,
            "net_correction": helpful - harmful,
            "net_per_1M": round(1e6 * (helpful - harmful) / n, 3),
            "harmful_rate_of_correct": round(harmful / max(int(g_ok.sum()), 1), 8),
            "harmful_rate_lo": round(lo, 8), "harmful_rate_hi": round(hi, 8),
            "override_precision": round(helpful / max(helpful + harmful, 1), 6),
            "override_prec_lo": round(prec_lo, 6),
            "override_prec_hi": round(prec_hi, 6),
            "acc_global_on_proposed": round(
                float((y_glob[m][prop[m]] == c).mean()), 6) if prop[m].any() else "",
            "acc_final_on_proposed": round(
                float((y_final[m][prop[m]] == c).mean()), 6) if prop[m].any() else "",
            "acc_global_on_accepted": round(
                float((y_glob[m][acc_m] == c).mean()), 6) if n_acc else "",
            "acc_final_on_accepted": round(
                float((y_final[m][acc_m] == c).mean()), 6) if n_acc else "",
        })
    # overall row -- conditional accuracy is only interpretable next to counts
    g_ok = y_glob == y
    f_ok = y_final == y
    helpful = int((~g_ok & f_ok).sum())
    harmful = int((g_ok & ~f_ok).sum())
    lo, hi = wilson(harmful, max(int(g_ok.sum()), 1))
    prec_lo, prec_hi = wilson(helpful, max(helpful + harmful, 1))
    rows.append({
        "run": run, "class": "ALL", "rows": len(y),
        "proposed": int(prop.sum()), "accepted": int(accepted.sum()),
        "g_ok_f_ok": int((g_ok & f_ok).sum()), "helpful": helpful,
        "harmful": harmful, "wrong_wrong": int((~g_ok & ~f_ok).sum()),
        "wrong_wrong_changed": int((~g_ok & ~f_ok & (y_glob != y_final)).sum()),
        "net_correction": helpful - harmful,
        "net_per_1M": round(1e6 * (helpful - harmful) / len(y), 3),
        "harmful_rate_of_correct": round(harmful / max(int(g_ok.sum()), 1), 8),
        "harmful_rate_lo": round(lo, 8), "harmful_rate_hi": round(hi, 8),
        "override_precision": round(helpful / max(helpful + harmful, 1), 6),
        "override_prec_lo": round(prec_lo, 6), "override_prec_hi": round(prec_hi, 6),
        "acc_global_on_proposed": round(float(g_ok[prop].mean()), 6) if prop.any() else "",
        "acc_final_on_proposed": round(float(f_ok[prop].mean()), 6) if prop.any() else "",
        "acc_global_on_accepted": round(float(g_ok[accepted].mean()), 6)
        if accepted.any() else "",
        "acc_final_on_accepted": round(float(f_ok[accepted].mean()), 6)
        if accepted.any() else "",
    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 9c -- call accounting
# --------------------------------------------------------------------------
def cost_table(run, y, proposal, accepted, class_names, timings):
    prop = proposal > 0
    rows = [{
        "run": run, "scope": "ALL", "rows": len(y),
        "proposed": int(prop.sum()),
        "proposal_rate": round(float(prop.mean()), 6),
        "accepted": int(accepted.sum()),
        "accept_rate_of_proposed": round(
            float(accepted.sum()) / max(int(prop.sum()), 1), 6),
        "rejected_calls": int(prop.sum() - accepted.sum()),
        "wasted_call_rate": round(
            float(prop.sum() - accepted.sum()) / max(int(prop.sum()), 1), 6),
        "calls_per_sample_nominal": round(1.0 + float(prop.mean()), 6),
    }]
    for c in range(len(class_names)):
        m = y == c
        if not m.any():
            continue
        rows.append({
            "run": run, "scope": class_names[c], "rows": int(m.sum()),
            "proposed": int(prop[m].sum()),
            "proposal_rate": round(float(prop[m].mean()), 6),
            "accepted": int(accepted[m].sum()),
            "accept_rate_of_proposed": round(
                float(accepted[m].sum()) / max(int(prop[m].sum()), 1), 6),
            "rejected_calls": int(prop[m].sum() - accepted[m].sum()),
            "wasted_call_rate": round(
                float(prop[m].sum() - accepted[m].sum()) / max(int(prop[m].sum()), 1), 6),
            "calls_per_sample_nominal": round(1.0 + float(prop[m].mean()), 6),
        })
    for k in range(1, int(proposal.max()) + 1):
        m = proposal == k
        if not m.any():
            continue
        rows.append({
            "run": run, "scope": f"expert{k}", "rows": int(m.sum()),
            "proposed": int(m.sum()), "proposal_rate": "",
            "accepted": int(accepted[m].sum()),
            "accept_rate_of_proposed": round(
                float(accepted[m].sum()) / int(m.sum()), 6),
            "rejected_calls": int(m.sum() - accepted[m].sum()),
            "wasted_call_rate": round(
                float(m.sum() - accepted[m].sum()) / int(m.sum()), 6),
            "calls_per_sample_nominal": "",
        })
    # measured compute multiplier -- 0831 §1: "1.158 calls" understates this
    if timings:
        eg = timings.get("eval_global_s")
        es = timings.get("eval_scorer_s")
        ee = timings.get("eval_experts_s")
        if eg:
            mult = (eg + (es or 0) + (ee or 0)) / eg
            rows.append({
                "run": run, "scope": "MEASURED_COMPUTE", "rows": "",
                "proposed": "", "proposal_rate": "", "accepted": "",
                "accept_rate_of_proposed": "", "rejected_calls": "",
                "wasted_call_rate": "",
                "calls_per_sample_nominal": round(mult, 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 9d -- bank utility / expert redundancy
# --------------------------------------------------------------------------
def bank_table(run, gain, y_route, class_names):
    K = gain.shape[1]
    best = np.maximum(gain.max(axis=1), 0.0)
    total = float(best.sum())
    rows = []
    for k in range(K):
        solo = float(np.maximum(gain[:, k], 0).sum())
        rows.append({
            "run": run, "item": f"expert{k + 1}_solo_recovery",
            "value": round(solo / max(total, 1e-12), 6),
            "note": "share of full-bank oracle utility recovered alone"})
    # variance decomposition: G_ik = L_i0 - L_ik share the SAME L_i0, which
    # inflates raw gain correlation (0831 §1 cause E).
    C = np.atleast_2d(np.cov(gain.T))          # K==1 -> np.cov returns 0-d
    V0 = float((C.sum() - np.trace(C)) / (K * K - K)) if K > 1 else float("nan")
    for k in range(K):
        Vk = float(C[k, k]) - V0
        rows.append({
            "run": run, "item": f"expert{k + 1}_shared_variance_share",
            "value": round(V0 / max(V0 + Vk, 1e-12), 6),
            "note": "fraction of gain variance that is the shared global term"})
    for c in range(len(class_names)):
        m = y_route == c
        if not m.any():
            continue
        rows.append({
            "run": run, "item": f"utility_share_{class_names[c]}",
            "value": round(float(best[m].sum()) / max(total, 1e-12), 6),
            "note": f"n={int(m.sum())}"})
        if K > 1:
            a, b = gain[m][:, 0] > 0, gain[m][:, 1] > 0
            rows.append({
                "run": run, "item": f"posJaccard_e1e2_{class_names[c]}",
                "value": round(float((a & b).sum()) / max(int((a | b).sum()), 1), 6),
                "note": "positive-gain set overlap, within class"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def process(run_dir, out_rows):
    run = os.path.basename(run_dir.rstrip("/"))
    need = ["probs_racepfn_system.npz", "probs_tabpfn_global.npz",
            "system_dump.npz"]
    if any(not os.path.exists(os.path.join(run_dir, f)) for f in need):
        print(f"  skip {run}: missing npz", flush=True)
        return
    S = np.load(os.path.join(run_dir, "probs_racepfn_system.npz"))
    G = np.load(os.path.join(run_dir, "probs_tabpfn_global.npz"))
    D = np.load(os.path.join(run_dir, "system_dump.npz"), allow_pickle=True)
    class_names = [str(x) for x in D["class_names"]]
    y = S["y_true"]
    probs_sys, probs_glob = S["probs"], G["probs"]
    y_glob, y_final = D["y_glob"], D["final"]
    proposal, accepted = D["proposal"], D["accepted"].astype(bool)
    g_lower = D["g_lower"] if "g_lower" in D.files else None
    timings = {}
    tp = os.path.join(run_dir, "timings.json")
    if os.path.exists(tp):
        t = json.load(open(tp))
        timings = t[0] if isinstance(t, list) else t

    selective_table(run, probs_sys, probs_glob, y, g_lower, class_names) \
        .to_csv(os.path.join(run_dir, "9a_selective.csv"), index=False)
    itab = intervention_table(run, y, y_glob, y_final, proposal, accepted,
                              class_names)
    itab.to_csv(os.path.join(run_dir, "9b_intervention.csv"), index=False)
    cost_table(run, y, proposal, accepted, class_names, timings) \
        .to_csv(os.path.join(run_dir, "9c_cost.csv"), index=False)

    rg = os.path.join(run_dir, "route_gain.npz")
    if os.path.exists(rg):
        R = np.load(rg)
        bank_table(run, R["gain"].astype(np.float64), R["y"], class_names) \
            .to_csv(os.path.join(run_dir, "9d_bank_utility.csv"), index=False)

    # cross-run frontier row
    pc = os.path.join(run_dir, "per_class_metrics.csv")
    sysf = globf = xgbf = None
    if os.path.exists(pc):
        t = pd.read_csv(pc)
        t = t[t["class"] == "macro_avg"].set_index("method")["f1"]
        sysf = float(t.get("racepfn_system", float("nan")))
        globf = float(t.get("global_tabpfn", float("nan")))
        xgbf = float(t.get("xgboost", float("nan")))
    a = itab[itab["class"] == "ALL"].iloc[0]
    out_rows.append({
        "run": run, "system_macro_f1": sysf, "global_macro_f1": globf,
        "paired_delta": None if (sysf is None or globf is None) else round(sysf - globf, 6),
        "xgboost_macro_f1": xgbf,
        "helpful": int(a["helpful"]), "harmful": int(a["harmful"]),
        "override_precision": a["override_precision"],
        "harmful_rate_hi": a["harmful_rate_hi"],
        "accepted": int(a["accepted"]), "proposed": int(a["proposed"]),
        "calls_per_sample_nominal": round(1.0 + float((proposal > 0).mean()), 6),
        "measured_compute_mult": round(
            (timings.get("eval_global_s", 0) + timings.get("eval_scorer_s", 0)
             + timings.get("eval_experts_s", 0))
            / timings["eval_global_s"], 4) if timings.get("eval_global_s") else None,
        "total_s": timings.get("total_s"),
        "peak_gpu_mem_gb": timings.get("peak_gpu_mem_gb"),
    })
    print(f"  ok   {run}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories (globs accepted)")
    ap.add_argument("--out", default="results/scripts_analysis",
                    help="where 9e_frontier.csv is written")
    args = ap.parse_args()

    dirs = []
    for pat in args.runs:
        dirs.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[")
                    else [pat])
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise SystemExit("no run directories matched")
    print(f"processing {len(dirs)} run(s)")

    out_rows = []
    for d in dirs:
        process(d, out_rows)

    if out_rows:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "9e_frontier.csv")
        pd.DataFrame(out_rows).to_csv(p, index=False)
        print(f"\nfrontier -> {p}")
        print(pd.DataFrame(out_rows).to_string(index=False))


if __name__ == "__main__":
    main()
