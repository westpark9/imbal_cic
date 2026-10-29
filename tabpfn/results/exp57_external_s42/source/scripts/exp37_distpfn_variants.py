#!/usr/bin/env python3
"""Post-hoc analysis for an exp37 (DistPFN) run dir.

    python scripts/exp37_distpfn_variants.py tabpfn/results/<ts>_nfv3_cic2018_exp37_distpfn

Recomputes, from the saved 4.02M x 7 posteriors (tabpfn_v3_test_proba.npy /
xgboost_test_proba.npy), the metrics the DistPFN paper and its released code use
(accuracy, log loss, macro ROC-AUC ovr; the paper itself reports accuracy / rank /
ECE / precision and never F1) next to this project's macro-F1, for:

  raw                    the model's posterior
  distpfn                paper eq. (3)/(4): Norm(p * P_test_avg / prior)        (README snippet)
  distpfn_t_scalar       paper eq. (5)/(6), default "Multiple" mode: one global
                         tau = CE(P_test_avg, prior), tempered = softmax(P_test_avg / tau)
                         (softmax over the PROBABILITY vector), Norm(p * tempered / prior)
                         -- this is what nfv3_v3_exp37_distpfn.py runs
  distpfn_t_persample    the paper's Appendix-M "Single" variant / the only reading under
                         which the repo's helper functions type-check: tau_i = CE(p_i, prior),
                         tempered_i = softmax(p_i / tau_i)
  saerens_em             Saerens-Latinne-Decaestecker EM to convergence (DistPFN = its first step)

Writes into the run dir: 9a_distpfn_variants_metrics.csv, 9b_priors_log.png,
9c_f1_variants.png, 9z_y_test_eval.npy (the reconstructed evaluation labels; the run
itself does not save them).  The evaluation labels are rebuilt from the same loader
and must reproduce the per-class supports in per_class_metrics.csv (asserted).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score, balanced_accuracy_score,
                             log_loss, roc_auc_score)

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMBALCIC_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RUN = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
    IMBALCIC_ROOT, "tabpfn", "results", "20260904_183615_nfv3_cic2018_exp37_distpfn")
OUT = RUN
YCACHE = os.path.join(RUN, "9z_y_test_eval.npy")
classes = ["benign", "bot", "brute_force", "ddos", "dos", "infiltration", "web_attacks"]
K = len(classes)


def load_labels(run_args):
    sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "scripts"))
    sys.path.insert(0, os.path.join(IMBALCIC_ROOT, "tabpfn", "scripts"))
    import nfv3_v3_common as core
    cfg = core.build_dataset_config(run_args["data_dir"])
    a = argparse.Namespace(data=run_args["data"])
    X, cn, tr, va, te, ytr, yte, aud, lf = cfg[run_args["target_dataset"]]["loader"](a)
    assert list(cn) == classes, cn
    te_eval = core.cap_per_class(te, yte, K, run_args["test_cap_per_class"], run_args["seed"] + 900)
    return lf(te_eval).astype(np.int64)


t0 = time.time()
run_args = pd.read_json(os.path.join(RUN, "args.json"), typ="series").to_dict()
if os.path.exists(YCACHE):
    y = np.load(YCACHE)
else:
    y = load_labels(run_args)
    np.save(YCACHE, y)
print(f"labels ready in {time.time()-t0:.0f}s: n={len(y):,} counts={np.bincount(y, minlength=K).tolist()}", flush=True)
pcm = pd.read_csv(os.path.join(RUN, "per_class_metrics.csv"))
sup = pcm[(pcm.method == "xgboost") & (~pcm["class"].isin(["macro_avg", "weighted_avg", "tail_avg"]))].set_index("class")["support"]
assert [int(sup[c]) for c in classes] == np.bincount(y, minlength=K).tolist(), "label order mismatch"
shift = pd.read_csv(os.path.join(RUN, "1a_prior_shift.csv"))
prior = shift["prior_train_context"].values.astype(np.float64)
eps = float(run_args.get("distpfn_eps", 1e-8))


def norm(p):
    return p / p.sum(axis=1, keepdims=True)


def distpfn(P):
    pa = P.mean(0)
    return norm(P * pa / (prior + eps)), pa


def distpfn_t_scalar(P):
    pa = P.mean(0)
    tau = float(-np.sum(pa * np.log(np.clip(prior, 1e-12, 1))))
    pt = softmax(pa / tau)
    return norm(P * pt / (prior + eps)), pt, tau


def distpfn_t_persample(P):
    tau = -(P * np.log(np.clip(prior, 1e-12, 1))).sum(1, keepdims=True)
    pt = softmax(P / tau, axis=1)
    return norm(P * pt / (prior + eps)), tau


def distpfn_t_readme_literal(P):
    """The one reading under which the README / eval_TABPFN_shift_O.py helpers type-check
    with a ONE-TOKEN change (cross_entropy(y_prob, prior) instead of cross_entropy(P_test_avg,
    prior)): per-sample tau_i = CE(p_i, prior) (N,), then softmax_temperature(P_test_avg, T=tau)
    broadcasts the GLOBAL average prior over each tau_i -> (N, C), and the adjustment is
    Norm(p_i * tempered_i / prior)."""
    pa = P.mean(0)
    tau = -(P * np.log(np.clip(prior, 1e-12, 1))).sum(1, keepdims=True)      # (N,1)
    pt = softmax(pa[None, :] / tau, axis=1)                                   # (N,C): global prior, per-row temperature
    return norm(P * pt / (prior + eps)), tau


def saerens_em(P, iters=50, tol=1e-8):
    pi = prior.copy()
    n_it = 0
    for n_it in range(1, iters + 1):
        adj = norm(P * (pi / (prior + eps)))
        new = adj.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return adj, pi, n_it


def metrics(name, P):
    pred = P.argmax(1)
    _, _, f1, _ = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    row = {"variant": name, "macro_f1": f1.mean(), "tail_f1": f1[[1, 5, 6]].mean(),
           "accuracy": accuracy_score(y, pred), "balanced_acc": balanced_accuracy_score(y, pred),
           "log_loss": log_loss(y, np.clip(P, 1e-12, 1), labels=list(range(K))),
           "roc_auc_ovr_macro": roc_auc_score(y, P, multi_class="ovr", average="macro")}
    row.update({f"f1_{c}": f1[i] for i, c in enumerate(classes)})
    return row


rows, priors, f1_bars = [], {"context_prior": prior, "true_test_prior": np.bincount(y, minlength=K) / len(y)}, {}
for base in ["tabpfn_v3", "xgboost"]:
    path = os.path.join(RUN, f"{base}_test_proba.npy")
    if not os.path.exists(path):
        continue
    P = norm(np.clip(np.load(path).astype(np.float64), 1e-12, 1))
    t = time.time()
    v = {"raw": P}
    v["distpfn"], pa = distpfn(P)
    v["distpfn_t_scalar"], pt, tau = distpfn_t_scalar(P)
    v["distpfn_t_persample"], tau_n = distpfn_t_persample(P)
    v["distpfn_t_readme_literal"], _ = distpfn_t_readme_literal(P)
    v["saerens_em"], pi_em, n_it = saerens_em(P)
    if base == "tabpfn_v3":
        priors.update({"P_test_avg": pa, "tempered_scalarT": pt, "EM_converged": pi_em})
    for name, Q in v.items():
        r = metrics(f"{base}:{name}", Q)
        rows.append(r)
        if base == "tabpfn_v3":
            f1_bars[name] = [r[f"f1_{c}"] for c in classes]
        print(f"{r['variant']:32s} macroF1 {r['macro_f1']:.4f} acc {r['accuracy']:.4f} bacc {r['balanced_acc']:.4f} "
              f"logloss {r['log_loss']:.4f} auc {r['roc_auc_ovr_macro']:.4f}", flush=True)
    print(f"  {base}: tau_scalar={tau:.4f}  tau_persample mean={tau_n.mean():.3f} [{tau_n.min():.3f},{tau_n.max():.3f}]  "
          f"EM iters={n_it} pi_em={np.round(pi_em, 4).tolist()}  ({time.time()-t:.0f}s)", flush=True)
    for name, key in [("raw", base), ("distpfn", f"{base}+distpfn"), ("distpfn_t_scalar", f"{base}+distpfn_t")]:
        rec = float(pcm[(pcm.method == key) & (pcm["class"] == "macro_avg")]["f1"].iloc[0])
        mine = [r for r in rows if r["variant"] == f"{base}:{name}"][0]["macro_f1"]
        print(f"  check {key}: run {rec:.4f} vs recomputed {mine:.4f} {'OK' if abs(rec - mine) < 5e-4 else 'MISMATCH'}")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "9a_distpfn_variants_metrics.csv"), index=False)
print(df.round(4).to_string(index=False))

x = np.arange(K)
w = 0.16
fig, ax = plt.subplots(figsize=(11, 4.2))
for i, (k, v) in enumerate(priors.items()):
    ax.bar(x + (i - 2) * w, v, w, label=k)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=20)
ax.set_ylabel("class prior (log)")
ax.set_title(f"{os.path.basename(RUN)}: context prior vs true test prior vs DistPFN estimates (TabPFN-v3)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "9b_priors_log.png"), dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.2))
names = list(f1_bars.keys())
w = 0.8 / max(len(names), 1)
for i, n in enumerate(names):
    ax.bar(x + (i - len(names) / 2 + 0.5) * w, f1_bars[n], w, label=n)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=20)
ax.set_ylabel("F1")
ax.set_ylim(0, 1.02)
ax.set_title("TabPFN-v3 per-class F1 on the evaluation set: raw vs DistPFN variants")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "9c_f1_variants.png"), dpi=140)
plt.close(fig)
print("wrote 9a/9b/9c into", OUT)
