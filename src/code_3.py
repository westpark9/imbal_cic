#!/usr/bin/env python3
"""
code_3.py – ConfMat-MoE
========================
Confusion-matrix-guided Mixture-of-Experts on CIFAR-10 / CIFAR-100.

Research question
-----------------
code_mat.py used a confusion-matrix-derived expert on CIC-IDS2017 and failed
(leakage between val/test distributions).  Here we test the same structural
idea on CIFAR-10/100—balanced, well-studied benchmarks—to determine whether
the failure was due to (a) a domain-specific data artefact or (b) the
confusion-matrix expert design being fundamentally weak.

Architecture
------------
1. PROBE phase   : train one XGBoost on 60 % train split.
2. CLUSTER phase : predict on 20 % val → normalised confusion matrix →
                   union-find clustering of mutually-confused class pairs
                   (off-diagonal conf[i,j] > τ).
3. EXPERT phase  : train K+1 XGBoost experts on full training data.
   • Anchor expert  – uniform sample weights.
   • ConfMat expert_k – samples from cluster k get weight w_focus (default 5);
                         all others weight 1.
4. ROUTER phase  : for each expert, fit a Logistic-Regression competence model
                   on val:  P(expert_e correct | x).
5. INFERENCE     : final proba = Σ_e α_e(x) · proba_e(x),
                   where α_e(x) = competence_e.predict_proba(x)[:,1].

Usage
-----
  python code_3.py                        # CIFAR-10 + CIFAR-100
  python code_3.py --datasets cifar10     # CIFAR-10 only
  python code_3.py --conf_threshold 0.08 --w_focus 8
"""

import argparse
import datetime
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    import torchvision
    import torchvision.transforms as T
    HAS_TV = True
except ImportError:
    HAS_TV = False

# ── Visualization (CLAUDE.md spec) ────────────────────────────────────────────
BLUE   = "#cce5ff"
RED    = "#ffcccc"
YELLOW = "#fff9cc"
GRAY   = "#f2f2f2"
WHITE  = "#ffffff"
EPS    = 0.001


def save_colored_table(rows, col_headers, path, title=""):
    n_cols = len(col_headers)
    n_rows = len(rows)
    cell_text, cell_colors = [], []
    for row in rows:
        texts, colors = [], []
        for col in col_headers:
            val = row.get(col, "")
            texts.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            if col.endswith("(M)"):
                b = row.get(col.replace("(M)", "(B)"))
                m = row.get(col)
                if isinstance(b, float) and isinstance(m, float):
                    colors.append(BLUE if m > b + EPS else (RED if m < b - EPS else YELLOW))
                else:
                    colors.append(WHITE)
            elif row.get("_footer") or col == "support":
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


# ── Union-Find ────────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)

    def groups(self):
        from collections import defaultdict
        d = defaultdict(list)
        for i in range(len(self.p)):
            d[self.find(i)].append(i)
        return list(d.values())


# ── ConfMat-MoE ───────────────────────────────────────────────────────────────
class ConfMatMoE:
    """
    Confusion-matrix-guided MoE with XGBoost experts.

    Attributes
    ----------
    experts_      : list of fitted XGBClassifier (anchor + conf experts)
    expert_labels_: list of str – human-readable name per expert
    cluster_sets_ : list of sets – class indices in each confusion cluster
    routers_      : list of fitted LR – competence P(expert correct | x)
    clusters_     : list of list[int] – classes per confusion expert
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        conf_threshold: float = 0.05,
        w_focus: float = 5.0,
        router_C: float = 1.0,
        tree_method: str = "hist",
        device: str = "cpu",
        seed: int = 42,
    ):
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.learning_rate   = learning_rate
        self.subsample       = subsample
        self.colsample_bytree = colsample_bytree
        self.conf_threshold  = conf_threshold
        self.w_focus         = w_focus
        self.router_C        = router_C
        self.tree_method     = tree_method
        self.device          = device
        self.seed            = seed

    # ── helpers ──────────────────────────────────────────────────────────────
    def _xgb_params(self):
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric="mlogloss",
            random_state=self.seed,
            n_jobs=-1,
            tree_method=self.tree_method,
            device=self.device,
            verbosity=0,
        )

    def _find_clusters(self, y_val_true, y_val_pred, n_classes):
        """Return list of clusters (each a list of class indices, len >= 2)."""
        cm = confusion_matrix(y_val_true, y_val_pred,
                              labels=list(range(n_classes))).astype(float)
        # row-normalise (each row sums to 1, gives recall-style confusion rate)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_norm = cm / row_sum

        uf = UnionFind(n_classes)
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm_norm[i, j] > self.conf_threshold:
                    uf.union(i, j)

        clusters = [g for g in uf.groups() if len(g) >= 2]
        return clusters, cm_norm

    def _sample_weights(self, y, cluster):
        """Weight vector: cluster classes → w_focus, others → 1.0."""
        w = np.ones(len(y), dtype=np.float32)
        mask = np.isin(y, cluster)
        w[mask] = self.w_focus
        return w

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, X_tr, y_tr, X_val, y_val):
        log = logging.getLogger(__name__)
        n_classes = int(y_tr.max()) + 1
        xp = self._xgb_params()

        # ── Phase 1: probe on train → predict val → confusion matrix ─────────
        log.info("  [Phase 1] Probe training ...")
        probe = xgb.XGBClassifier(**xp)
        probe.fit(X_tr, y_tr, verbose=False)
        y_val_pred = probe.predict(X_val)

        clusters, cm_norm = self._find_clusters(y_val, y_val_pred, n_classes)
        self.clusters_  = clusters
        self.cm_norm_   = cm_norm
        log.info(f"  [Phase 1] Found {len(clusters)} confusion cluster(s): "
                 f"{[len(c) for c in clusters]} classes each")
        for ci, cl in enumerate(clusters):
            log.info(f"    Cluster {ci}: classes {cl}")

        # ── Phase 2: train experts ────────────────────────────────────────────
        self.experts_       = []
        self.expert_labels_ = []

        # Anchor – uniform weights
        log.info("  [Phase 2] Training anchor expert ...")
        anchor = xgb.XGBClassifier(**xp)
        anchor.fit(X_tr, y_tr, verbose=False)
        self.experts_.append(anchor)
        self.expert_labels_.append("anchor")

        # ConfMat experts – one per cluster
        for ci, cluster in enumerate(clusters):
            log.info(f"  [Phase 2] Training confusion expert {ci} "
                     f"(classes {cluster}, w_focus={self.w_focus}) ...")
            w = self._sample_weights(y_tr, cluster)
            exp = xgb.XGBClassifier(**xp)
            exp.fit(X_tr, y_tr, sample_weight=w, verbose=False)
            self.experts_.append(exp)
            self.expert_labels_.append(f"conf_{ci}")

        n_experts = len(self.experts_)
        log.info(f"  [Phase 2] {n_experts} expert(s) trained.")

        # ── Phase 3: fit competence router on val ─────────────────────────────
        log.info("  [Phase 3] Fitting competence routers on val ...")
        self.routers_ = []
        for ei, exp in enumerate(self.experts_):
            y_hat = exp.predict(X_val)
            correct = (y_hat == y_val).astype(int)
            # need at least one positive and one negative for LR
            if correct.sum() == 0 or correct.sum() == len(correct):
                # degenerate: constant predictor – use uniform weights
                self.routers_.append(None)
            else:
                lr = LogisticRegression(
                    C=self.router_C,
                    max_iter=300,
                    random_state=self.seed,
                    n_jobs=-1,
                )
                lr.fit(X_val, correct)
                self.routers_.append(lr)
            acc = correct.mean()
            log.info(f"    Expert {self.expert_labels_[ei]:12s}: val acc={acc:.4f}")

        return self

    # ── predict ───────────────────────────────────────────────────────────────
    def predict_proba(self, X):
        """Weighted average of expert probas; weights = competence score."""
        n = len(X)
        n_experts = len(self.experts_)
        # shape (n_experts, n_samples)
        alpha = np.zeros((n_experts, n), dtype=np.float64)
        for ei, (exp, lr) in enumerate(zip(self.experts_, self.routers_)):
            if lr is None:
                alpha[ei] = 0.5  # uniform if degenerate
            else:
                alpha[ei] = lr.predict_proba(X)[:, 1]

        # softmax-normalise across experts per sample
        alpha_exp = np.exp(alpha - alpha.max(axis=0, keepdims=True))
        alpha_norm = alpha_exp / alpha_exp.sum(axis=0, keepdims=True)  # (E, N)

        # weighted sum of expert probas
        proba_sum = None
        for ei, exp in enumerate(self.experts_):
            p = exp.predict_proba(X)               # (N, C)
            w = alpha_norm[ei, :, np.newaxis]      # (N, 1)
            if proba_sum is None:
                proba_sum = p * w
            else:
                proba_sum += p * w
        return proba_sum

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_cifar(kind: str, n_pca: int = 128, seed: int = 42):
    """
    Load CIFAR-10 or CIFAR-100, flatten, PCA, stratified 60/20/20 split.
    Returns X_tr, y_tr, X_val, y_val, X_test, y_test, class_names.
    """
    assert HAS_TV, "torchvision required for CIFAR"
    import torch

    transform = T.Compose([T.ToTensor()])
    root = os.path.expanduser("~/.cache/torchvision")

    if kind == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=transform)
        test_ds  = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)
        class_names = train_ds.classes
    elif kind == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(root, train=True,  download=True, transform=transform)
        test_ds  = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=transform)
        class_names = train_ds.classes
    else:
        raise ValueError(f"Unknown dataset: {kind}")

    def ds_to_np(ds):
        imgs = np.stack([np.array(img) for img, _ in ds]).reshape(len(ds), -1).astype(np.float32) / 255.0
        lbls = np.array([lbl for _, lbl in ds])
        return imgs, lbls

    X_all_tr, y_all_tr = ds_to_np(train_ds)
    X_all_te, y_all_te = ds_to_np(test_ds)

    # combine all → stratified 60/20/20
    X_all = np.concatenate([X_all_tr, X_all_te], axis=0)
    y_all = np.concatenate([y_all_tr, y_all_te], axis=0)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    tr_i, tmp_i = next(sss1.split(X_all, y_all))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va_i, te_i = next(sss2.split(X_all[tmp_i], y_all[tmp_i]))

    X_tr   = X_all[tr_i]
    y_tr   = y_all[tr_i]
    X_val  = X_all[tmp_i][va_i]
    y_val  = y_all[tmp_i][va_i]
    X_test = X_all[tmp_i][te_i]
    y_test = y_all[tmp_i][te_i]

    # PCA
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_val  = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    n_pca = min(n_pca, X_tr.shape[1], X_tr.shape[0])
    pca = PCA(n_components=n_pca, random_state=seed)
    X_tr   = pca.fit_transform(X_tr).astype(np.float32)
    X_val  = pca.transform(X_val).astype(np.float32)
    X_test = pca.transform(X_test).astype(np.float32)

    logging.getLogger(__name__).info(
        f"  {kind}: train={len(y_tr)} val={len(y_val)} test={len(y_test)} "
        f"PCA={n_pca} explained={pca.explained_variance_ratio_.sum():.3f}"
    )
    return X_tr, y_tr, X_val, y_val, X_test, y_test, class_names


# ── Experiment runner ─────────────────────────────────────────────────────────
COL_HEADERS = [
    "class", "support",
    "prec(B)", "prec(M)",
    "recall(B)", "recall(M)",
    "f1(B)", "f1(M)",
    "delta_f1",
]


def run_experiment(name, X_tr, y_tr, X_val, y_val, X_test, y_test,
                   class_names, out_dir, args):
    log = logging.getLogger(__name__)
    log.info(f"\n{'='*60}")
    log.info(f"Dataset : {name}")
    log.info(f"Train={X_tr.shape}  Val={X_val.shape}  Test={X_test.shape}")
    log.info(f"Classes : {len(class_names)}")

    xp = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=-1,
        tree_method=args.tree_method,
        device=args.device,
        verbosity=0,
    )

    # ── Baseline ──────────────────────────────────────────────────────────────
    log.info("\n[Baseline] Training ...")
    t0 = time.time()
    baseline = xgb.XGBClassifier(**xp)
    baseline.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    log.info(f"[Baseline] Done in {time.time()-t0:.1f}s")

    y_base = baseline.predict(X_test)
    rep_b  = classification_report(y_test, y_base, output_dict=True, zero_division=0)
    log.info(f"[Baseline] macro-F1={rep_b['macro avg']['f1-score']:.4f}  "
             f"wt-F1={rep_b['weighted avg']['f1-score']:.4f}")

    # ── ConfMat-MoE ────────────────────────────────────────────────────────────
    log.info("\n[ConfMat-MoE] Training ...")
    t0 = time.time()
    moe = ConfMatMoE(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        conf_threshold=args.conf_threshold,
        w_focus=args.w_focus,
        router_C=args.router_C,
        tree_method=args.tree_method,
        device=args.device,
        seed=args.seed,
    )
    moe.fit(X_tr, y_tr, X_val, y_val)
    log.info(f"[ConfMat-MoE] Done in {time.time()-t0:.1f}s")

    y_moe = moe.predict(X_test)
    rep_m = classification_report(y_test, y_moe, output_dict=True, zero_division=0)
    log.info(f"[ConfMat-MoE] macro-F1={rep_m['macro avg']['f1-score']:.4f}  "
             f"wt-F1={rep_m['weighted avg']['f1-score']:.4f}")

    delta = rep_m['macro avg']['f1-score'] - rep_b['macro avg']['f1-score']
    log.info(f"[Delta] macro-F1 MoE - Base = {delta:+.4f}")

    # ── Per-class table ────────────────────────────────────────────────────────
    all_cls = sorted(np.unique(y_test))

    def cname(c):
        return class_names[c] if c < len(class_names) else str(c)

    rows = []
    for c in all_cls:
        k = str(c)
        bm = rep_b.get(k, {})
        dm = rep_m.get(k, {})
        rows.append({
            "class":     cname(c),
            "support":   int(bm.get("support", 0)),
            "prec(B)":   float(bm.get("precision", 0.0)),
            "prec(M)":   float(dm.get("precision", 0.0)),
            "recall(B)": float(bm.get("recall", 0.0)),
            "recall(M)": float(dm.get("recall", 0.0)),
            "f1(B)":     float(bm.get("f1-score", 0.0)),
            "f1(M)":     float(dm.get("f1-score", 0.0)),
            "delta_f1":  float(dm.get("f1-score", 0.0)) - float(bm.get("f1-score", 0.0)),
        })
    rows.sort(key=lambda r: r["support"], reverse=True)

    for avg in ("macro avg", "weighted avg"):
        bm = rep_b.get(avg, {})
        dm = rep_m.get(avg, {})
        rows.append({
            "class":   avg, "support": int(bm.get("support", 0)),
            "prec(B)": float(bm.get("precision", 0.0)),
            "prec(M)": float(dm.get("precision", 0.0)),
            "recall(B)": float(bm.get("recall", 0.0)),
            "recall(M)": float(dm.get("recall", 0.0)),
            "f1(B)":   float(bm.get("f1-score", 0.0)),
            "f1(M)":   float(dm.get("f1-score", 0.0)),
            "delta_f1": float(dm.get("f1-score", 0.0)) - float(bm.get("f1-score", 0.0)),
            "_footer": True,
        })

    for r in rows:
        tag = "  [avg]" if r.get("_footer") else "      "
        log.info(f"{tag} {r['class']:30s} sup={r['support']:6d}  "
                 f"f1(B)={r['f1(B)']:.4f}  f1(M)={r['f1(M)']:.4f}  "
                 f"Δ={r['delta_f1']:+.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    safe = name.replace("/", "_").replace(" ", "_")
    png_path = out_dir / f"{safe}_comparison.png"
    csv_path = out_dir / f"{safe}_comparison.csv"

    save_colored_table(rows, COL_HEADERS, png_path,
                       title=f"ConfMat-MoE vs Baseline — {name}")

    # Save cluster info
    cluster_rows = []
    for ci, cl in enumerate(moe.clusters_):
        names = [cname(c) for c in cl]
        cluster_rows.append({
            "expert": f"conf_{ci}",
            "n_classes": len(cl),
            "classes": "; ".join(names),
        })
    cluster_rows.append({
        "expert": "anchor", "n_classes": len(class_names),
        "classes": "(all classes, uniform weights)",
    })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(cluster_rows).to_csv(
        out_dir / f"{safe}_clusters.csv", index=False)

    log.info(f"Saved: {png_path}")
    log.info(f"Saved: {csv_path}")

    return {
        "dataset":        name,
        "n_clusters":     len(moe.clusters_),
        "macro_f1_base":  rep_b['macro avg']['f1-score'],
        "macro_f1_moe":   rep_m['macro avg']['f1-score'],
        "delta_macro_f1": delta,
        "wt_f1_base":     rep_b['weighted avg']['f1-score'],
        "wt_f1_moe":      rep_m['weighted avg']['f1-score'],
        "cluster_sizes":  str([len(c) for c in moe.clusters_]),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["cifar10", "cifar100"],
                   choices=["cifar10", "cifar100"])
    p.add_argument("--n_pca",          type=int,   default=128)
    p.add_argument("--n_estimators",   type=int,   default=200)
    p.add_argument("--max_depth",      type=int,   default=6)
    p.add_argument("--learning_rate",  type=float, default=0.1)
    p.add_argument("--subsample",      type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--conf_threshold", type=float, default=0.05,
                   help="Normalised confusion rate above which two classes are 'confused'")
    p.add_argument("--w_focus",        type=float, default=5.0,
                   help="Sample weight for cluster classes in ConfMat experts")
    p.add_argument("--router_C",       type=float, default=1.0)
    p.add_argument("--tree_method",    default="hist")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"code3_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "experiment.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger(__name__)
    log.info(f"Output dir : {out_dir}")
    log.info(f"Args       : {vars(args)}")

    summary_rows = []

    for ds in args.datasets:
        n_pca = args.n_pca if ds == "cifar10" else min(256, args.n_pca * 2)
        log.info(f"\n{'='*60}\nLoading {ds.upper()} (PCA={n_pca}) ...")
        X_tr, y_tr, X_val, y_val, X_test, y_test, cnames = load_cifar(
            ds, n_pca=n_pca, seed=args.seed)

        sr = run_experiment(
            ds.upper(), X_tr, y_tr, X_val, y_val, X_test, y_test,
            cnames, out_dir, args)
        summary_rows.append(sr)

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'Dataset':15s} {'Clusters':>8} {'MacroF1-B':>10} "
             f"{'MacroF1-M':>10} {'Delta':>8}")
    for r in summary_rows:
        sign = "▲" if r["delta_macro_f1"] > EPS else ("▼" if r["delta_macro_f1"] < -EPS else "–")
        log.info(f"{r['dataset']:15s} {r['n_clusters']:>8d} "
                 f"{r['macro_f1_base']:>10.4f} {r['macro_f1_moe']:>10.4f} "
                 f"{r['delta_macro_f1']:>+8.4f} {sign}")

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    log.info(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
