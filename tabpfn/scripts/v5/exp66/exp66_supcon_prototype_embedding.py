#!/usr/bin/env python3
"""EXP66 (feasibility) — supervised-contrastive encoder with prototype label embeddings, used as extra
TabPFN input features.

Idea source: Lopez-Martin et al., "Supervised contrastive learning over prototype-label embeddings for
network intrusion detection" (user link). Here: a small MLP encoder f(x) in R^d is trained on the route
pool with (a) SupCon loss over class-balanced batches and (b) a learnable prototype per class pulled to its
members (prototype-label embedding). The frozen TabPFN then receives [raw 46 features + d embedding
features] (or embedding only) with the SAME C0 context rows; the only knob is the feature set.

Protocol guard: the encoder is fit on D_route rows only (never test, never the C0 rows themselves);
evaluation is a stratified test subsample. Single seed, feasibility scale.
"""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = repo_root(__file__)
POOLS = ROOT / 'data/derived/pools_s43'
CACHE = {'toniot': ROOT / 'tabpfn/results/v4/exp63/20260930_exp63_k_sweep_s43/toniot_k6/cache',
         'cic2018': ROOT / 'tabpfn/results/v4/exp63/20260930_exp63_k_sweep_s43/cic2018_k6/cache'}
CKPT = ROOT / 'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'


def parse():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='toniot', choices=['toniot', 'cic2018'])
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--dim', type=int, default=16, help='embedding size appended to the raw features')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--steps-per-epoch', type=int, default=300)
    ap.add_argument('--batch-per-class', type=int, default=64)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--proto-weight', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--test-per-class', type=int, default=20000, help='stratified test subsample cap per class (all rows of smaller classes)')
    ap.add_argument('--n-estimators', type=int, default=4)
    ap.add_argument('--feature-sets', default='raw,raw+emb,emb', help='comma list of TabPFN input variants to evaluate')
    ap.add_argument('--out', type=Path, default=None)
    return ap.parse_args()


class Encoder(torch.nn.Module):
    def __init__(self, n_in, hidden, dim, n_classes):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, dim))
        self.prototypes = torch.nn.Parameter(torch.randn(n_classes, dim) * 0.1)

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def supcon_loss(z, y, t):
    sim = z @ z.T / t
    n = len(y)
    mask_self = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask_self, -1e9)
    pos = (y[:, None] == y[None, :]) & ~mask_self
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * pos).sum(1).div(pos.sum(1).clamp_min(1)).mean()


def proto_loss(z, y, protos, t):
    logits = z @ F.normalize(protos, dim=1).T / t
    return F.cross_entropy(logits, y)


def macro_table(y, pred, names):
    rows = []
    for i, n in enumerate(names):
        tp = int(((pred == i) & (y == i)).sum()); fp = int(((pred == i) & (y != i)).sum()); fn = int(((pred != i) & (y == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0; f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append(dict(cls=n, support=int((y == i).sum()), precision=p, recall=r, f1=f, FP=fp))
    return rows


def main():
    args = parse(); print('Args:', vars(args), flush=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    ds = args.dataset; cache = CACHE[ds]; pool = POOLS / ds
    names = json.load(open(cache / 'COMPLETE.json'))['class_names']; C = len(names)
    out = args.out or ROOT / f'tabpfn/results/v5/exp66/{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}_{ds}_exp66_supcon_s{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    RX = np.load(cache / 'route_X.npy'); Ry = np.load(cache / 'route_y.npy').astype(int)
    c0X = np.load(pool / 'c0_X.npy'); c0y = np.load(pool / 'c0_y.npy').astype(int)
    EX = np.load(cache / 'eval_X.npy', mmap_mode='r'); Ey = np.load(cache / 'eval_y.npy').astype(int)
    mu = RX.mean(0); sd = RX.std(0) + 1e-6
    Z = lambda X: np.clip((np.asarray(X, dtype=np.float32) - mu) / sd, -10, 10).astype(np.float32)
    # test subsample: all rows of small classes, cap for large ones (stratified, seeded)
    te = np.concatenate([rng.choice(np.flatnonzero(Ey == c), min(args.test_per_class, int((Ey == c).sum())), replace=False) for c in range(C)])
    te = np.sort(te); tX = np.asarray(EX[te]); ty = Ey[te]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ---- encoder training on the route pool (class-balanced batches)
    enc = Encoder(RX.shape[1], args.hidden, args.dim, C).to(dev); opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-4)
    by_class = [np.flatnonzero(Ry == c) for c in range(C)]
    RXz = torch.tensor(Z(RX)); Ryt = torch.tensor(Ry)
    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        tot = 0.0
        for _ in range(args.steps_per_epoch):
            idx = np.concatenate([rng.choice(ix, min(args.batch_per_class, len(ix)), replace=len(ix) < args.batch_per_class) for ix in by_class])
            xb = RXz[idx].to(dev); yb = Ryt[idx].to(dev)
            z = enc(xb); loss = supcon_loss(z, yb, args.temperature) + args.proto_weight * proto_loss(z, yb, enc.prototypes, args.temperature)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        log.append(dict(epoch=ep + 1, loss=tot / args.steps_per_epoch, seconds=time.time() - t0)); print(json.dumps(log[-1]), flush=True)
    enc.eval()

    @torch.no_grad()
    def embed(X):
        outs = []
        for i in range(0, len(X), 65536):
            outs.append(enc(torch.tensor(Z(X[i:i + 65536])).to(dev)).cpu().numpy())
        return np.concatenate(outs).astype(np.float32)

    e_c0, e_te = embed(c0X), embed(tX)
    torch.save(enc.state_dict(), out / 'encoder.pt')
    # ---- TabPFN with the same C0 rows, three feature sets
    from tabpfn import TabPFNClassifier
    results = {}
    variants = {'raw': (c0X, tX), 'raw+emb': (np.hstack([c0X, e_c0]), np.hstack([tX, e_te])), 'emb': (e_c0, e_te)}
    for name in args.feature_sets.split(','):
        cx, qx = variants[name]
        clf = TabPFNClassifier(model_path=str(CKPT), device=dev, n_estimators=args.n_estimators, random_state=args.seed, ignore_pretraining_limits=True, fit_mode='fit_with_cache')
        t1 = time.time(); clf.fit(cx, c0y); fit_s = time.time() - t1
        preds = []
        for i in range(0, len(qx), 100000):
            preds.append(clf.predict_proba(qx[i:i + 100000]).argmax(1))
        pred = np.concatenate(preds); t2 = time.time() - t1 - fit_s
        rows = macro_table(ty, pred, names); macro = float(np.mean([r['f1'] for r in rows]))
        results[name] = dict(macro_f1=macro, accuracy=float((pred == ty).mean()), fit_seconds=fit_s, predict_seconds=t2, classes=rows, n_features=int(cx.shape[1]))
        print(name, f'macro-F1 {macro:.4f}', 'per-class F1', {r['cls']: round(r['f1'], 3) for r in rows}, flush=True)
        del clf; torch.cuda.empty_cache()
    json.dump(dict(args=vars(args) | {'out': str(out)}, dataset=ds, class_names=names, test_rows=int(len(te)), test_counts=np.bincount(ty, minlength=C).tolist(), encoder_log=log, results=results), open(out / 'results.json', 'w'), ensure_ascii=False, indent=2, default=str)
    print('saved', out)


if __name__ == '__main__':
    main()
