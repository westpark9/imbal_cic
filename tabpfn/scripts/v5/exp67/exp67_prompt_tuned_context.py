#!/usr/bin/env python3
"""EXP67 (feasibility) — in-context data distillation / prompt tuning: optimize a SMALL context by gradient
descent through the frozen TabPFN (Ma, Thomas et al., "In-Context Data Distillation with TabPFN", ICLR 2024 WS;
API: TabPFNClassifier(differentiable_input=True).fit_with_differentiable_input, see examples/prompt_tuning_classifier.py).

Setup (seed 43): context initialised as a class-balanced sample of the D_global pool (standardized features, labels
fixed), queries sampled class-balanced from the rest of D_global (never test), NLL minimised w.r.t. the context
features only. Arms on the SAME test subsample:
  init     : the initial small context, untuned (n_estimators=1, no sklearn preprocessing)
  tuned    : after --steps optimisation steps
  c0_ref   : the recorded 100k C0 Global (4 estimators, standard pipeline) — the number the small context must approach
Memory: ~18 GiB at 5,000 context rows on a 24 GB GPU; larger contexts OOM in the differentiable path.
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
import time

import numpy as np
import torch

from exp6x_context_common import load_dataset, standardizer, test_subsample, summarize, tabpfn_global, predict, run_dir, dump, CKPT


def balanced_sample(y, per_class, rng, exclude=None):
    out = []
    for c in range(int(y.max()) + 1):
        idx = np.flatnonzero(y == c)
        if exclude is not None:
            idx = np.setdiff1d(idx, exclude)
        out.append(rng.choice(idx, min(per_class, len(idx)), replace=False))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='toniot', choices=['toniot', 'cic2018'])
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--ctx-per-class', type=int, default=500, help='initial context rows per class (benign included)')
    ap.add_argument('--benign-ctx', type=int, default=None, help='override benign rows in the context (default = ctx-per-class)')
    ap.add_argument('--query-pool-per-class', type=int, default=4000)
    ap.add_argument('--query-batch', type=int, default=512)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--query-sampling', default='balanced', choices=['balanced', 'natural'])
    ap.add_argument('--test-per-class', type=int, default=20000)
    ap.add_argument('--full-test', action='store_true')
    ap.add_argument('--skip-c0-ref', action='store_true')
    args = ap.parse_args(); print('Args:', vars(args), flush=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); d = load_dataset(args.dataset); C = d['C']
    Z = standardizer(d['route_X'][:200000]); dev = 'cuda'
    out = run_dir(args.dataset, 'exp67_prompt_tuned', args.seed)
    te_idx, tX, ty = test_subsample(d, args.test_per_class, rng, args.full_test)
    pool_y = d['pool_y']
    ctx = balanced_sample(pool_y, args.ctx_per_class, rng)
    if args.benign_ctx is not None:
        b = d['benign']; keep = ctx[pool_y[ctx] != b]; ctx = np.concatenate([keep, rng.choice(np.flatnonzero(pool_y == b), args.benign_ctx, replace=False)])
    ctx = np.sort(ctx)
    if args.query_sampling == 'balanced':
        qry = balanced_sample(pool_y, args.query_pool_per_class, rng, exclude=ctx)
    else:
        rest = np.setdiff1d(np.arange(len(pool_y)), ctx); qry = rng.choice(rest, args.query_pool_per_class * C, replace=False)
    qry = np.sort(qry)
    px0 = Z(d['pool_X'][ctx]); py = pool_y[ctx]
    px = torch.tensor(px0, device=dev); pyt = torch.tensor(py, dtype=torch.float32, device=dev)
    qx = torch.tensor(Z(d['pool_X'][qry]), device=dev); qy = torch.tensor(pool_y[qry], dtype=torch.long, device=dev)
    tXz = torch.tensor(Z(tX))
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(model_path=str(CKPT), ignore_pretraining_limits=True, device=dev, n_estimators=1, random_state=args.seed,
                           inference_precision=torch.float32, differentiable_input=True)
    clf.n_classes_ = C
    print(f'context {len(ctx):,} rows {np.bincount(py, minlength=C).tolist()} | query pool {len(qry):,} | test {len(ty):,}', flush=True)

    @torch.no_grad()
    def evaluate(x):
        clf.fit_with_differentiable_input(x.detach(), pyt); preds = []
        for i in range(0, len(tXz), 2048):
            preds.append(clf.forward(tXz[i:i + 2048].to(dev), use_inference_mode=True).argmax(1).cpu().numpy())
        return summarize(ty, np.concatenate(preds), d)

    results = {}
    t0 = time.time(); results['init'] = evaluate(px); results['init']['seconds'] = time.time() - t0
    print('init  macro-F1 %.4f tail %.4f | %s' % (results['init']['macro_f1'], results['init']['tail_f1'], ' '.join(f'{r["cls"]}={r["f1"]:.3f}' for r in results['init']['classes'])), flush=True)
    px.requires_grad_(True); opt = torch.optim.Adam([px], lr=args.lr); lossfn = torch.nn.NLLLoss(); log = []; nan_grad_entries = []
    t0 = time.time()
    for step in range(args.steps):
        perm = torch.randperm(len(qy), device=dev)[:args.query_batch]
        opt.zero_grad(); clf.fit_with_differentiable_input(px, pyt); p = clf.forward(qx[perm], use_inference_mode=True)
        loss = lossfn(torch.log(p.clamp_min(1e-9)), qy[perm])
        if not torch.isfinite(loss):
            opt.zero_grad(); continue
        loss.backward()
        with torch.no_grad():
            bad = ~torch.isfinite(px.grad); nan_grad_entries.append(int(bad.sum().item())); px.grad[bad] = 0.0
        opt.step()
        with torch.no_grad():
            px.nan_to_num_(0.0, 10.0, -10.0); px.clamp_(-10, 10)
        log.append(float(loss.item()))
        if (step + 1) % 25 == 0:
            print(f'step {step+1} query NLL {np.mean(log[-25:]):.4f} | non-finite grad entries (last 25 steps) {sum(nan_grad_entries[-25:])} ({time.time()-t0:.0f}s)', flush=True)
    px.requires_grad_(False)
    assert torch.isfinite(px).all(), 'context contains non-finite values after tuning'
    results['tuned'] = evaluate(px); results['tuned']['seconds'] = time.time() - t0; results['tuned']['query_nll'] = log; results['tuned']['non_finite_grad_entries'] = nan_grad_entries
    print('tuned macro-F1 %.4f tail %.4f | %s' % (results['tuned']['macro_f1'], results['tuned']['tail_f1'], ' '.join(f'{r["cls"]}={r["f1"]:.3f}' for r in results['tuned']['classes'])), flush=True)
    np.save(out / 'context_init_X.npy', px0); np.save(out / 'context_tuned_X.npy', px.detach().cpu().numpy()); np.save(out / 'context_y.npy', py); np.save(out / 'context_pool_ids.npy', d['pool_ids'][ctx])
    drift = np.linalg.norm(px.detach().cpu().numpy() - px0, axis=1); results['tuned']['context_drift'] = dict(median=float(np.median(drift)), p90=float(np.percentile(drift, 90)), max=float(drift.max()))
    del clf; torch.cuda.empty_cache()
    if not args.skip_c0_ref:
        ref, fit_s = tabpfn_global(d['c0_X'], d['c0_y'], args.seed); t1 = time.time(); pred = predict(ref, tX)
        results['c0_ref'] = summarize(ty, pred, d); results['c0_ref'].update(fit_seconds=fit_s, predict_seconds=time.time() - t1, context_rows=int(len(d['c0_y'])))
        print('c0_ref macro-F1 %.4f tail %.4f' % (results['c0_ref']['macro_f1'], results['c0_ref']['tail_f1']), flush=True)
    dump(out / 'results.json', dict(args=vars(args), dataset=args.dataset, class_names=d['names'], context_rows=int(len(ctx)), context_counts=np.bincount(py, minlength=C).tolist(), test_rows=int(len(ty)), results=results))
    print('saved', out)


if __name__ == '__main__':
    main()
