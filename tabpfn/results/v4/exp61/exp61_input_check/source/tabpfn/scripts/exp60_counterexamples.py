#!/usr/bin/env python3
"""EXP60: matched random/near cross-class context examples, frozen EXP59 bank.

Only D_expert builds contexts. Global, anchor, PCA, scales, R/O evaluation
memberships, seed, and per-expert context budgets are inherited from EXP59.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

import numpy as np
import pandas as pd

from exp59_residual_membership_oracle import write, sha, confusion, scores

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'tabpfn/results/20260928_exp59_residual_oracle_s43'
FROZEN = ROOT / 'tabpfn/results/20260922_143529_exp57_expert_quality_s42_44/source'
ARMS = ['counter_random', 'counter_near']
SEED = 43
FRACTION = 0.20
QUERY_LIMIT = 512


def core_subset(reference, y, fraction, seed):
    """Retain a common stratified core with exact total and all original classes."""
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y[reference], return_counts=True)
    counter = min(max(1, int(round(len(reference) * fraction))), len(reference) - len(labels))
    target = len(reference) - counter
    allocation = np.ones(len(labels), dtype=int)
    remaining = target - len(labels)
    capacity = counts - 1
    if remaining:
        raw = capacity * (remaining / capacity.sum())
        extra = np.floor(raw).astype(int)
        for i in np.argsort(-(raw - extra), kind='stable')[:remaining-extra.sum()]:
            extra[i] += 1
        allocation += extra
    selected = np.concatenate([rng.choice(reference[y[reference] == c], size=int(n), replace=False)
                               for c, n in zip(labels, allocation)])
    assert len(selected) == target
    return np.sort(selected), counter


def query_subset(core, y, hashes, residual, limit, seed):
    """Distinct representative seeds, balanced across retained true classes.

    Residual-weighted draws retain correct-but-difficult samples as well as errors.
    No class is defined as the unique specialty of an expert.
    """
    _, first = np.unique(hashes[core], return_index=True)
    unique = core[first]
    labels = np.unique(y[unique])
    rng = np.random.default_rng(seed)
    quota = max(1, int(np.ceil(limit / len(labels))))
    selected = []
    for c in labels:
        rows = unique[y[unique] == c]
        weights = np.maximum(residual[rows], 1e-8).astype(float)
        selected.extend(rng.choice(rows, size=min(quota, len(rows)), replace=False,
                                   p=weights / weights.sum()).tolist())
    return np.asarray(selected, dtype=np.int64)


def round_robin_neighbors(indices, distances, queries, y, budget, seed):
    """Unique candidate per context; each recorded pair has different labels."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(queries))
    used, pairs = set(), []
    for rank in range(indices.shape[1]):
        for q in order:
            idx = int(indices[q, rank])
            if idx < 0 or idx in used:
                continue
            assert y[idx] != y[queries[q]]
            used.add(idx)
            pairs.append((int(queries[q]), idx, float(distances[q, rank])))
            if len(pairs) == budget:
                return pairs
    return pairs


def matched_random(pairs, candidates, y, seed):
    """Same query, counterpart class, counts and eligible pool as near pairs."""
    rng = np.random.default_rng(seed)
    chosen = np.asarray([p[1] for p in pairs], dtype=np.int64)
    random_ids = np.empty(len(chosen), dtype=np.int64)
    for c in np.unique(y[chosen]):
        positions = np.flatnonzero(y[chosen] == c)
        pool = candidates[y[candidates] == c]
        random_ids[positions] = rng.choice(pool, len(positions), replace=False)
    assert len(np.unique(random_ids)) == len(random_ids)
    assert np.array_equal(y[random_ids], y[chosen])
    return random_ids


def nearest_pairs(obs, y, candidates, queries, budget, seed, progress):
    import faiss
    faiss.omp_set_num_threads(16)
    trees = []
    for c in np.unique(y[candidates]):
        pool = np.sort(candidates[y[candidates] == c])
        tree = faiss.IndexFlatL2(obs.shape[1])
        tree.add(np.ascontiguousarray(obs[pool], dtype=np.float32))
        trees.append((c, pool, tree))
    width = max(64, int(np.ceil(budget / len(queries))) * 4)
    while True:
        progress('nearest_counterexample_search', queries=len(queries), neighbors_per_class=width,
                 requested=budget, eligible_candidates=len(candidates))
        all_i, all_d = [], []
        for c, pool, tree in trees:
            selected = np.flatnonzero(y[queries] != c)
            k = min(width, len(pool))
            ix = np.full((len(queries), k), -1, dtype=np.int64)
            dist = np.full((len(queries), k), np.inf, dtype=np.float32)
            if len(selected):
                dd, ii = tree.search(np.ascontiguousarray(obs[queries[selected]], dtype=np.float32), k)
                ix[selected] = pool[ii]
                dist[selected] = dd
            all_i.append(ix); all_d.append(dist)
        ix, dd = np.concatenate(all_i, 1), np.concatenate(all_d, 1)
        # Deterministic ranking for exact distance ties.
        order = np.lexsort((ix, dd), axis=1)
        ix, dd = np.take_along_axis(ix, order, 1), np.take_along_axis(dd, order, 1)
        pairs = round_robin_neighbors(ix, dd, queries, y, budget, seed)
        if len(pairs) == budget:
            return pairs
        if width >= max(len(p) for _, p, _ in trees):
            raise ValueError(f'Insufficient distinct cross-class examples: {len(pairs)}/{budget}')
        width *= 2


def frozen_helpers(dataset):
    sys.path.insert(0, str(FROZEN / 'tabpfn/scripts'))
    sys.path.insert(0, str(FROZEN / 'scripts'))
    name = 'exp60_frozen_helpers_' + dataset
    module = types.ModuleType(name)
    module.__file__ = str(FROZEN / 'tabpfn/scripts/nfv3_v3_exp31_c0alloc.py')
    sys.modules[name] = module
    code = (BASE / f'{dataset}_s43/executed_fresh_prefix.py').read_text()
    exec(compile(code, module.__file__, 'exec'), module.__dict__)
    return module


def feature_hash(X):
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()


class Worker:
    def __init__(self, root, dataset):
        self.root, self.dataset = root, dataset
        self.out = root / f'{dataset}_s43'
        self.base = BASE / f'{dataset}_s43'
        self.source = self.base / 'fresh_bank'
        self.started = time.time()
        for d in ['', 'contexts', 'predictions', 'probabilities', 'cache']:
            (self.out / d).mkdir(parents=True, exist_ok=True)

    def progress(self, phase, **kw):
        obj = dict(dataset=self.dataset, phase=phase, pid=os.getpid(), updated_epoch=time.time(),
                   elapsed_seconds=time.time()-self.started, **kw)
        write(self.out / 'progress.json', obj)
        print('EXP60 ' + json.dumps(obj), flush=True)

    def infer(self, clf, X, stage, embedding=False, batch=500000):
        import torch
        hashes = feature_hash(X)
        _, first, inv = np.unique(hashes, return_index=True, return_inverse=True)
        chunks = []
        for start in range(0, len(first), batch):
            self.progress(stage, done=start, total=len(first), original_rows=len(X))
            part = X[first[start:start+batch]]
            a = np.asarray(clf.get_embeddings(part, 'test') if embedding else clf.predict_proba(part))
            if embedding and a.ndim == 3:
                a = a[0]
            chunks.append(a.astype(np.float32))
        result = np.concatenate(chunks)[inv]
        del chunks
        torch.cuda.empty_cache()
        return result

    def prepare(self):
        import torch
        import joblib
        from tabpfn import TabPFNClassifier
        self.helpers = h = frozen_helpers(self.dataset)
        self.args = json.loads((self.source / 'job.json').read_text())['args']
        self.design = json.loads((self.source / 'design.json').read_text())
        self.names = self.design['class_names']; self.C = len(self.names)
        state = np.load(self.base / 'residual_state.npz')
        self.state = {k: state[k] for k in state.files}
        ids = np.load(self.source / 'shared_context_ids.npz')
        self.exp_ids, self.anchor_ids = ids['expert_pool'], ids['anchor']
        identity = np.load(self.source / 'evaluation_identity.npz')
        self.eval_ids, self.y_eval = identity['ids'], identity['y']
        np.testing.assert_array_equal(self.exp_ids, self.state['train_ids'])
        self.y_exp = self.state['train_labels']
        clean = Path(self.args['clean_manifest']).parent
        assert np.isin(self.exp_ids, np.load(clean / 'train_idx.npy')).all()
        assert not np.intersect1d(self.exp_ids, self.eval_ids).size
        self.progress('loading_frozen_dataset')
        suite = h.core.load_pickle(self.args['data'])
        X, families = suite['X'], np.asarray(suite['families'])
        class_index = {name: i for i, name in enumerate(self.names)}
        def labels(ix):
            return np.asarray([class_index[s] for s in families[ix]], dtype=np.int16)
        self.y_anchor = labels(self.anchor_ids)
        np.testing.assert_array_equal(labels(self.exp_ids), self.y_exp)
        np.testing.assert_array_equal(labels(self.eval_ids), self.y_eval)
        self.X_exp = np.nan_to_num(np.asarray(X[self.exp_ids], dtype=np.float32))
        self.X_anchor = np.nan_to_num(np.asarray(X[self.anchor_ids], dtype=np.float32))
        self.X_eval = np.nan_to_num(np.asarray(X[self.eval_ids], dtype=np.float32))
        # Reconstruct only the original tune IDs, without fitting or tuning anything.
        val = np.load(clean / 'val_idx.npy')
        y_val = labels(val)
        tune, _, _ = h.scenario_stratified_split2(val, y_val, np.asarray(suite['timestamps'])[val],
                                                 np.asarray(suite['attack_scenarios'])[val], 0.4, self.names)
        tune = h.core.cap_per_class(tune, labels(tune), self.C, self.args['tune_cap_per_class'],
                                    SEED + h.SEED_BAND_TUNE_CAP)
        self.tune_ids, self.y_tune = tune, labels(tune)
        self.X_tune = np.nan_to_num(np.asarray(X[tune], dtype=np.float32))
        assert len(tune) == len(np.load(self.source / 'probabilities/global_tune.npy', mmap_mode='r'))
        np.savez(self.out / 'tune_identity.npz', ids=tune, y=self.y_tune)
        h.core._PICKLE_CACHE.clear()
        del suite, X, families, val, y_val
        gc.collect()
        self.hashes, self.anchor_hashes = feature_hash(self.X_exp), feature_hash(self.X_anchor)
        cache = self.out / 'cache'
        if (cache / 'TRAIN_REPRESENTATION_COMPLETE.json').exists():
            self.obs = np.load(cache / 'train_observable.npy')
            self.p0_exp = np.load(cache / 'train_global_corrected.npy')
            self.progress('reuse_saved_train_representation')
            return
        self.progress('restore_fitted_global')
        glob = TabPFNClassifier.load_from_fit_state(self.source / 'global_fit_state.tabpfn_fit', device='cuda')
        # Verify checkpoint replay against the original model before using it to mine.
        # Match the original hash-deduplicated order and batch shape. The PFN
        # auto-precision path can differ for a small diagnostic-only batch.
        _, eval_unique = np.unique(feature_hash(self.X_eval), return_index=True)
        check = eval_unique[:self.args['test_batch_size']]
        self.progress('verify_original_global_batch', rows=len(check))
        raw = np.asarray(glob.predict_proba(self.X_eval[check]), dtype=np.float32)
        expected = np.load(self.source / 'probabilities/global_test.npy', mmap_mode='r')[check]
        difference = float(np.max(np.abs(raw - expected)))
        write(self.out / 'global_probability_replay_check.json', dict(rows=len(check), probability_max_abs=difference,
              mean_abs=float(np.mean(np.abs(raw-expected))), argmax_differences=int((raw.argmax(1)!=expected.argmax(1)).sum()),
              original_batch_order=True))
        np.testing.assert_allclose(raw, expected, rtol=0, atol=1e-5)
        pca = joblib.load(self.base / 'feature_pca.joblib')
        check = eval_unique[:self.args['embed_chunk']]
        emb = np.asarray(glob.get_embeddings(self.X_eval[check], 'test'))
        if emb.ndim == 3:
            emb = emb[0]
        zcheck = pca.transform(emb).astype(np.float32)
        zexpected = np.load(self.base / 'test_z.npy', mmap_mode='r')[check]
        zdiff = float(np.max(np.abs(zcheck-zexpected)))
        np.testing.assert_allclose(zcheck, zexpected, rtol=1e-4, atol=1e-3)
        write(self.out / 'global_replay_check.json', dict(probability_rows=len(expected), embedding_rows=len(check), probability_max_abs=difference,
                                                        pca_feature_max_abs=zdiff, same_fitted_checkpoint=True))
        p_raw = self.infer(glob, self.X_exp, 'global_train_probabilities')
        logits = np.log(np.clip(p_raw, 1e-12, None)) / float(self.state['global_temperature'])
        logits -= logits.max(1, keepdims=True)
        self.p0_exp = np.exp(logits); self.p0_exp /= self.p0_exp.sum(1, keepdims=True)
        z = pca.transform(self.infer(glob, self.X_exp, 'global_train_embeddings', embedding=True,
                                    batch=100000)).astype(np.float32)
        parts = []
        for key, a in [('z', z), ('p', self.p0_exp)]:
            parts.append(((a-self.state[key+'_mean'])/self.state[key+'_std']) / np.sqrt(a.shape[1]))
        self.obs = np.ascontiguousarray(np.concatenate(parts, 1), dtype=np.float32)
        np.save(cache / 'train_observable.npy', self.obs)
        np.save(cache / 'train_global_corrected.npy', self.p0_exp)
        write(cache / 'TRAIN_REPRESENTATION_COMPLETE.json', dict(rows=len(self.exp_ids), dimensions=self.obs.shape[1],
              global_fit_sha256=sha(self.source / 'global_fit_state.tabpfn_fit'), train_ids_sha256=sha(self.base/'residual_state.npz')))
        del glob, p_raw, logits, z, emb
        gc.collect(); torch.cuda.empty_cache()

    def select_contexts(self):
        if (self.out / 'CONTEXTS_COMPLETE.json').exists():
            self.progress('reuse_frozen_context_ids')
            return
        residual = self.helpers.balanced_ce(self.p0_exp, self.y_exp, self.state['w_bal'])
        _, distinct = np.unique(self.hashes, return_index=True)
        composition, audits = [], []
        for k in range(1, 5):
            self.progress('select_context', expert=k)
            reference_ids = np.load(self.source / f'contexts/designed_e{k}.npy')
            reference = np.searchsorted(self.exp_ids, reference_ids)
            np.testing.assert_array_equal(self.exp_ids[reference], reference_ids)
            core, budget = core_subset(reference, self.y_exp, FRACTION, SEED+6000+k)
            excluded = np.unique(np.concatenate([self.hashes[core], self.anchor_hashes]))
            candidates = distinct[~np.isin(self.hashes[distinct], excluded)]
            queries = query_subset(core, self.y_exp, self.hashes, residual, QUERY_LIMIT, SEED+6100+k)
            pairs = nearest_pairs(self.obs, self.y_exp, candidates, queries, budget, SEED+6200+k,
                                  lambda stage, **kw: self.progress(stage, expert=k, **kw))
            near = np.asarray([p[1] for p in pairs], dtype=np.int64)
            random = matched_random(pairs, candidates, self.y_exp, SEED+6300+k)
            q = np.asarray([p[0] for p in pairs], dtype=np.int64)
            np.save(self.out / f'contexts/e{k}_shared_core.npy', self.exp_ids[core])
            np.savez(self.out / f'contexts/e{k}_counterexample_pairs.npz', query_ids=self.exp_ids[q],
                     near_ids=self.exp_ids[near], random_ids=self.exp_ids[random],
                     query_labels=self.y_exp[q], counterpart_labels=self.y_exp[near],
                     near_distance=np.linalg.norm(self.obs[q]-self.obs[near], axis=1),
                     random_distance=np.linalg.norm(self.obs[q]-self.obs[random], axis=1))
            for arm, counter in [('counter_random', random), ('counter_near', near)]:
                selected = np.sort(np.concatenate([core, counter]))
                assert len(selected) == len(reference) and len(np.unique(selected)) == len(selected)
                assert not np.intersect1d(core, counter).size
                assert np.all(self.y_exp[q] != self.y_exp[counter])
                np.save(self.out / f'contexts/{arm}_e{k}.npy', self.exp_ids[selected])
                counts = np.bincount(self.y_exp[selected], minlength=self.C)
                composition.append(dict(arm=arm, expert=k, block_rows=len(selected), core_rows=len(core),
                                        counterpart_rows=len(counter), **dict(zip(self.names, counts))))
            np.testing.assert_array_equal(np.bincount(self.y_exp[near], minlength=self.C),
                                          np.bincount(self.y_exp[random], minlength=self.C))
            audit = dict(expert=k, block_rows=len(reference), shared_core_rows=len(core),
                         counterpart_rows=budget, query_rows=len(queries), candidates=len(candidates),
                         per_class_counts_equal=True, all_pairs_cross_class=True,
                         near_mean_distance=float(np.linalg.norm(self.obs[q]-self.obs[near], axis=1).mean()),
                         random_mean_distance=float(np.linalg.norm(self.obs[q]-self.obs[random], axis=1).mean()),
                         near_counter_global_correct=int((self.p0_exp[near].argmax(1)==self.y_exp[near]).sum()),
                         random_counter_global_correct=int((self.p0_exp[random].argmax(1)==self.y_exp[random]).sum()),
                         counterpart_class_counts=dict(zip(self.names, np.bincount(self.y_exp[near], minlength=self.C))))
            audits.append(audit)
            write(self.out / f'contexts/e{k}_selection.json', audit)
            pd.DataFrame(composition).to_csv(self.out / 'context_compositions.csv', index=False)
        write(self.out / 'CONTEXTS_COMPLETE.json', dict(arms=ARMS, experts=audits,
                                                      selection_pool='D_expert only', test_used=False))

    def fit_predict(self):
        import torch
        from tabpfn import TabPFNClassifier
        for k in range(1, 5):
            for arm in ARMS:
                name = f'{arm}_e{k}'
                done = self.out / f'{name}_COMPLETE.json'
                if done.exists():
                    continue
                ids = np.load(self.out / f'contexts/{name}.npy')
                loc = np.searchsorted(self.exp_ids, ids)
                np.testing.assert_array_equal(self.exp_ids[loc], ids)
                X = np.concatenate([self.X_anchor, self.X_exp[loc]])
                y = np.concatenate([self.y_anchor, self.y_exp[loc]])
                self.progress('fitting_expert', model=name, context_rows=len(y))
                t = time.time()
                clf = TabPFNClassifier(device='auto', model_path=self.args['model_path'],
                    ignore_pretraining_limits=self.args['ignore_pretraining_limits'], random_state=SEED,
                    n_estimators=4, auto_scale_n_estimators=False, fit_mode='fit_with_cache',
                    keep_cache_on_device=False)
                clf.fit(X, y)
                tune = self.infer(clf, self.X_tune, name+'/tune')
                np.save(self.out / f'probabilities/{name}_tune.npy', tune)
                test = self.infer(clf, self.X_eval, name+'/test')
                np.save(self.out / f'probabilities/{name}_test.npy', test)
                np.save(self.out / f'predictions/{name}_raw.npy', test.argmax(1).astype(np.int16))
                write(done, dict(model=name, context_rows=len(y), seconds=time.time()-t,
                                 test_rows=len(test), context_sha256=sha(self.out / f'contexts/{name}.npy')))
                self.progress('expert_complete', model=name, seconds=time.time()-t)
                del clf, X, y, test, tune
                gc.collect(); torch.cuda.empty_cache()
                analyze(self.root, self.dataset, partial=True)

    def run(self):
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(16); torch.set_num_interop_threads(4)
        torch.manual_seed(SEED); np.random.seed(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with threadpool_limits(16):
            self.prepare(); self.select_contexts(); self.fit_predict()
            analyze(self.root, self.dataset)
        self.progress('complete')
        write(self.out / 'COMPLETE.json', dict(dataset=self.dataset, seed=SEED, K=4, arms=ARMS,
              seconds=time.time()-self.started, baseline=str(self.source), additional_models=8))


def analyze(root, dataset, partial=False):
    """Class F1/precision/recall, paired corrections and confusions in fixed A/B."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from analyze_exp59_expert_capability import evaluate
    base = BASE / f'{dataset}_s43'; source = base / 'fresh_bank'; out = root / f'{dataset}_s43'
    design = json.loads((source / 'design.json').read_text()); names = design['class_names']; C = len(names)
    y = np.load(source / 'evaluation_identity.npz')['y']
    g = np.load(source / 'predictions/global_raw.npy')
    r, o = np.load(base / 'residual_test_regions.npy'), np.load(source / 'fixed_regions.npz')['test']
    model_rows, class_rows, view_rows, confusion_rows = [], [], [], []
    matrices = {}
    all_models = [('global', None, g, None)]
    for k in range(1, 5):
        all_models.append((f'designed_e{k}', k, np.load(source/f'predictions/designed_e{k}_raw.npy'), None))
    for arm in ARMS:
        matrices[arm] = []
        for k in range(1, 5):
            name = f'{arm}_e{k}'
            if not (out / f'{name}_COMPLETE.json').exists():
                if not partial: raise RuntimeError('Incomplete '+name)
                continue
            pred = np.load(out / f'predictions/{name}_raw.npy')
            probabilities = np.load(out / f'probabilities/{name}_test.npy', mmap_mode='r')
            np.testing.assert_array_equal(pred, probabilities.argmax(1))
            matrices[arm].append(pred)
            all_models.append((name, k, pred, probabilities))
    for name, k, pred, probabilities in all_models:
        s = scores(confusion(y, pred, C))
        model_rows.append(dict(model=name, scope='full_test', rows=len(y), macro_f1=s['macro_f1'],
                               accuracy=s['accuracy'], fixed=int(((g!=y)&(pred==y)).sum()),
                               harmed=int(((g==y)&(pred!=y)).sum())))
        for c, cl in enumerate(names):
            class_rows.append(dict(model=name, scope='full_test', class_name=cl,
                                   **{key:s[key][c] for key in ['support','TP','FP','FN','precision','recall','f1']}))
        if k is None: continue
        for scope, memberships in [('residual',r), ('observable',o)]:
            v = evaluate(y, g, pred, names, memberships==k-1)
            view_rows.append(dict(model=name, expert=k, scope=scope, rows=v['rows'], fixed=v['fixed'], harmed=v['harmed']))
            for c in v['classes']:
                class_rows.append(dict(model=name, scope=scope, class_name=c['name'], **c['expert_metrics'],
                                       fixed=c['fixed'], harmed=c['harmed'], global_f1=c['global_metrics']['f1'],
                                       global_FP=c['global_metrics']['FP'], new_fp_harm=c['new_fp_harm']))
            for p in v['confusion_pairs']:
                confusion_rows.append(dict(model=name, scope=scope, **p))
    for arm, preds in matrices.items():
        if len(preds)!=4:continue
        pred_matrix = np.stack(preds,1)
        for scope, memberships in [('residual_oracle',r),('input_nearest',o)]:
            pred = pred_matrix[np.arange(len(y)),memberships]
            np.save(out/f'predictions/{arm}_{scope}.npy',pred)
            s=scores(confusion(y,pred,C))
            model_rows.append(dict(model=arm+'_'+scope,scope='full_test',rows=len(y),macro_f1=s['macro_f1'],accuracy=s['accuracy'],
                                   fixed=int(((g!=y)&(pred==y)).sum()),harmed=int(((g==y)&(pred!=y)).sum())))
            for c,cl in enumerate(names):class_rows.append(dict(model=arm+'_'+scope,scope='full_test',class_name=cl,
                **{key:s[key][c] for key in ['support','TP','FP','FN','precision','recall','f1']}))
    for name, path in [('designed_residual_oracle',base/'designed_residual_oracle_pred.npy'),
                       ('designed_input_nearest',source/'predictions/designed_fixed_region.npy')]:
        pred=np.load(path);s=scores(confusion(y,pred,C))
        model_rows.append(dict(model=name,scope='full_test',rows=len(y),macro_f1=s['macro_f1'],accuracy=s['accuracy'],
                               fixed=int(((g!=y)&(pred==y)).sum()),harmed=int(((g==y)&(pred!=y)).sum())))
        for c,cl in enumerate(names):class_rows.append(dict(model=name,scope='full_test',class_name=cl,
            **{key:s[key][c] for key in ['support','TP','FP','FN','precision','recall','f1']}))
    pd.DataFrame(model_rows).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(class_rows).to_csv(out/'class_metrics.csv',index=False)
    pd.DataFrame(view_rows).to_csv(out/'capability_summary.csv',index=False)
    pd.DataFrame(confusion_rows).to_csv(out/'confusion_pairs.csv',index=False)
    write(out/'analysis_status.json',dict(partial=partial,models=len(all_models),updated_epoch=time.time(),
                                         evaluation_memberships='Frozen EXP59 residual and observable IDs'))


def controller(root):
    root.mkdir(parents=True, exist_ok=True)
    completed=[]; started=time.time()
    for dataset in ['cic2018','toniot']:
        out=root/f'{dataset}_s43';out.mkdir(exist_ok=True)
        if (out/'COMPLETE.json').exists():completed.append(dataset);continue
        env={**os.environ,'OMP_NUM_THREADS':'16','MKL_NUM_THREADS':'16','OPENBLAS_NUM_THREADS':'16',
             'NUMEXPR_NUM_THREADS':'16','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONUNBUFFERED':'1',
             'PYTHONFAULTHANDLER':'1','CUDA_VISIBLE_DEVICES':'0'}
        with (out/'worker.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--stage','worker',
                                    '--root',str(root),'--dataset',dataset],stdout=log,stderr=subprocess.STDOUT,env=env)
            write(root/'status.json',dict(state='running',active_job=dataset,controller_pid=os.getpid(),worker_pid=child.pid,
                                          started_epoch=started,completed=completed))
            code=child.wait()
        if code:
            write(root/'status.json',dict(state='needs_recovery',active_job=dataset,exit_code=code,completed=completed))
            raise SystemExit(code)
        completed.append(dataset)
    write(root/'status.json',dict(state='complete',seconds=time.time()-started,completed=completed))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--dataset',choices=['cic2018','toniot'])
    parser.add_argument('--stage',choices=['controller','worker','analyze'],required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.stage=='controller':controller(root)
    elif args.stage=='analyze':analyze(root,args.dataset)
    else:
        try:Worker(root,args.dataset).run()
        except BaseException:
            write(root/f'{args.dataset}_s43/ERROR.json',dict(time=time.time(),traceback=traceback.format_exc()))
            raise
