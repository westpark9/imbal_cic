#!/usr/bin/env python3
"""EXP38: replay an EXP31 frozen context bank and persist full-information data.

Helper definitions below are copied from EXP31 without modifying that record.
No context selection, expert mining, hyperparameter tuning or test-label routing.
The reference global is reconstructed once; all EXP39 policies share its cache.
Partial arrays are atomic and resumable. Exact source-context IDs are retained.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import joblib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nfv3_v3_common as core

def uniq_rows(Xr):
    h = pd.util.hash_pandas_object(pd.DataFrame(Xr), index=False).values
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    return Xr[first], inv

def full_proba(proba, model_classes, n_classes):
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(model_classes, dtype=np.int64)] = proba
    return out

def entropy_of(p):
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1).astype(np.float32)

def margin_of(p):
    if p.shape[1] < 2:
        return np.ones(len(p), dtype=np.float32)
    part = np.partition(p, -2, axis=1)
    return (part[:, -1] - part[:, -2]).astype(np.float32)

class PhiRawPCA:
    def __init__(self, feats_fit, dim, seed):
        self.scaler = RobustScaler().fit(feats_fit)
        self.pca = PCA(n_components=dim, random_state=seed).fit(
            self.scaler.transform(feats_fit))

    def transform(self, feats, chunk=500_000):
        outs = [self.pca.transform(self.scaler.transform(feats[s0:s0 + chunk]))
                .astype(np.float32) for s0 in range(0, len(feats), chunk)]
        return np.concatenate(outs) if outs else \
            np.zeros((0, self.pca.n_components_), dtype=np.float32)

class PhiQuantilePCA:
    def __init__(self, feats_fit, dim, seed):
        self.qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, len(feats_fit)),
            subsample=len(feats_fit), random_state=seed).fit(feats_fit)
        self.pca = PCA(n_components=dim, random_state=seed).fit(
            self.qt.transform(feats_fit))

    def transform(self, feats, chunk=500_000):
        outs = [self.pca.transform(self.qt.transform(feats[s0:s0 + chunk]))
                .astype(np.float32) for s0 in range(0, len(feats), chunk)]
        return np.concatenate(outs) if outs else \
            np.zeros((0, self.pca.n_components_), dtype=np.float32)

class PhiEmbedPCA:
    def __init__(self, feats_fit, dim, seed, embed_fn):
        self.embed_fn = embed_fn
        emb = embed_fn(feats_fit)
        self.pca = PCA(n_components=min(dim, emb.shape[1]),
                       random_state=seed).fit(emb)

    def transform(self, feats, chunk=None):
        return self.pca.transform(self.embed_fn(feats)).astype(np.float32)

class ResidualSignature:
    """Guide §9 residual failure signature, per-block standardized (exp22c)."""

    BLOCKS = ("z", "p", "e", "r")

    def __init__(self, a_p, a_e, a_r):
        self.alpha = {"z": 1.0, "p": a_p, "e": a_e, "r": a_r}
        self.stats = {}

    @staticmethod
    def _blocks(z, p0, y=None, r_bar=None):
        out = {"z": np.asarray(z, dtype=np.float32),
               "p": np.asarray(p0, dtype=np.float32)}
        if y is not None:
            onehot = np.zeros(out["p"].shape, dtype=np.float32)
            onehot[np.arange(len(y)), y] = 1.0
            out["e"] = onehot - out["p"]
            out["r"] = np.log1p(r_bar).astype(np.float32)[:, None]
        return out

    def _std(self, name, arr):
        m, s = self.stats[name]
        return ((arr - m) / s) * np.float32(
            self.alpha[name] / np.sqrt(arr.shape[1]))

    def fit_full(self, z, p0, y, r_bar):
        for name, arr in self._blocks(z, p0, y, r_bar).items():
            m = arr.mean(axis=0).astype(np.float32)
            s = np.maximum(arr.std(axis=0), 1e-6).astype(np.float32)
            self.stats[name] = (m, s)
        return self.full(z, p0, y, r_bar)

    def full(self, z, p0, y, r_bar):
        b = self._blocks(z, p0, y, r_bar)
        return np.concatenate([self._std(n, b[n]) for n in self.BLOCKS], axis=1)

    def observable(self, z, p0):
        b = self._blocks(z, p0)
        return np.concatenate([self._std(n, b[n]) for n in ("z", "p")], axis=1)

    @property
    def obs_dim(self):
        return int(self.stats["z"][0].shape[0] + self.stats["p"][0].shape[0])

def sq_dist_to_centroids(z, mu, chunk=500_000):
    outs = []
    for s0 in range(0, len(z), chunk):
        zb = z[s0:s0 + chunk]
        outs.append(((zb[:, None, :] - mu[None, :, :]) ** 2).sum(-1))
    return np.concatenate(outs).astype(np.float32)

class AffinityRef:
    def __init__(self, z_ref, nn):
        self.nn = min(nn, len(z_ref))
        self.knn = NearestNeighbors(n_neighbors=self.nn).fit(z_ref)

    def score(self, z, chunk=500_000):
        outs = []
        for s0 in range(0, len(z), chunk):
            d, _ = self.knn.kneighbors(z[s0:s0 + chunk])
            outs.append(-d.mean(axis=1).astype(np.float32))
        return np.concatenate(outs)

class PriorCorrector:
    """z~ = log(p_raw+eps) + beta*(log pi_ref - log pi_hat); softmax(z~/T)."""

    def __init__(self, ctx_labels, n_classes, ref_prior, alpha):
        counts = np.bincount(ctx_labels, minlength=n_classes).astype(np.float64)
        self.pi_hat = (counts + alpha) / (counts.sum() + alpha * n_classes)
        self.shift = (np.log(np.clip(ref_prior, 1e-12, None))
                      - np.log(self.pi_hat))

    def correct(self, p_raw, beta, temp):
        z = np.log(np.clip(p_raw, 1e-12, None)) + beta * self.shift[None, :]
        z = z / max(temp, 1e-6)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

def build_pair_pre(p0, z, d_k, a0, qk):
    n = len(p0)
    return np.concatenate(
        [p0.astype(np.float32), entropy_of(p0)[:, None], margin_of(p0)[:, None],
         z, d_k[:, None].astype(np.float32), a0[:, None],
         np.repeat(qk[None, :], n, axis=0)], axis=1)

def build_pair_post(p0, pk, a0, ak, dk, qk):
    n = len(p0)
    return np.concatenate(
        [p0.astype(np.float32), pk.astype(np.float32),
         (pk - p0).astype(np.float32),
         entropy_of(p0)[:, None], entropy_of(pk)[:, None],
         margin_of(p0)[:, None], margin_of(pk)[:, None],
         a0[:, None], ak[:, None], dk[:, None].astype(np.float32),
         np.repeat(qk[None, :], n, axis=0)], axis=1)


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def save_array(path, value):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    print('Args: ' + json.dumps(vars(args), sort_keys=True), flush=True)
    source, cache = Path(args.source_run).resolve(), Path(args.cache_dir).resolve()
    parent = json.loads((source / 'args.json').read_text())
    if parent['test_cap_per_class'] != 0 or parent['target_dataset'] != 'cic2018':
        raise ValueError('First replay requires the uncapped CIC2018 reference.')
    old = dict(np.load(source / 'system_dump.npz', allow_pickle=False))
    ids = dict(np.load(source / 'context_rows.npz', allow_pickle=False))
    K = len(old['qk'])
    required = ['C0', 'anchor', 'phi_fit', 'route', 'cal', 'eval']
    required += [f'expert{k + 1}_block' for k in range(K)]
    if not all(k in ids for k in required):
        raise ValueError('Source context IDs incomplete')
    source_timing = json.loads((source / 'timings.json').read_text())
    if isinstance(source_timing, list):
        source_timing = source_timing[0]
    data_stat = Path(parent['data']).stat()
    expected = source_timing['data_file']
    if (data_stat.st_size, int(data_stat.st_mtime)) != (expected['bytes'], expected['mtime']):
        raise ValueError('Dataset changed since the reference run')
    identity = {
        'schema': 1, 'source_run': str(source),
        'source_args_sha256': sha256(source / 'args.json'),
        'contexts_sha256': sha256(source / 'context_rows.npz'),
        'source_dump_sha256': sha256(source / 'system_dump.npz'),
        'script_sha256': sha256(__file__),
        'model_path': parent['model_path'],
        'model_sha256': sha256(parent['model_path']),
        'data_bytes': data_stat.st_size, 'data_mtime': int(data_stat.st_mtime),
        'batch_size': args.batch_size, 'device': args.device,
    }
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / 'PREPARING.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f'Cache writer already active: pid={pid}')
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    try:
        manifest = cache / 'identity.json'
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError('Cache identity mismatch; use a new cache directory')
        write_json(manifest, identity)
        shutil.copy2(__file__, cache / 'producer_source.py')
        if (cache / 'COMPLETE.json').exists():
            print('Complete frozen cache already exists: ' + str(cache), flush=True)
            return
        _prepare_arrays(args, parent, old, ids, cache, source_timing)
    finally:
        lock.unlink(missing_ok=True)


def _prepare_arrays(args, parent, old, ids, cache, source_timing):
    from tabpfn import TabPFNClassifier
    import torch
    started = time.monotonic()
    names = old['class_names'].astype(str).tolist()
    C, K = len(names), len(old['qk'])
    # Load exactly the parent loader; its class order and held-out IDs are checked.
    load_args = SimpleNamespace(**parent)
    X, loaded_names, train_ids, _, test_ids, _, _, _, label = core.load_cic2018(load_args)
    if list(loaded_names) != names or not np.array_equal(ids['eval'], test_ids):
        raise ValueError('Class ordering or full test IDs differ from the source run')
    data = core.load_pickle(parent['data'])
    timestamp = np.asarray(data['timestamps'], dtype=np.int64)
    scenarios = np.asarray(data['attack_scenarios']).astype(str)
    train_counts = np.bincount(label(train_ids), minlength=C)
    ref = pd.read_csv(Path(args.source_run) / '2c_context_priors.csv')
    ref_prior = ref.loc[ref['context'] == 'ref(D_global)', names].iloc[0].to_numpy(float)
    ref_prior = ref_prior / ref_prior.sum()
    beta, temperature = float(old['prior_beta']), float(old['prior_T'])
    splits = ['route', 'cal', 'eval']

    def feats(idx):
        return np.nan_to_num(np.asarray(X[idx], dtype=np.float32))

    def cached(name, producer):
        path = cache / (name + '.npy')
        if not path.exists():
            t = time.monotonic()
            save_array(path, producer())
            print(f'cached {name}: {time.monotonic() - t:.1f}s', flush=True)
        return np.load(path, mmap_mode='r', allow_pickle=False)

    # Physical rows, not labels, define inference equivalence and overlap checks.
    for split in splits:
        cached(split + '_ids', lambda s=split: ids[s])
        cached(split + '_X', lambda s=split: feats(ids[s]))
        cached(split + '_y', lambda s=split: label(ids[s]).astype(np.int32))
        cached(split + '_time', lambda s=split: timestamp[ids[s]])
        cached(split + '_scenario', lambda s=split: scenarios[ids[s]])
        cached(split + '_hash', lambda s=split: pd.util.hash_pandas_object(
            pd.DataFrame(np.load(cache / (s + '_X.npy'), mmap_mode='r')),
            index=False).to_numpy())
    route_hash = np.load(cache / 'route_hash.npy', mmap_mode='r')
    cached('cal_mask', lambda: ~np.isin(
        np.load(cache / 'cal_hash.npy'), np.unique(route_hash)))
    # Coverage audit reports known scenario counts, without assuming missing modes.
    composition = []
    for key in ['C0', 'anchor'] + [f'expert{k + 1}_block' for k in range(K)]:
        idx = ids[key]
        frame = pd.DataFrame({'class': np.asarray(names)[label(idx)],
                              'scenario': scenarios[idx]})
        rows = frame.value_counts().rename('rows').reset_index()
        rows.insert(0, 'context', key); composition.append(rows)
    pd.concat(composition).to_csv(cache / 'context_scenario_counts.csv', index=False)

    def make_clf(idx):
        clf = TabPFNClassifier(
            device=args.device, model_path=parent['model_path'],
            random_state=parent['seed'], n_estimators=parent['n_estimators'],
            auto_scale_n_estimators=False, fit_mode=parent['fit_mode'],
            keep_cache_on_device=parent['keep_cache_on_device'],
            ignore_pretraining_limits=parent['ignore_pretraining_limits'])
        clf.fit(feats(idx), label(idx))
        return clf

    def predict(clf, arr, embed=False):
        unique, inverse = uniq_rows(arr)
        outputs = []
        batch = parent['embed_chunk'] if embed else args.batch_size
        for begin in range(0, len(unique), batch):
            block = unique[begin:begin + batch]
            if embed:
                value = np.asarray(clf.get_embeddings(block, 'test'))
                if value.ndim == 3:
                    value = value[0]
            else:
                value = full_proba(clf.predict_proba(block), clf.classes_, C)
            outputs.append(value.astype(np.float32))
            print(f'  {"embed" if embed else "predict"}: '
                  f'{min(begin + batch, len(unique)):,}/{len(unique):,} unique', flush=True)
        out = np.concatenate(outputs)[inverse]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    # Replay global once; all policy variants use these exact resulting arrays.
    global_model = make_clf(ids['C0'])
    corr0 = PriorCorrector(label(ids['C0']), C, ref_prior, parent['prior_alpha'])
    for split in splits:
        cached(split + '_p0', lambda s=split: corr0.correct(predict(
            global_model, np.load(cache / (s + '_X.npy'), mmap_mode='r')),
            0., temperature).astype(np.float32))
    pca_path = cache / 'phi_pca.joblib'
    if pca_path.exists():
        pca = joblib.load(pca_path)
    else:
        if parent['phi_mode'] != 'embed_pca':
            raise ValueError('This first frozen replay expects embed_pca')
        emb = predict(global_model, feats(ids['phi_fit']), embed=True)
        pca = PCA(n_components=parent['phi_dim'],
                  random_state=parent['seed'] + 1250).fit(emb)
        joblib.dump(pca, pca_path)
        del emb
    for split in splits:
        cached(split + '_z', lambda s=split: pca.transform(predict(
            global_model, np.load(cache / (s + '_X.npy'), mmap_mode='r'),
            embed=True)).astype(np.float32))
    rng = np.random.default_rng(parent['seed'] + 1600)
    references = [ids['C0'][rng.permutation(len(ids['C0']))[:parent['affinity_ref_rows']]]]
    for k in range(K):
        block = ids[f'expert{k + 1}_block']
        references.append(block[rng.permutation(len(block))[:parent['affinity_ref_rows']]])
    affinities = []
    for k, reference in enumerate(references):
        zr = cached(f'affinity_ref_{k}', lambda idx=reference: pca.transform(
            predict(global_model, feats(idx), embed=True)).astype(np.float32))
        affinities.append(AffinityRef(zr, parent['affinity_nn']))
    del global_model
    gc.collect(); torch.cuda.empty_cache()

    # Frozen source centroids/statistics are descriptors, never recomputed with test labels.
    zdim = parent['phi_dim']
    sig = ResidualSignature(*old['sig_alpha'][1:].tolist())
    offset = 0
    for name, dim in [('z', zdim), ('p', C), ('e', C), ('r', 1)]:
        sig.stats[name] = (old['sig_mean'][offset:offset + dim],
                           old['sig_std'][offset:offset + dim])
        offset += dim
    for split in splits:
        z = np.load(cache / (split + '_z.npy'), mmap_mode='r')
        p0 = np.load(cache / (split + '_p0.npy'), mmap_mode='r')
        cached(split + '_distance', lambda: np.sqrt(sq_dist_to_centroids(
            sig.observable(z, p0), old['mu'][:, :zdim + C])))
        for k, affinity in enumerate(affinities):
            def score_affinity(a=affinity):
                unique, inverse = uniq_rows(z)
                return a.score(unique)[inverse]
            cached(f'{split}_affinity_{k}', score_affinity)
    for k in range(K):
        if all((cache / f'{s}_p{k + 1}.npy').exists() for s in splits):
            continue
        # Same order as EXP31: anchor rows followed by selected residual block.
        idx = np.concatenate([ids['anchor'], ids[f'expert{k + 1}_block']])
        print(f'expert {k + 1}/{K}: {len(idx):,} context rows', flush=True)
        clf = make_clf(idx)
        corr = PriorCorrector(label(idx), C, ref_prior, parent['prior_alpha'])
        for split in splits:
            cached(f'{split}_p{k + 1}', lambda s=split: corr.correct(predict(
                clf, np.load(cache / (s + '_X.npy'), mmap_mode='r')),
                beta, temperature).astype(np.float32))
        del clf
        gc.collect(); torch.cuda.empty_cache()
    save_array(cache / 'qk.npy', old['qk'])
    # Audit reconstruction variance explicitly instead of calling it bit-identical.
    ytest = np.load(cache / 'eval_y.npy')
    if not np.array_equal(ytest, old['y_true']):
        raise ValueError('Test labels no longer match the stored row IDs')
    new_global = np.load(cache / 'eval_p0.npy', mmap_mode='r').argmax(1)
    from sklearn.metrics import f1_score
    audit = {'changed_labels_vs_source_global': int((new_global != old['y_glob']).sum()),
             'source_global_macro_f1': float(f1_score(ytest, old['y_glob'], average='macro')),
             'replay_global_macro_f1': float(f1_score(ytest, new_global, average='macro'))}
    expected_cal = len(ids['cal']) - source_timing['cal_rows_route_dup']
    actual_cal = int(np.load(cache / 'cal_mask.npy').sum())
    if actual_cal != expected_cal:
        raise ValueError(f'Calibration hash-mask mismatch: {actual_cal} != {expected_cal}')
    write_json(cache / 'reconstruction_audit.json', audit)
    metadata = {
        'class_names': names, 'tail_classes': ['bot', 'infiltration', 'web_attacks'],
        'protected_classes': ['brute_force', 'ddos', 'dos'],
        'train_counts': train_counts.tolist(), 'n_experts': K,
        'source_run': str(Path(args.source_run).resolve()),
        'cal_mask_rows': actual_cal, 'full_test_rows': len(ids['eval']),
        'seconds': time.monotonic() - started,
        'reconstruction': audit, 'test_role': 'previously-used development holdout',
        'source_config': parent,
    }
    # Complete only after every required array is present and row-aligned.
    for split in splits:
        for stem in ['X', 'y', 'z', 'distance'] + [f'p{k}' for k in range(K + 1)]:
            arr = np.load(cache / f'{split}_{stem}.npy', mmap_mode='r')
            if len(arr) != len(ids[split]):
                raise ValueError(f'Cache row mismatch: {split}/{stem}')
    write_json(cache / 'COMPLETE.json', metadata)
    print('CACHE COMPLETE: ' + json.dumps(metadata, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=500000)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    with threadpool_limits(args.threads):
        prepare(args)


if __name__ == '__main__':
    main()
