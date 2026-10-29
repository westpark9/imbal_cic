"""Shared contracts for the two EXP91 prototypes (no evaluation labels in fit)."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
from pathlib import Path
import json
import numpy as np


def matrix(X, *, features=None, allow_empty=False):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or not X.shape[1] or (not allow_empty and not len(X)):
        raise ValueError('Expected a nonempty two-dimensional feature matrix')
    if features is not None and X.shape[1] != features:
        raise ValueError('Feature count changed')
    if np.isinf(X).any():
        raise ValueError('Infinite features must be handled explicitly before fitting')
    return X


def labels(y, rows):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != rows:
        raise ValueError('Labels must be one-dimensional and match X')
    if y.dtype.kind not in 'iu' or np.any(y < 0):
        raise ValueError('Use nonnegative integer class IDs')
    if len(np.unique(y)) < 2:
        raise ValueError('At least two observed classes are required')
    return y.astype(np.int64)


def row_groups(X):
    """Exact float32 feature groups; canonicalize NaNs and signed zeros."""
    clean = np.ascontiguousarray(X, dtype=np.float32).copy()
    clean[clean == 0] = 0.
    clean[np.isnan(clean)] = np.nan
    return np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in clean])


def reject_overlap(X, U, name='L/U'):
    overlap = np.intersect1d(row_groups(X), row_groups(U))
    if len(overlap):
        raise ValueError(f'{name}: {len(overlap)} exact feature groups overlap')


def probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Invalid predicted probabilities')
    total = p.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError('Probability row has zero mass')
    return p / total


def aligned(model, X, classes, batch_size=1024):
    if not len(X):
        return np.empty((0, len(classes)), dtype=np.float64)
    raw = np.concatenate([probabilities(model.predict_proba(X[i:i+batch_size]))
                          for i in range(0, len(X), batch_size)])
    out = np.zeros((len(X), len(classes)), dtype=np.float64)
    columns = {int(c): i for i, c in enumerate(classes)}
    if raw.shape[1] != len(model.classes_):
        raise ValueError('Estimator class/probability dimensions disagree')
    for j, c in enumerate(model.classes_):
        if int(c) not in columns:
            raise ValueError('Estimator produced an unknown class')
        out[:, columns[int(c)]] = raw[:, j]
    return probabilities(out)


def close(model):
    if hasattr(model, 'close'):
        model.close()


def state_digest(modules):
    h = hashlib.sha256()
    for module in modules:
        for name, value in sorted(module.state_dict().items()):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class TabPFNFactory:
    checkpoint: str
    device: str = 'cuda:0'
    n_estimators: int = 4
    seed: int = 43
    categorical_columns: tuple[int, ...] = ()

    def __call__(self):
        return FrozenTabPFN(self)


class FrozenTabPFN:
    """Real TabPFN adapter. fit changes context/cache, never optimizer weights."""
    def __init__(self, factory):
        self.factory = factory

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier
        checkpoint = Path(self.factory.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.model = TabPFNClassifier(
            model_path=str(checkpoint), device=self.factory.device,
            random_state=self.factory.seed, n_estimators=self.factory.n_estimators,
            auto_scale_n_estimators=False, fit_mode='fit_with_cache',
            keep_cache_on_device=False, n_preprocessing_jobs=1,
            categorical_features_indices=list(self.factory.categorical_columns) or None,
            inference_config={'SUBSAMPLE_SAMPLES': None})
        self.model.fit(X, y)
        self.classes_ = np.asarray(self.model.classes_)
        self.weight_sha256_ = state_digest(self.model.models_)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def verify_frozen(self):
        if state_digest(self.model.models_) != self.weight_sha256_:
            raise RuntimeError('Frozen TabPFN backbone weights changed')
        return self.weight_sha256_

    def save(self, path):
        self.verify_frozen()
        self.model.save_fit_state(Path(path))
        Path(str(path)+'.json').write_text(json.dumps({'weight_sha256': self.weight_sha256_}))

    @classmethod
    def load(cls, path, device='cpu'):
        from tabpfn import TabPFNClassifier
        obj = cls(None)
        obj.model = TabPFNClassifier.load_from_fit_state(Path(path), device=device)
        obj.classes_ = np.asarray(obj.model.classes_)
        obj.weight_sha256_ = json.loads(Path(str(path)+'.json').read_text())['weight_sha256']
        obj.verify_frozen()
        return obj

    def close(self):
        if hasattr(self, 'model'):
            if hasattr(self, 'weight_sha256_'):
                self.verify_frozen()
            del self.model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class FlowPreprocessor:
    """Train-only numeric scaling, explicit categorical one-hot and missingness.

    Category blocks include missing and unknown slots. The decoder uses block
    cross-entropy; category codes are never treated as continuous distances.
    """
    def __init__(self, categorical_columns=()):
        self.categorical_columns = tuple(categorical_columns)

    def fit(self, X):
        X = matrix(X)
        self.n_features_in_ = X.shape[1]
        if len(set(self.categorical_columns)) != len(self.categorical_columns):
            raise ValueError('Duplicate categorical column')
        if any(c < 0 or c >= X.shape[1] for c in self.categorical_columns):
            raise ValueError('Categorical column outside feature matrix')
        self.numeric = [c for c in range(X.shape[1]) if c not in self.categorical_columns]
        self.medians, self.scales, self.categories = {}, {}, {}
        self.blocks, cursor = [], 0
        for c in range(X.shape[1]):
            values = X[np.isfinite(X[:, c]), c]
            if c in self.categorical_columns:
                self.categories[c] = np.unique(values)
                width = len(self.categories[c]) + 2
                kind = 'category'
            else:
                med = float(np.median(values)) if len(values) else 0.
                scale = float(np.std(values.astype(np.float64))) if len(values) else 1.
                self.medians[c], self.scales[c] = med, max(scale, 1e-6)
                width, kind = 1, 'numeric'
            self.blocks.append((c, cursor, cursor+width, kind))
            cursor += width
        self.value_dim = cursor
        self.output_dim = cursor + X.shape[1]
        return self

    def transform(self, X):
        X = matrix(X, features=self.n_features_in_, allow_empty=True)
        out = np.zeros((len(X), self.output_dim), dtype=np.float32)
        observed = np.isfinite(X)
        for c, start, stop, kind in self.blocks:
            if kind == 'numeric':
                value = np.where(observed[:, c], X[:, c], self.medians[c])
                out[:, start] = np.clip((value-self.medians[c])/self.scales[c], -20, 20)
            else:
                cats = self.categories[c]
                idx = np.full(len(X), len(cats)+1, dtype=int)  # unknown
                idx[~observed[:, c]] = len(cats)  # missing
                for j, value in enumerate(cats):
                    idx[X[:, c] == value] = j
                out[np.arange(len(X)), start+idx] = 1.
        out[:, self.value_dim:] = ~observed
        return out, observed
