"""Proposal B: grouped-OOF pair selection and bounded log-odds residuals.

Every final PFN retains the supplied labels relevant to its task. No extra
validation labels, neural selector, test priors, or query-dependent retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
import json
import time
import warnings
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
if __package__:
    from .common import FrozenTabPFN, aligned, close, labels, matrix, row_groups
else:
    from common import FrozenTabPFN, aligned, close, labels, matrix, row_groups


@dataclass
class PairConfig:
    max_experts: int = 3
    folds: int = 3
    minimum_groups: int = 3
    alphas: tuple[float, ...] = (.25, .5)
    caps: tuple[float, ...] = (1., 2.)
    minimum_improvement: float = 1e-6
    allowed_benign_fp_increase: int = 0
    require_fold_improvement: bool = True
    batch_size: int = 1024
    seed: int = 43

    def validate(self):
        if self.max_experts < 0 or self.folds < 2 or self.minimum_groups < 3 or self.batch_size < 1:
            raise ValueError('Invalid expert/fold/group/batch budget')
        if not self.alphas or any(a <= 0 or a > 1 for a in self.alphas):
            raise ValueError('Correction strengths must be in (0,1]')
        if not self.caps or any(b <= 0 for b in self.caps):
            raise ValueError('Correction caps must be positive')
        if self.allowed_benign_fp_increase < 0 or self.minimum_improvement < 0:
            raise ValueError('Invalid selection thresholds')


def correct_pair(scores, probability_a, a, b, alpha, cap):
    """Change only two logits; other normalized probabilities may change."""
    if not 0 <= alpha <= 1 or cap <= 0 or a == b:
        raise ValueError('Invalid pair correction')
    p = np.clip(np.asarray(probability_a, dtype=np.float64), 1e-9, 1-1e-9)
    if p.shape != (len(scores),) or not np.isfinite(p).all():
        raise ValueError('Invalid pair posterior')
    out = np.array(scores, dtype=np.float64, copy=True)
    target = np.log(p)-np.log1p(-p)
    delta = alpha*np.clip(target-(out[:, a]-out[:, b]), -cap, cap)
    out[:, a] += .5*delta
    out[:, b] -= .5*delta
    return out


def balanced_nll(scores, y):
    p = softmax(scores, axis=1)
    return float(np.mean([-np.log(np.maximum(p[y == c, c], 1e-12)).mean() for c in np.unique(y)]))


def benign_fp(scores, y, benign):
    return int(np.sum((y == benign) & (scores.argmax(1) != benign)))


class PairResidualClassifier:
    def __init__(self, estimator_factory, benign_class=0, config=None):
        self.estimator_factory = estimator_factory
        self.benign_class = benign_class
        self.config = config or PairConfig()

    def _fit_predict(self, X, y, query, classes):
        model = self.estimator_factory()
        try:
            model.fit(X, y)
            p = aligned(model, query, classes, self.config.batch_size)
            if hasattr(model, 'verify_frozen'):
                self.audit_['backbone_hashes'].append(model.verify_frozen())
            return p
        finally:
            close(model)

    def fit(self, X, y):
        self.config.validate()
        X = matrix(X)
        y_original = labels(y, len(X))
        self.classes_, y = np.unique(y_original, return_inverse=True)
        if self.benign_class not in self.classes_:
            raise ValueError('The declared benign class must be present')
        benign = int(np.flatnonzero(self.classes_ == self.benign_class)[0])
        self.n_features_in_ = X.shape[1]
        cfg, K = self.config, len(self.classes_)
        started = time.perf_counter()
        groups = row_groups(X)
        group_counts = [len(np.unique(groups[y == c])) for c in range(K)]
        self.audit_ = {'label_rows': len(X), 'label_counts': np.bincount(y, minlength=K).tolist(),
            'group_counts': group_counts, 'evaluation_labels_used': False,
            'additional_validation_labels': 0, 'unlabelled_pool_used': False,
            'backbone_hashes': [], 'folds': [], 'selection': [], 'config': asdict(cfg)}
        self.steps_, self.experts_ = [], []
        candidates = [pair for pair in combinations(range(K), 2)
                      if all(group_counts[c] >= cfg.minimum_groups for c in pair)]
        self.audit_['eligible_pairs'] = [list(p) for p in candidates]
        self.fold_ids_ = np.full(len(X), -1, dtype=int)
        self.oof_probabilities_ = None
        can_crossfit = len(np.unique(groups)) >= 2 and bool(candidates) and cfg.max_experts > 0
        if can_crossfit:
            splits = min(cfg.folds, 2 if min(np.bincount(y)) <= 3 else cfg.folds,
                         len(np.unique(groups)), int(np.bincount(y).min()))
            can_crossfit = splits >= 2
        if can_crossfit:
            splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=cfg.seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                folds = list(splitter.split(X, y, groups))
            valid_global = True
            for k, (tr, va) in enumerate(folds):
                if np.intersect1d(groups[tr], groups[va]).size:
                    raise RuntimeError('Exact feature group crossed OOF folds')
                self.fold_ids_[va] = k
                missing = sorted(set(range(K))-set(y[tr].tolist()))
                valid_global &= not missing and bool(len(va))
                self.audit_['folds'].append({'fold': k, 'train_indices': tr.tolist(),
                    'validation_indices': va.tolist(), 'missing_training_classes': missing})
            if valid_global:
                self._select(X, y, groups, folds, candidates, benign, K)
            else:
                self.audit_['fallback_reason'] = 'missing_class_in_global_fold; no reliable OOF correction'
        else:
            self.audit_['fallback_reason'] = 'insufficient_independent_groups_or_zero_expert_budget'

        # Full contexts are fitted only after the structure/coefficients are fixed.
        self.global_ = self.estimator_factory().fit(X, y_original)
        for step in self.steps_:
            a, b = step['pair']
            rows = np.isin(y, [a, b])
            expert = self.estimator_factory().fit(X[rows], y_original[rows])
            self.experts_.append(expert)
        self.audit_['selected_steps'] = self.steps_
        self.audit_['global_context_rows'] = len(X)
        self.audit_['expert_context_rows'] = [int(np.isin(y, s['pair']).sum()) for s in self.steps_]
        self.audit_['fit_seconds'] = time.perf_counter()-started
        self.audit_['backbone_hashes'] = sorted(set(self.audit_['backbone_hashes']))
        return self

    def _select(self, X, y, groups, folds, candidates, benign, K):
        cfg = self.config
        global_p = np.zeros((len(X), K))
        for tr, va in folds:
            global_p[va] = self._fit_predict(X[tr], self.classes_[y[tr]], X[va], self.classes_)
        self.oof_probabilities_ = global_p.copy()
        scores = np.log(np.maximum(global_p, 1e-12))
        base_fp = benign_fp(scores, y, benign)
        cached = {}
        for stage in range(cfg.max_experts):
            pred = scores.argmax(1)
            def confusion(pair):
                a, b = pair
                return float(np.mean(pred[y == a] == b)+np.mean(pred[y == b] == a))
            ranked = sorted(candidates, key=lambda p: (-confusion(p), p))
            before = balanced_nll(scores, y)
            chosen = None
            for a, b in ranked:
                if confusion((a, b)) <= 0:
                    continue
                if (a, b) not in cached:
                    pair_p = np.empty(len(X))
                    valid = True
                    for tr, va in folds:
                        selected = tr[np.isin(y[tr], [a, b])]
                        if len(np.unique(y[selected])) != 2:
                            valid = False
                            break
                        pair_p[va] = self._fit_predict(X[selected], self.classes_[y[selected]],
                                                     X[va], self.classes_[[a, b]])[:, 0]
                    if not valid:
                        continue
                    cached[(a, b)] = pair_p
                best = None
                for alpha in cfg.alphas:
                    for cap in cfg.caps:
                        changed = correct_pair(scores, cached[(a, b)], a, b, alpha, cap)
                        gain = before-balanced_nll(changed, y)
                        fp = benign_fp(changed, y, benign)
                        fold_gain = [balanced_nll(scores[va], y[va])-balanced_nll(changed[va], y[va])
                                     for _, va in folds]
                        safe = fp <= base_fp+cfg.allowed_benign_fp_increase
                        stable = not cfg.require_fold_improvement or min(fold_gain) > 0
                        if gain > cfg.minimum_improvement and safe and stable:
                            if best is None or gain > best[0]:
                                best = (gain, alpha, cap, changed, fp, fold_gain)
                self.audit_['selection'].append({'stage': stage, 'pair': [a, b],
                    'confusion_score': confusion((a, b)), 'accepted': best is not None})
                if best is not None:
                    gain, alpha, cap, scores, fp, fold_gain = best
                    chosen = {'pair': [a, b], 'class_ids': self.classes_[[a, b]].tolist(),
                        'alpha': alpha, 'cap': cap, 'oof_nll_gain_selection_only': gain,
                        'oof_benign_false_positives': fp, 'fold_nll_gains_selection_only': fold_gain}
                    break  # highest-ranked pair that satisfies all safeguards
            if chosen is None:
                self.audit_['stop_reason'] = 'no_candidate_met_selection_constraints'
                break
            self.steps_.append(chosen)
            candidates.remove(tuple(chosen['pair']))  # do not duplicate an identical specialist
        self.audit_['oof_results_are_selection_only'] = True

    def predict_proba(self, X):
        X = matrix(X, features=self.n_features_in_, allow_empty=True)
        p = aligned(self.global_, X, self.classes_, self.config.batch_size)
        if not self.steps_ or not len(X):
            return p  # exact unmodified Global probabilities on fallback
        scores = np.log(np.maximum(p, 1e-12))
        for step, expert in zip(self.steps_, self.experts_):
            a, b = step['pair']
            pair_p = aligned(expert, X, self.classes_[[a, b]], self.config.batch_size)[:, 0]
            scores = correct_pair(scores, pair_p, a, b, step['alpha'], step['cap'])
        return softmax(scores, axis=1)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def verify_frozen(self):
        return [model.verify_frozen() for model in [self.global_, *self.experts_]
                if hasattr(model, 'verify_frozen')]

    def close(self):
        for model in [getattr(self, 'global_', None), *getattr(self, 'experts_', [])]:
            if model is not None:
                close(model)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.verify_frozen()
        self.global_.save(directory/'global.tabpfn_fit')
        for i, model in enumerate(self.experts_):
            model.save(directory/f'expert_{i}.tabpfn_fit')
        (directory/'structure.json').write_text(json.dumps({'format': 1, 'classes': self.classes_.tolist(),
            'benign_class': int(self.benign_class), 'n_features': self.n_features_in_,
            'config': asdict(self.config), 'steps': self.steps_, 'audit': self.audit_}, indent=2)+'\n')

    @classmethod
    def load(cls, directory, device='cpu'):
        directory = Path(directory)
        data = json.loads((directory/'structure.json').read_text())
        if data['format'] != 1:
            raise ValueError('Unsupported pair export version')
        obj = cls(None, data['benign_class'], PairConfig(**data['config']))
        obj.classes_, obj.n_features_in_ = np.array(data['classes']), data['n_features']
        obj.steps_, obj.audit_ = data['steps'], data['audit']
        obj.global_ = FrozenTabPFN.load(directory/'global.tabpfn_fit', device)
        obj.experts_ = [FrozenTabPFN.load(directory/f'expert_{i}.tabpfn_fit', device)
                        for i in range(len(obj.steps_))]
        return obj
