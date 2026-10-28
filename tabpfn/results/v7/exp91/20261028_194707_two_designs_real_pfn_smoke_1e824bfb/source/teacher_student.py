"""Proposal A: frozen PFN teacher, soft assignment, trainable prototype student.

This is the report's own numerical-flow adaptation, not LotBoNC/BETA code.
The deployed estimator contains neither the teacher nor the reconstruction head.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import (FlowPreprocessor, aligned, close, labels, matrix,
                    probabilities, reject_overlap, state_digest)


@dataclass
class StudentConfig:
    hidden: int = 128
    embedding: int = 64
    warmup_epochs: int = 10
    rounds: int = 5
    epochs_per_round: int = 3
    batch_size: int = 256
    predict_batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mask_probability: float = .15
    temperature: float = .2
    teacher_strength: float = 1.
    geometry_strength: float = 1.
    assignment_entropy: float = .2
    prior_strength: float = .1
    assignment_steps: int = 100
    unlabelled_strength: float = 1.
    reconstruction_strength: float = .2
    confidence_floor: float = 0.
    prototype_mix: float = .5
    prior_mix: float = .25
    prior_momentum: float = .9
    prior_max_ratio: float = 2.
    minimum_pseudo_mass: float = 1.
    seed: int = 43
    device: str = 'cuda:0'
    categorical_columns: tuple[int, ...] = ()

    def validate(self):
        for name in ('hidden', 'embedding', 'batch_size', 'predict_batch_size', 'assignment_steps'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.warmup_epochs < 1 or self.rounds < 0 or self.epochs_per_round < 1:
            raise ValueError('Invalid epoch/round counts')
        for name in ('temperature', 'assignment_entropy', 'learning_rate'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('teacher_strength', 'geometry_strength', 'prior_strength',
                     'unlabelled_strength', 'reconstruction_strength', 'weight_decay',
                     'minimum_pseudo_mass'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} cannot be negative')
        for name in ('mask_probability', 'confidence_floor', 'prior_momentum'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if not 0 <= self.prototype_mix <= .5 or not 0 <= self.prior_mix <= .5:
            raise ValueError('Prototype and prior mixing must preserve the >=50% anchor')
        if self.prior_max_ratio < 1:
            raise ValueError('prior_max_ratio must be >=1')


class Encoder(nn.Module):
    def __init__(self, inputs, hidden, embedding):
        super().__init__()
        self.entry = nn.Linear(inputs, hidden)
        self.block = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.output = nn.Linear(hidden, embedding)

    def forward(self, x):
        h = F.gelu(self.entry(x))
        return F.normalize(self.output(h+self.block(h)), dim=1, eps=1e-8)


def soft_assignment(cost, prior, entropy=.2, strength=.1, steps=100):
    """Log-domain unbalanced Sinkhorn: fixed row mass, soft column KL.

    Minimize mean(Q*C) + entropy*mean(Q*log Q)
      + strength*KL(mean_rows(Q), prior), subject to Q 1 = 1.
    All tensors are detached; pseudo-label assignment is not backpropagated.
    """
    if entropy <= 0 or strength < 0 or steps < 1:
        raise ValueError('Invalid assignment coefficients')
    if cost.ndim != 2 or prior.shape != (cost.shape[1],) or not len(cost):
        raise ValueError('Assignment dimensions disagree')
    if not torch.isfinite(cost).all() or not torch.isfinite(prior).all() or torch.any(prior <= 0):
        raise ValueError('Assignment cost/prior must be finite with positive prior')
    with torch.no_grad():
        prior = prior/prior.sum()
        kernel = -cost/entropy
        v = torch.zeros_like(prior)
        exponent = strength/(strength+entropy)
        target = torch.log(prior)+np.log(len(cost))
        for _ in range(steps):
            u = -torch.logsumexp(kernel+v, dim=1)
            new_v = exponent*(target-torch.logsumexp(kernel+u[:, None], dim=0))
            if torch.max(torch.abs(new_v-v)).item() < 1e-6:
                v = new_v
                break
            v = new_v
        return torch.softmax(kernel+v, dim=1)


def confidence_weights(teacher, student, floor=0.):
    t, s = teacher.clamp_min(1e-9), student.clamp_min(1e-9)
    m = .5*(t+s)
    entropy = -(t*t.log()).sum(1)/np.log(t.shape[1])
    js = .5*((t*(t.log()-m.log())).sum(1)+(s*(s.log()-m.log())).sum(1))
    return ((1-entropy)*(1-js/np.log(2))).clamp(min=floor, max=1.)


class TeacherStudentClassifier:
    def __init__(self, teacher_factory, config=None):
        self.teacher_factory = teacher_factory
        self.config = config or StudentConfig()

    def _embedding(self, x):
        self.encoder_.eval()
        with torch.no_grad():
            return torch.cat([self.encoder_(x[i:i+self.config.predict_batch_size])
                              for i in range(0, len(x), self.config.predict_batch_size)])

    def _logits(self, z):
        return -torch.cdist(z, self.prototypes_).square()/self.config.temperature

    def _anchors(self, x, y):
        z = self._embedding(x)
        return F.normalize(torch.stack([z[y == c].mean(0) for c in range(len(self.classes_))]), dim=1)

    def _mask_loss(self, x, observed):
        corrupted = x.clone()
        selected = (torch.rand(observed.shape, device=x.device) < self.config.mask_probability) & observed
        for c, start, stop, _ in self.preprocessor_.blocks:
            corrupted[selected[:, c], start:stop] = 0.
            corrupted[selected[:, c], self.preprocessor_.value_dim+c] = 1.
        decoded = self.decoder_(self.encoder_(corrupted))
        losses = []
        for c, start, stop, kind in self.preprocessor_.blocks:
            keep = selected[:, c]
            if keep.any():
                if kind == 'numeric':
                    losses.append(F.huber_loss(decoded[keep, start], x[keep, start]))
                else:
                    losses.append(F.cross_entropy(decoded[keep, start:stop], x[keep, start:stop].argmax(1)))
        return torch.stack(losses).mean() if losses else decoded.sum()*0.

    def fit(self, X, y, U):
        self.config.validate()
        X = matrix(X)
        U = matrix(U, features=X.shape[1], allow_empty=True)
        y = labels(y, len(X))
        reject_overlap(X, U)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        cfg = self.config
        device = torch.device(cfg.device)
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg.seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(cfg.seed)
            return self._fit(X, encoded, U, device)

    def _fit(self, X, y, U, device):
        started = time.perf_counter()
        cfg, K = self.config, len(self.classes_)
        self.history_, self.audit_ = [], {'label_rows': len(X), 'unlabelled_rows': len(U),
            'label_counts': np.bincount(y, minlength=K).tolist(), 'evaluation_labels_used': False,
            'teacher_context_rows': len(X), 'prediction_model': 'student_encoder_and_prototypes',
            'categorical_columns': list(cfg.categorical_columns)}
        teacher_started = time.perf_counter()
        teacher = self.teacher_factory()
        try:
            teacher.fit(X, self.classes_[y])
            pt = aligned(teacher, U, self.classes_, cfg.predict_batch_size)
            if hasattr(teacher, 'verify_frozen'):
                self.audit_['teacher_weights_sha256'] = teacher.verify_frozen()
        finally:
            close(teacher)
        del teacher
        self.audit_['teacher_seconds'] = time.perf_counter()-teacher_started
        self.preprocessor_ = FlowPreprocessor(cfg.categorical_columns).fit(np.concatenate([X, U]))
        xl, ol = self.preprocessor_.transform(X)
        xu, ou = self.preprocessor_.transform(U)
        xl, xu = [torch.tensor(a, device=device) for a in (xl, xu)]
        ol, ou = [torch.tensor(a, device=device) for a in (ol, ou)]
        yl = torch.tensor(y, dtype=torch.long, device=device)
        pt = torch.tensor(pt, dtype=torch.float32, device=device)
        self.encoder_ = Encoder(self.preprocessor_.output_dim, cfg.hidden, cfg.embedding).to(device)
        self.decoder_ = nn.Linear(cfg.embedding, self.preprocessor_.value_dim).to(device)
        initial = state_digest([self.encoder_])
        optimizer = torch.optim.AdamW([*self.encoder_.parameters(), *self.decoder_.parameters()],
                                     lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.prototypes_ = self._anchors(xl, yl)
        self.history_.append({'phase': 'initialized', 'encoder_sha256': initial})
        counts = torch.bincount(yl, minlength=K)
        supervision_weights = 1/counts[yl].float()/K

        def epoch(q=None, w=None, ramp=0.):
            self.encoder_.train()
            losses = []
            order = torch.randperm(len(xu), device=device)
            batches = [order[i:i+cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
            if not batches:
                batches = [order]
            for idx in batches:
                optimizer.zero_grad(set_to_none=True)
                sup = (F.cross_entropy(self._logits(self.encoder_(xl)), yl, reduction='none')*supervision_weights).sum()
                unsup = sup*0.
                if q is not None and len(idx):
                    loss = -(q[idx]*F.log_softmax(self._logits(self.encoder_(xu[idx])), dim=1)).sum(1)
                    unsup = (w[idx]*loss).sum()/w[idx].sum().clamp_min(1e-8)
                li = torch.randperm(len(xl), device=device)[:cfg.batch_size]
                recon = self._mask_loss(torch.cat([xl[li], xu[idx]]), torch.cat([ol[li], ou[idx]]))
                total = sup+cfg.unlabelled_strength*ramp*unsup+cfg.reconstruction_strength*recon
                if not torch.isfinite(total):
                    raise FloatingPointError('Non-finite student training loss')
                total.backward()
                nn.utils.clip_grad_norm_([*self.encoder_.parameters(), *self.decoder_.parameters()], 5.)
                optimizer.step()
                losses.append([v.detach().item() for v in (total, sup, unsup, recon)])
            return np.mean(losses, axis=0).tolist()

        for e in range(cfg.warmup_epochs):
            losses = epoch()
            self.prototypes_ = self._anchors(xl, yl)
            self.history_.append({'phase': 'warmup', 'epoch': e, 'loss_total_supervised_unlabelled_mask': losses})

        prior0 = pt.mean(0).clamp_min(1e-6) if len(U) else torch.ones(K, device=device)/K
        prior0 = prior0/prior0.sum()
        prior, mass_ema = prior0.clone(), prior0.clone()
        self.pseudo_labels_ = np.empty((len(U), K), dtype=np.float32)
        for r in range(cfg.rounds):
            if len(U):
                zu = self._embedding(xu)
                cost = -cfg.teacher_strength*pt.clamp_min(1e-9).log()+cfg.geometry_strength*torch.cdist(zu, self.prototypes_).square()
                q = torch.cat([soft_assignment(cost[i:i+cfg.batch_size], prior,
                    cfg.assignment_entropy, cfg.prior_strength, cfg.assignment_steps)
                    for i in range(0, len(U), cfg.batch_size)])
                w = confidence_weights(pt, self._logits(zu).softmax(1), cfg.confidence_floor)
            else:
                q, w = pt, torch.empty(0, device=device)
            for e in range(cfg.epochs_per_round):
                losses = epoch(q, w, (r+1)/max(cfg.rounds, 1))
                self.history_.append({'phase': 'adapt', 'round': r, 'epoch': e,
                                     'loss_total_supervised_unlabelled_mask': losses})
            anchors = self._anchors(xl, yl)
            self.prototypes_ = anchors.clone()
            if len(U):
                zu = self._embedding(xu)
                weights = w[:, None]*q
                mass = weights.sum(0)
                pseudo = F.normalize(weights.T@zu/mass[:, None].clamp_min(1e-8), dim=1)
                supported = mass >= cfg.minimum_pseudo_mass
                mixed = F.normalize((1-cfg.prototype_mix)*anchors+cfg.prototype_mix*pseudo, dim=1)
                self.prototypes_[supported] = mixed[supported]
                mass_ema = cfg.prior_momentum*mass_ema+(1-cfg.prior_momentum)*q.mean(0)
                candidate = (1-cfg.prior_mix)*prior0+cfg.prior_mix*mass_ema
                candidate = torch.maximum(torch.minimum(candidate, prior*cfg.prior_max_ratio), prior/cfg.prior_max_ratio)
                prior = candidate.clamp_min(1e-6)/candidate.clamp_min(1e-6).sum()
                self.pseudo_labels_ = q.cpu().numpy()
                self.history_.append({'phase': 'state_update', 'round': r, 'soft_mass': q.sum(0).cpu().tolist(),
                    'reliable_mass': mass.cpu().tolist(), 'prior': prior.cpu().tolist()})
        if not cfg.rounds and len(U):
            self.pseudo_labels_ = pt.cpu().numpy()
        self.encoder_.eval()
        self.audit_.update(encoder_initial_sha256=initial, encoder_final_sha256=state_digest([self.encoder_]),
                           fit_seconds=time.perf_counter()-started, config=asdict(cfg))
        self.prior_ = prior.cpu().numpy()
        del self.decoder_  # inference does not carry the reconstruction head
        return self

    def predict_proba(self, X):
        X = matrix(X, features=self.n_features_in_, allow_empty=True)
        if not len(X):
            return np.empty((0, len(self.classes_)))
        transformed, _ = self.preprocessor_.transform(X)
        device = next(self.encoder_.parameters()).device
        x = torch.tensor(transformed, device=device)
        with torch.no_grad():
            return probabilities(self._logits(self._embedding(x)).softmax(1).cpu().numpy())

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def save(self, path):
        import joblib
        joblib.dump({'format': 1, 'config': asdict(self.config), 'classes': self.classes_,
            'n_features': self.n_features_in_, 'preprocessor': self.preprocessor_,
            'encoder': {k: v.detach().cpu() for k, v in self.encoder_.state_dict().items()},
            'prototypes': self.prototypes_.detach().cpu(), 'prior': self.prior_,
            'audit': self.audit_}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load a trusted locally generated model export."""
        import joblib
        data = joblib.load(path)
        if data['format'] != 1:
            raise ValueError('Unsupported student export version')
        cfg = StudentConfig(**dict(data['config'], device=device))
        obj = cls(None, cfg)
        obj.classes_, obj.n_features_in_ = data['classes'], data['n_features']
        obj.preprocessor_, obj.prior_, obj.audit_ = data['preprocessor'], data['prior'], data['audit']
        obj.encoder_ = Encoder(obj.preprocessor_.output_dim, cfg.hidden, cfg.embedding).to(device)
        obj.encoder_.load_state_dict(data['encoder'])
        obj.encoder_.eval()
        obj.prototypes_ = data['prototypes'].to(device)
        return obj
