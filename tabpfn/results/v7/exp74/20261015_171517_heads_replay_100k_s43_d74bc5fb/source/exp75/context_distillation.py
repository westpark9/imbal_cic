"""Label-balanced in-context distillation with frozen TabPFN-v3 weights.

Gradient training and checkpoint selection use only currently available train
and validation rows. Synthetic context tokens are not claimed to be valid flows.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def context_indices(y, budget, seed):
    """Balanced initialization, keeping at least half of each class as queries."""
    rng = np.random.default_rng(seed)
    C = int(np.max(y))+1
    counts = np.bincount(y, minlength=C)
    if np.any(counts < 2):
        raise ValueError('ICD needs distinct initialization and query examples per class')
    caps = counts//2; take = np.zeros(C, dtype=int)
    remaining = min(budget, int(caps.sum()))
    while remaining:
        available = np.flatnonzero(take < caps)
        if not len(available): break
        for c in available:
            if remaining == 0: break
            take[c] += 1; remaining -= 1
    return np.concatenate([rng.permutation(np.flatnonzero(y == c))[:take[c]] for c in range(C)])


def make_differentiable(config, C):
    from tabpfn import TabPFNClassifier
    model = TabPFNClassifier(model_path=config['checkpoint'], ignore_pretraining_limits=True,
        device='cuda:0', n_estimators=1, auto_scale_n_estimators=False,
        random_state=config['seed'], inference_precision=torch.float32,
        differentiable_input=True, n_preprocessing_jobs=1)
    model.n_classes_ = C
    return model


def freeze_backbone(model):
    for network in model.models_:
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)


@torch.no_grad()
def predict_tensor(model, context, labels, queries, batch):
    model.fit_with_differentiable_input(context.detach(), labels)
    return torch.cat([model.forward(queries[i:i+batch], use_inference_mode=True)
                      for i in range(0, len(queries), batch)])


def fit_context(data, config, out, progress=None):
    torch.cuda.synchronize(); work_start = time.perf_counter()
    out = Path(out); C = len(data['names'])
    torch.manual_seed(config['seed']); rng = np.random.default_rng(config['seed'])
    X = data['Xtr']; y = data['ytr']
    mean = X.mean(0, dtype=np.float64).astype('float32')
    scale = X.std(0, dtype=np.float64).astype('float32'); scale[scale < 1e-6] = 1.
    selected = context_indices(y, config['context_budget'], config['seed'])
    query_indices = np.setdiff1d(np.arange(len(y)), selected)
    query_by_class = [query_indices[y[query_indices] == c] for c in range(C)]
    initial = ((X[selected]-mean)/scale).astype('float32')
    context = torch.tensor(initial, device='cuda', requires_grad=True)
    labels = torch.tensor(y[selected], dtype=torch.float32, device='cuda')
    # Training queries and selection cases are separated; test never enters this function.
    select_x = torch.tensor((data['X_cal']-mean)/scale, device='cuda')
    select_y = torch.tensor(data['y_cal'], dtype=torch.long, device='cuda')
    model = make_differentiable(config, C)
    model.fit_with_differentiable_input(context, labels); freeze_backbone(model)
    torch.cuda.synchronize(); initial_prepare_seconds = time.perf_counter()-work_start
    from distribution_heads import balanced_nll

    def evaluate():
        p = predict_tensor(model, context, labels, select_x, config['icd_eval_batch'])
        loss = float(balanced_nll(p.clamp_min(1e-12).log(), select_y, C))
        pred = p.argmax(1)
        benign = data['names'].index('benign')
        fpr = float((pred[select_y == benign] != benign).float().mean())
        return dict(selection_balanced_nll=loss, selection_benign_fpr=fpr)

    initial_score = evaluate(); best = initial_score['selection_balanced_nll']
    best_context = context.detach().cpu().clone(); best_step = 0
    history = [dict(step=0, **initial_score)]
    optimizer = torch.optim.Adam([context], lr=config['icd_lr'])
    start = time.perf_counter()
    for step in range(1, config['icd_steps']+1):
        class_draws = rng.integers(0, C, size=config['icd_query_batch'])
        ids = np.array([rng.choice(query_by_class[c]) for c in class_draws])
        qx = torch.tensor((X[ids]-mean)/scale, device='cuda')
        qy = torch.tensor(y[ids], dtype=torch.long, device='cuda')
        optimizer.zero_grad()
        model.fit_with_differentiable_input(context, labels)
        p = model.forward(qx, use_inference_mode=True)
        loss = F.nll_loss(p.clamp_min(1e-12).log(), qy)
        if not torch.isfinite(loss): raise FloatingPointError('ICD loss is non-finite')
        loss.backward()
        if context.grad is None or not torch.isfinite(context.grad).all():
            raise FloatingPointError('ICD context gradient is missing or non-finite; do not silently zero it')
        gradient_norm = float(context.grad.norm())
        if gradient_norm == 0: raise FloatingPointError('ICD context has a zero gradient')
        if any(p.grad is not None for n in model.models_ for p in n.parameters()):
            raise AssertionError('Frozen backbone received parameter gradients')
        optimizer.step()
        if not torch.isfinite(context).all(): raise FloatingPointError('ICD context is non-finite')
        if step % config['icd_eval_every'] == 0 or step == config['icd_steps']:
            score = evaluate()
            row = dict(step=step, training_balanced_query_nll=float(loss.detach()),
                       gradient_norm=gradient_norm, elapsed_seconds=time.perf_counter()-start, **score)
            history.append(row)
            if score['selection_balanced_nll'] < best:
                best = score['selection_balanced_nll']; best_step = step
                best_context = context.detach().cpu().clone()
            if progress: progress(**row)
    torch.cuda.synchronize(); work_seconds = time.perf_counter()-work_start
    np.savez(out/'context.npz', initial=initial, tuned=best_context.numpy(), labels=y[selected],
             mean=mean, scale=scale, context_indices=selected, query_indices=query_indices)
    (out/'context_training.json').write_text(json.dumps(dict(history=history, best_step=best_step,
        context_rows=len(selected), context_counts=np.bincount(y[selected], minlength=C).tolist(),
        query_counts=[len(v) for v in query_by_class], query_sampling='uniform classes with replacement',
        selection='minimum class-balanced validation NLL; initial context is an eligible checkpoint',
        non_finite_gradient_policy='fail, never replace non-finite gradients',
        backbone_frozen=True, synthetic_context_only=True,
        changed_norm=float(np.linalg.norm(best_context.numpy()-initial))), indent=2)+'\n')
    del model, optimizer
    return dict(work_seconds=work_seconds, initial_prepare_seconds=initial_prepare_seconds,
                selected_step=best_step, context_rows=len(selected))


class ContextPredictor:
    def __init__(self, out, config, C, variant='tuned'):
        d = np.load(Path(out)/'context.npz')
        self.mean = d['mean']; self.scale = d['scale']
        self.model = make_differentiable(config, C)
        self.context = torch.tensor(d[variant], device='cuda')
        self.labels = torch.tensor(d['labels'], dtype=torch.float32, device='cuda')
        self.model.fit_with_differentiable_input(self.context, self.labels)
        freeze_backbone(self.model)

    @torch.no_grad()
    def predict_with_aux(self, X):
        query = torch.tensor((X-self.mean)/self.scale, device='cuda')
        return self.model.forward(query, use_inference_mode=True).cpu().numpy(), {}

    def close(self):
        self.model = None
