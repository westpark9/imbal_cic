"""Distribution-targeted heads on a frozen, full-context TabPFN representation.

This is an adaptation of BalPoE, not a reproduction of its vision architecture.
For source prior q and target r, CE(f + log(q) - log(r), y) learns a head
whose unadjusted scores target r. No query label is accepted at inference.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LAMBDAS = (1., 0., -1.)
HEAD_NAMES = ('head_natural', 'head_balanced', 'head_inverse')


def prior_offset(source_counts, reference_counts, exponent):
    q = np.asarray(source_counts, dtype=np.float64)
    pi = np.asarray(reference_counts, dtype=np.float64)
    if np.any(q <= 0) or np.any(pi <= 0):
        raise ValueError('Every currently available class must have training examples')
    q /= q.sum(); pi /= pi.sum()
    return (np.log(q) - exponent * np.log(pi)).astype('float32')


def flatten_embeddings(embeddings, rows):
    z = np.asarray(embeddings)
    if z.ndim == 3:
        if z.shape[1] != rows:
            raise ValueError(f'Unexpected embedding axes: {z.shape}, rows={rows}')
        z = z.transpose(1, 0, 2).reshape(rows, -1)
    if z.ndim != 2 or len(z) != rows or not np.isfinite(z).all():
        raise ValueError('Invalid frozen representation')
    return np.ascontiguousarray(z, dtype='float32')


def embed(model, X, batch=1024):
    return np.concatenate([flatten_embeddings(model.get_embeddings(X[i:i+batch], 'test'),
                          len(X[i:i+batch])) for i in range(0, len(X), batch)])


class Head(nn.Module):
    def __init__(self, features, classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(),
                                 nn.Dropout(.1), nn.Linear(hidden, classes))

    def forward(self, x):
        return self.net(x)


def balanced_nll(logits, y, C):
    loss = F.cross_entropy(logits, y, reduction='none')
    return torch.stack([loss[y == c].mean() for c in range(C)]).mean()


def calibrate_temperature(likelihood_logits, labels, max_iter=50):
    """Fit scalar T on disjoint calibration rows, accounting for their prior."""
    C = likelihood_logits.shape[1]
    q = torch.bincount(labels, minlength=C).float(); q /= q.sum()
    if (q == 0).any():
        raise ValueError('Calibration split misses a class')
    log_t = torch.zeros((), device=likelihood_logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=.3, max_iter=max_iter,
                                line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(likelihood_logits / log_t.exp() + q.log(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    result = float(log_t.detach().exp())
    if not np.isfinite(result) or result <= 0:
        raise ValueError('Non-finite fitted temperature')
    return result


def fit_heads(zfit, yfit, zselect, yselect, zcal, ycal, reference_counts,
              config, out, progress=None):
    torch.cuda.synchronize(); work_start = time.perf_counter()
    out = Path(out)
    C = len(reference_counts)
    mean = zfit.mean(0); scale = zfit.std(0); scale[scale < 1e-6] = 1.
    arrays = [(zfit-mean)/scale, (zselect-mean)/scale, (zcal-mean)/scale]
    train, select, cal = [torch.tensor(z, device='cuda') for z in arrays]
    yt, ys, yc = [torch.tensor(y, dtype=torch.long, device='cuda')
                  for y in (yfit, yselect, ycal)]
    pi = np.asarray(reference_counts, dtype='float64'); pi /= pi.sum()
    logpi = torch.tensor(np.log(pi), dtype=torch.float32, device='cuda')
    states = []; temperatures = []; histories = []; durations = []
    for index, exponent in enumerate(LAMBDAS):
        torch.manual_seed(config['seed'] + 2011 + index)
        rng = np.random.default_rng(config['seed'] + 3011 + index)
        net = Head(train.shape[1], C, config['head_hidden']).cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=config['head_lr'], weight_decay=1e-4)
        offset = torch.tensor(prior_offset(np.bincount(yfit, minlength=C),
                              reference_counts, exponent), device='cuda')
        best = float('inf'); best_state = None; best_epoch = -1; history = []
        started = time.perf_counter()
        for epoch in range(config['head_epochs']):
            net.train(); order = torch.randperm(len(yt), device='cuda'); losses = []
            for start in range(0, len(order), config['head_batch']):
                idx = order[start:start+config['head_batch']]
                other = idx[torch.randperm(len(idx), device='cuda')]
                mix = float(rng.beta(.2, .2))
                logits = net(mix*train[idx] + (1-mix)*train[other]) + offset
                loss = mix*F.cross_entropy(logits, yt[idx]) + (1-mix)*F.cross_entropy(logits, yt[other])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Head training loss is not finite')
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
            net.eval()
            with torch.no_grad():
                # Uniform-target validation: each class contributes equally.
                score = float(balanced_nll(net(select)-exponent*logpi, ys, C))
            history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), selection_balanced_nll=score))
            if score < best:
                best = score; best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if progress and (epoch == 0 or (epoch+1) % 10 == 0):
                progress(head=HEAD_NAMES[index], epoch=epoch+1, selection_balanced_nll=score)
            if epoch-best_epoch >= config['head_patience']:
                break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            calibration_logits = (net(cal)-exponent*logpi).detach()
        temp = calibrate_temperature(calibration_logits, yc)
        torch.cuda.synchronize(); durations.append(time.perf_counter()-started)
        states.append(best_state); temperatures.append(temp)
        histories.append(dict(name=HEAD_NAMES[index], exponent=exponent, best_epoch=best_epoch+1,
                              best_selection_balanced_nll=best, temperature=temp, epochs=history))
        del net, optimizer
    payload = dict(states=states, temperatures=temperatures, mean=torch.tensor(mean),
                   scale=torch.tensor(scale), reference_counts=list(map(int, reference_counts)),
                   features=train.shape[1], classes=C, hidden=config['head_hidden'])
    torch.cuda.synchronize(); work_seconds = time.perf_counter()-work_start
    torch.save(payload, out/'heads.pt')
    (out/'head_training.json').write_text(json.dumps(dict(heads=histories,
        head_fit_seconds=durations, fit_rows=len(yfit), selection_rows=len(yselect),
        calibration_rows=len(ycal), source_counts=np.bincount(yfit, minlength=C).tolist(),
        calibration='disjoint per-head likelihood temperature with calibration prior',
        ensemble='mean calibrated log scores; target-prior exponents sum to zero',
        mixup_alpha=.2, input_features=train.shape[1]), indent=2)+'\n')
    return dict(work_seconds=work_seconds, heads=histories)


class HeadsPredictor:
    def __init__(self, out, base):
        self.base = base
        self.payload = torch.load(Path(out)/'heads.pt', map_location='cpu', weights_only=True)
        d = self.payload
        self.mean = d['mean'].numpy(); self.scale = d['scale'].numpy()
        pi = np.asarray(d['reference_counts'], dtype='float64'); pi /= pi.sum()
        self.logpi = torch.tensor(np.log(pi), dtype=torch.float32, device='cuda')
        self.models = []
        for state in d['states']:
            net = Head(d['features'], d['classes'], d['hidden']).cuda()
            net.load_state_dict(state); net.eval(); self.models.append(net)

    @torch.no_grad()
    def predict_with_aux(self, X):
        z = flatten_embeddings(self.base.get_embeddings(X, 'test'), len(X))
        z = torch.tensor((z-self.mean)/self.scale, device='cuda')
        scores = [(net(z)-lam*self.logpi)/temp + lam*self.logpi
                  for net, lam, temp in zip(self.models, LAMBDAS, self.payload['temperatures'])]
        p = torch.softmax(torch.stack(scores).mean(0), -1).cpu().numpy()
        auxiliary = {name: logits.argmax(-1).cpu().numpy() for name, logits in zip(HEAD_NAMES, scores)}
        return p, auxiliary

    def close(self):
        self.base = None; self.models = []
