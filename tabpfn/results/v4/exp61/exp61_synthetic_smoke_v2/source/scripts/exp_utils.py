#!/usr/bin/env python3
"""Shared experiment helpers for the NF-v3 OOD-expert scripts (s43+).

Helpers are copied verbatim from their frozen origins so past experiments'
code paths never change (CLAUDE.md M3):
  * s24_code_ood_v1.py  : setup_logger, render_table_png (+palette)
  * s31_cic_mlp_energy_oe.py : seeding/device/io, datasets, TabularMLP,
    pretrain/energy/finetune_energy, ood_metrics
  * s40_nfv3_cic2018_global_oracle.py : scenario_chronological_split,
    variable_feature_mask, labels_for

New scripts should import from here instead of from other experiment
scripts.  Do not change an existing helper's behavior in place — add a new
helper alongside instead.  Import pattern from src/ scripts:

    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    from exp_utils import ...
"""

import copy
import logging
import math
import os
import pickle
import random
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


# --- Table palette (s24_code_ood_v1) -----------------------------------------

BLUE = "#cce5ff"
RED = "#ffcccc"
YELLOW = "#fff9cc"
GRAY = "#f2f2f2"
WHITE = "#ffffff"
EPS = 0.001


def setup_logger(path):
    log = logging.getLogger("ood_tta_gate")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    for h in [logging.StreamHandler(), logging.FileHandler(path)]:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def _cell_color(col, val, delta_cols, high_good, low_good):
    if not isinstance(val, (int, float)) or (isinstance(val, float) and np.isnan(val)):
        return WHITE
    if col in delta_cols:
        return BLUE if val > EPS else RED if val < -EPS else YELLOW
    if col in high_good:        # closer to 1.0 is better
        return BLUE if val >= 0.9 else YELLOW if val >= 0.7 else RED
    if col in low_good:         # closer to 0.0 is better
        return BLUE if val <= 0.02 else YELLOW if val <= 0.1 else RED
    return WHITE


def render_table_png(df, path, title="", delta_cols=(), high_good=(),
                     low_good=(), fmt="{:.3f}"):
    """Render a DataFrame as a color-coded PNG table (good=blue, bad=red)."""
    cols = list(df.columns)
    cell_text, cell_colors = [], []
    for _, row in df.iterrows():
        texts, colors = [], []
        for col in cols:
            v = row[col]
            if isinstance(v, float):
                texts.append("" if np.isnan(v) else fmt.format(v))
            else:
                texts.append(str(v))
            colors.append(_cell_color(col, v, delta_cols, high_good, low_good))
        cell_text.append(texts)
        cell_colors.append(colors)
    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 1.35),
                                    max(3, len(df) * 0.42)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Torch tabular Energy-OOD stack (s31_cic_mlp_energy_oe) ------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_output_dir(root, tag):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.join(root, f"{timestamp}_pid{os.getpid()}_{tag}")
    path = base
    suffix = 1
    while os.path.exists(path):
        path = f"{base}_{suffix:02d}"
        suffix += 1
    os.makedirs(path)
    return path


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def subset_indices(indices, maximum, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, maximum, replace=False))


def fit_scaler(X, indices, chunk_size=200_000):
    total = 0
    sums = np.zeros(X.shape[1], dtype=np.float64)
    sums_sq = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        values = np.asarray(X[indices[start:start + chunk_size]], dtype=np.float64)
        sums += values.sum(axis=0)
        sums_sq += np.square(values).sum(axis=0)
        total += len(values)
    mean = sums / max(total, 1)
    variance = np.maximum(sums_sq / max(total, 1) - np.square(mean), 1e-12)
    scale = np.sqrt(variance)
    return mean.astype(np.float32), scale.astype(np.float32)


class IndexedDataset(Dataset):
    def __init__(self, X, indices, labels, mean, scale):
        self.X = X
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mean = mean
        self.scale = scale

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x = (np.asarray(self.X[self.indices[item]], dtype=np.float32) - self.mean) / self.scale
        return torch.from_numpy(x), int(self.labels[item])


class FeatureDataset(Dataset):
    def __init__(self, X, indices, mean, scale):
        self.X = X
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean
        self.scale = scale

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x = (np.asarray(self.X[self.indices[item]], dtype=np.float32) - self.mean) / self.scale
        return torch.from_numpy(x)


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(previous, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            previous = hidden
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous, n_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


def parse_hidden_dims(value):
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"Invalid hidden dimensions: {value}")
    return dims


def make_loader(dataset, batch_size, shuffle, workers):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def sqrt_balanced_weights(labels, n_classes):
    counts = np.maximum(np.bincount(labels, minlength=n_classes), 1)
    weights = np.sqrt(len(labels) / (n_classes * counts))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_classification(model, loader, device, n_classes):
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            truth.append(y.numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(predictions)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=np.arange(n_classes),
            average="macro", zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, labels=np.arange(n_classes),
            average="weighted", zero_division=0,
        )),
    }


def pretrain(model, train_loader, val_loader, class_weights, args, device, log):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay
    )
    best_state, best_f1, stale = None, -math.inf, 0
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        total_loss, total_rows = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            total_rows += len(y)
        metrics = evaluate_classification(model, val_loader, device, len(class_weights))
        log.info(
            f"Pretrain epoch={epoch:03d} loss={total_loss/max(total_rows,1):.5f} "
            f"val_macro_f1={metrics['macro_f1']:.5f} val_acc={metrics['accuracy']:.5f}"
        )
        if metrics["macro_f1"] > best_f1 + 1e-6:
            best_f1 = metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if args.pretrain_patience > 0 and stale >= args.pretrain_patience:
                log.info(f"Pretrain early stop at epoch={epoch}")
                break
    model.load_state_dict(best_state)
    return best_f1


def energy(logits, temperature):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def collect_outputs(model, loader, device, temperature, labels=True):
    model.eval()
    energies, predictions, truth = [], [], []
    with torch.no_grad():
        for batch in loader:
            if labels:
                x, y = batch
                truth.append(y.numpy())
            else:
                x = batch
            logits = model(x.to(device, non_blocking=True))
            energies.append(energy(logits, temperature).cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    result = {
        "energy": np.concatenate(energies).astype(np.float32),
        "prediction": np.concatenate(predictions).astype(np.int32),
    }
    if labels:
        result["truth"] = np.concatenate(truth).astype(np.int32)
    return result


def auto_margins(model, val_loader, device, temperature, gap_std):
    values = collect_outputs(model, val_loader, device, temperature)["energy"]
    m_in = float(np.quantile(values, 0.95))
    spread = float(max(np.std(values), 1e-3))
    m_out = float(m_in + gap_std * spread)
    return m_in, m_out, spread


def finetune_energy(model, id_loader, aux_loader, val_loader, class_weights,
                    m_in, m_out, args, device, log):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay
    )
    aux_iterator = iter(aux_loader)
    history = []
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "energy": 0.0, "rows": 0}
        for x_in, y_in in id_loader:
            try:
                x_out = next(aux_iterator)
            except StopIteration:
                aux_iterator = iter(aux_loader)
                x_out = next(aux_iterator)
            x_in = x_in.to(device, non_blocking=True)
            y_in = y_in.to(device, non_blocking=True)
            x_out = x_out.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_in = model(x_in)
            logits_out = model(x_out)
            ce = F.cross_entropy(logits_in, y_in, weight=class_weights)
            energy_in = energy(logits_in, args.temperature)
            energy_out = energy(logits_out, args.temperature)
            margin_loss = (
                F.relu(energy_in - m_in).pow(2).mean()
                + F.relu(m_out - energy_out).pow(2).mean()
            )
            loss = ce + args.energy_weight * margin_loss
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n = len(y_in)
            totals["loss"] += float(loss.detach()) * n
            totals["ce"] += float(ce.detach()) * n
            totals["energy"] += float(margin_loss.detach()) * n
            totals["rows"] += n
        metrics = evaluate_classification(model, val_loader, device, len(class_weights))
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / totals["rows"],
            "ce_loss": totals["ce"] / totals["rows"],
            "energy_margin_loss": totals["energy"] / totals["rows"],
            "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
        }
        history.append(row)
        log.info(
            f"Finetune epoch={epoch:03d} loss={row['loss']:.5f} "
            f"ce={row['ce_loss']:.5f} energy={row['energy_margin_loss']:.5f} "
            f"val_macro_f1={row['val_macro_f1']:.5f} val_acc={row['val_accuracy']:.5f}"
        )
    return pd.DataFrame(history)


def ood_metrics(id_energy, ood_energy, threshold):
    # Lower energy is expected for ID; use -energy as the ID-positive score.
    y_true = np.concatenate([
        np.ones(len(id_energy), dtype=np.int8),
        np.zeros(len(ood_energy), dtype=np.int8),
    ])
    knownness = -np.concatenate([id_energy, ood_energy])
    fpr, tpr, _ = roc_curve(y_true, knownness)
    index = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[index[0]]) if len(index) else 1.0
    return {
        "auroc": float(roc_auc_score(y_true, knownness)),
        "aupr_id": float(average_precision_score(y_true, knownness)),
        "fpr95": fpr95,
        "id_retain_at_tau": float((id_energy <= threshold).mean()),
        "ood_detect_at_tau": float((ood_energy > threshold).mean()),
    }


# --- NF-v3 suite split helpers (s40_nfv3_cic2018_global_oracle) --------------


def scenario_chronological_split(indices, scenarios, timestamps):
    split = {"train": [], "val": [], "test": []}
    audit = []
    for scenario in sorted(np.unique(scenarios[indices])):
        selected = indices[scenarios[indices] == scenario]
        ordered = selected[np.argsort(timestamps[selected], kind="stable")]
        n_train = int(len(ordered) * 0.6)
        n_val = int(len(ordered) * 0.2)
        n_test = len(ordered) - n_train - n_val
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(f"Scenario {scenario!r} is too small for 60/20/20")
        split["train"].extend(ordered[:n_train])
        split["val"].extend(ordered[n_train:n_train + n_val])
        split["test"].extend(ordered[n_train + n_val:])
        audit.append({
            "scenario": str(scenario), "total": len(ordered),
            "train": n_train, "val": n_val, "test": n_test,
            "train_last_timestamp": int(timestamps[ordered[n_train - 1]]),
            "val_first_timestamp": int(timestamps[ordered[n_train]]),
            "val_last_timestamp": int(timestamps[ordered[n_train + n_val - 1]]),
            "test_first_timestamp": int(timestamps[ordered[n_train + n_val]]),
        })
    return {
        name: np.sort(np.asarray(values, dtype=np.int64))
        for name, values in split.items()
    }, pd.DataFrame(audit)


def variable_feature_mask(X, indices, chunk_size=200_000):
    minimum = np.full(X.shape[1], np.inf, dtype=np.float64)
    maximum = np.full(X.shape[1], -np.inf, dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        values = np.asarray(X[indices[start:start + chunk_size]])
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
    return maximum > minimum


def labels_for(indices, families, class_names):
    mapping = {name: i for i, name in enumerate(class_names)}
    return np.asarray([mapping[str(x)] for x in families[indices]], dtype=np.int64)
