#!/usr/bin/env python3
"""
Analyze CIC-IDS2017 v2 class feature geometry.

This script reads data/cic2017_chrono_v2.pkl and, by default, uses the full
selected split for PCA plots and centroid-distance tables. t-SNE can be enabled
on a capped subset because full-dataset t-SNE is usually impractical.
"""
import argparse
import os
import pickle
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


DEFAULT_FAMILIES = {
    "normal": "Normal",
    "bot": "Friday_OOD",
    "ddos": "Friday_OOD",
    "portscan": "Friday_OOD",
    "dos-goldeneye": "DoS",
    "dos-hulk": "DoS",
    "dos-slowhttptest": "DoS",
    "dos-slowloris": "DoS",
    "heartbleed": "Rare",
    "ftp-patator": "Credential",
    "ssh-patator": "Credential",
    "web-attack-brute-force": "Web",
    "web-attack-xss": "Web",
    "web-attack-sql-injection": "Rare",
    "infiltration": "Rare",
}


NORMAL_COLOR = "#000000"
RAINBOW_COLORS = [
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#bcbd22",  # yellow/olive
    "#2ca02c",  # green
    "#17becf",  # cyan
    "#1f77b4",  # blue
    "#4b0082",  # indigo
    "#9467bd",  # violet
    "#e377c2",  # magenta
    "#8c564b",  # brown
    "#ff9896",
    "#ffbb78",
    "#dbdb8d",
    "#98df8a",
    "#9edae5",
    "#aec7e8",
    "#c5b0d5",
    "#f7b6d2",
]


def make_out_dir(root):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(root, f"{ts}_embedding_v2")
    os.makedirs(path, exist_ok=False)
    return path


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def split_indices(data, split):
    if split == "train":
        return np.asarray(data["train_indices"], dtype=int)
    if split == "val":
        return np.asarray(data["val_indices"], dtype=int)
    if split == "test_known":
        return np.asarray(data["test_known_indices"], dtype=int)
    if split == "test_ood":
        return np.asarray(data["test_ood_indices"], dtype=int)
    if split == "test_combined":
        return np.asarray(data["test_combined_indices"], dtype=int)
    if split == "all":
        return np.arange(len(data["y"]), dtype=int)
    raise ValueError(f"Unknown split: {split}")


def select_by_class(y, candidate_idx, max_per_class, seed):
    if max_per_class is None or max_per_class <= 0:
        rows = []
        for cls in sorted(np.unique(y[candidate_idx]).tolist()):
            cls_idx = candidate_idx[y[candidate_idx] == cls]
            rows.append({"class_id": int(cls), "available": int(len(cls_idx)), "selected": int(len(cls_idx))})
        return np.sort(candidate_idx), pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    selected = []
    rows = []
    for cls in sorted(np.unique(y[candidate_idx]).tolist()):
        cls_idx = candidate_idx[y[candidate_idx] == cls]
        n_take = min(len(cls_idx), max_per_class)
        chosen = rng.choice(cls_idx, size=n_take, replace=False)
        selected.append(chosen)
        rows.append({"class_id": int(cls), "available": int(len(cls_idx)), "selected": int(n_take)})
    return np.sort(np.concatenate(selected)), pd.DataFrame(rows)


def class_names_for_ids(class_names, ids):
    return [class_names[int(i)] for i in ids]


def ordered_labels(df, color_col, normal_label=None, label_order=None):
    if label_order is None:
        counts = df[color_col].value_counts()
        labels = counts.index.tolist()
    else:
        present = set(df[color_col].tolist())
        labels = [label for label in label_order if label in present]
        labels.extend(label for label in df[color_col].value_counts().index if label not in labels)
    if normal_label is not None and normal_label in labels:
        labels = [normal_label] + [label for label in labels if label != normal_label]
    return labels


def make_color_map(labels, normal_label=None):
    color_map = {}
    color_idx = 0
    for label in labels:
        if normal_label is not None and label == normal_label:
            color_map[label] = NORMAL_COLOR
        else:
            color_map[label] = RAINBOW_COLORS[color_idx % len(RAINBOW_COLORS)]
            color_idx += 1
    return color_map


def save_color_legend(labels, color_map, path):
    rows = [{"label": label, "color": color_map[label]} for label in labels]
    pd.DataFrame(rows).to_csv(path, index=False)


def save_scatter(df, x_col, y_col, color_col, path, title, normal_label=None,
                 legend_path=None, label_order=None, point_size=2.0,
                 alpha=0.22, rasterized=True):
    labels = ordered_labels(df, color_col, normal_label, label_order)
    color_map = make_color_map(labels, normal_label)
    if legend_path:
        save_color_legend(labels, color_map, legend_path)
    fig, ax = plt.subplots(figsize=(12, 9))
    for label in labels:
        sub = df[df[color_col] == label]
        ax.scatter(sub[x_col], sub[y_col], s=point_size, alpha=alpha,
                   color=color_map[label], label=label, linewidths=0,
                   rasterized=rasterized)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(markerscale=2.0, fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_centroid_distances(embedding_df, label_col, out_dir, tag):
    centers = embedding_df.groupby(label_col)[["pca1", "pca2"]].mean()
    dists = pairwise_distances(centers.values, metric="euclidean")
    dist_df = pd.DataFrame(dists, index=centers.index, columns=centers.index)
    dist_df.to_csv(os.path.join(out_dir, f"{tag}_centroid_distance_matrix.csv"))
    pairs = []
    labels = list(centers.index)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append({
                f"{label_col}_a": labels[i],
                f"{label_col}_b": labels[j],
                "distance": float(dists[i, j]),
            })
    pd.DataFrame(pairs).sort_values("distance").to_csv(
        os.path.join(out_dir, f"{tag}_nearest_centroid_pairs.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cic2017_chrono_v2.pkl")
    parser.add_argument("--split", default="test_combined",
                        choices=["train", "val", "test_known", "test_ood", "test_combined", "all"])
    parser.add_argument("--exclude_normal", action="store_true",
                        help="Exclude normal/benign rows before PCA and t-SNE analysis.")
    parser.add_argument("--max_per_class", type=int, default=0,
                        help="0 or negative means use all rows in the selected split")
    parser.add_argument("--run_tsne", action="store_true",
                        help="Run t-SNE on a capped subset. Disabled by default for full-data analysis.")
    parser.add_argument("--max_tsne_samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", default="results")
    args = parser.parse_args()

    data = load_data(args.data)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=int)
    class_names = list(data["class_names"])
    candidate_idx = split_indices(data, args.split)
    if args.exclude_normal:
        normal_ids = [i for i, name in enumerate(class_names) if name == "normal"]
        if normal_ids:
            candidate_idx = candidate_idx[~np.isin(y[candidate_idx], normal_ids)]
        if len(candidate_idx) == 0:
            raise ValueError(f"No rows remain after --exclude_normal for split={args.split}")
    selected_idx, selection_summary = select_by_class(
        y, candidate_idx, args.max_per_class, args.seed)

    out_dir = make_out_dir(args.out_root)
    selection_summary["class"] = class_names_for_ids(class_names, selection_summary["class_id"])
    selection_summary["family"] = selection_summary["class"].map(DEFAULT_FAMILIES).fillna("Other")
    selection_summary.to_csv(os.path.join(out_dir, "class_count_summary.csv"), index=False)
    class_order = selection_summary.sort_values("available", ascending=False)["class"].tolist()
    family_order = (
        selection_summary.groupby("family", as_index=False)["available"].sum()
        .sort_values("available", ascending=False)["family"].tolist()
    )

    X_selected = X[selected_idx]
    y_selected = y[selected_idx]
    labels = np.array(class_names_for_ids(class_names, y_selected), dtype=object)
    families = np.array([DEFAULT_FAMILIES.get(label, "Other") for label in labels], dtype=object)
    split_roles = np.where(np.isin(selected_idx, data["test_ood_indices"]), "Friday_OOD", "Known_Source")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    pca_model = PCA(n_components=min(20, X_scaled.shape[0], X_scaled.shape[1]), random_state=args.seed)
    pca_features = pca_model.fit_transform(X_scaled)

    df = pd.DataFrame({
        "index": selected_idx,
        "class_id": y_selected,
        "class": labels,
        "family": families,
        "split_role": split_roles,
        "pca1": pca_features[:, 0],
        "pca2": pca_features[:, 1],
    })
    df.to_csv(os.path.join(out_dir, "embedding_points_pca.csv"), index=False)
    pd.DataFrame({
        "component": np.arange(1, len(pca_model.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca_model.explained_variance_ratio_,
    }).to_csv(os.path.join(out_dir, "pca_explained_variance.csv"), index=False)

    save_scatter(df, "pca1", "pca2", "class",
                 os.path.join(out_dir, "pca_by_class.png"),
                 f"PCA by class ({args.split}, attacks only, full split)"
                 if args.exclude_normal and args.max_per_class <= 0
                 else f"PCA by class ({args.split}, full split)" if args.max_per_class <= 0
                 else f"PCA by class ({args.split}, attacks only, max_per_class={args.max_per_class})"
                 if args.exclude_normal
                 else f"PCA by class ({args.split}, max_per_class={args.max_per_class})",
                 normal_label="normal",
                 legend_path=os.path.join(out_dir, "class_color_legend.csv"),
                 label_order=class_order)
    save_scatter(df, "pca1", "pca2", "family",
                 os.path.join(out_dir, "pca_by_family.png"),
                 f"PCA by family ({args.split})",
                 normal_label="Normal",
                 legend_path=os.path.join(out_dir, "family_color_legend.csv"),
                 label_order=family_order)
    save_scatter(df, "pca1", "pca2", "split_role",
                 os.path.join(out_dir, "pca_by_split_role.png"),
                 f"PCA by source vs Friday OOD ({args.split})")
    save_centroid_distances(df, "class", out_dir, "pca_class")
    save_centroid_distances(df, "family", out_dir, "pca_family")

    if not args.run_tsne:
        print(f"Saved embedding analysis to {out_dir}")
        print("Skipped t-SNE. Re-run with --run_tsne to generate t-SNE plots.")
        return

    if len(df) > args.max_tsne_samples:
        rng = np.random.default_rng(args.seed + 17)
        tsne_rows = np.sort(rng.choice(len(df), size=args.max_tsne_samples, replace=False))
    else:
        tsne_rows = np.arange(len(df))
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                random_state=args.seed)
    tsne_xy = tsne.fit_transform(pca_features[tsne_rows, :min(20, pca_features.shape[1])])
    tsne_df = df.iloc[tsne_rows].copy()
    tsne_df["tsne1"] = tsne_xy[:, 0]
    tsne_df["tsne2"] = tsne_xy[:, 1]
    tsne_df.to_csv(os.path.join(out_dir, "embedding_points_tsne.csv"), index=False)
    save_scatter(tsne_df, "tsne1", "tsne2", "class",
                 os.path.join(out_dir, "tsne_by_class.png"),
                 f"t-SNE by class ({args.split}, n={len(tsne_df)})",
                 normal_label="normal",
                 label_order=class_order)
    save_scatter(tsne_df, "tsne1", "tsne2", "family",
                 os.path.join(out_dir, "tsne_by_family.png"),
                 f"t-SNE by family ({args.split}, n={len(tsne_df)})",
                 normal_label="Normal",
                 label_order=family_order)
    save_scatter(tsne_df, "tsne1", "tsne2", "split_role",
                 os.path.join(out_dir, "tsne_by_split_role.png"),
                 f"t-SNE by source vs Friday OOD ({args.split}, n={len(tsne_df)})")

    print(f"Saved embedding analysis to {out_dir}")


if __name__ == "__main__":
    main()
