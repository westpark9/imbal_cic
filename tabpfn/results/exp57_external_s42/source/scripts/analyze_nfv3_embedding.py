#!/usr/bin/env python3
"""
Analyze NF-v3 class feature geometry for one target dataset.

Forked from analyze_cic2017_embedding_v2.py, which reads the older
data/cic2017_chrono_v2.pkl.  This one reads the NF-v3 suite pickle used by
scripts/s43_nfv3_independent_expert_energy.py / s44_nfv3_resolved_expert_pipeline.py
(data/nfv3_energy_suite_uncapped_scenarios.pkl by default) -- different
dataset, different feature schema (NetFlow v3, not CICFlowMeter), do not mix
results between the two embedding scripts.

Colors by raw class ("family" field in the suite) and by expert-ownership
group (configs/nfv3_experts.json), plus a class_dispersion table: each
class's mean distance to its own PCA centroid vs its distance to benign's
centroid, in full (not just 2D-plot) PCA space -- answers "does this class
have a centroid-shaped, compact distribution, and how close does it sit to
benign" quantitatively, not just visually.
"""
import argparse
import json
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


NORMAL_COLOR = "#000000"
RAINBOW_COLORS = [
    "#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#17becf", "#1f77b4",
    "#4b0082", "#9467bd", "#e377c2", "#8c564b", "#ff9896", "#ffbb78",
]


def make_out_dir(root):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(root, f"{ts}_nfv3_embedding")
    os.makedirs(path, exist_ok=False)
    return path


def load_group_map(config_path, target):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if target not in config:
        raise ValueError(f"Target {target!r} absent from {config_path}")
    group_map = {"benign": "benign"}
    for group, classes in config[target].items():
        for class_name in classes:
            group_map[class_name] = group
    return group_map


def select_by_class(families, max_per_class, seed):
    rng = np.random.default_rng(seed)
    selected, rows = [], []
    for name in sorted(np.unique(families).tolist()):
        idx = np.flatnonzero(families == name)
        n_take = len(idx) if max_per_class <= 0 else min(len(idx), max_per_class)
        chosen = rng.choice(idx, size=n_take, replace=False) if n_take < len(idx) else idx
        selected.append(chosen)
        rows.append({"class": name, "available": int(len(idx)), "selected": int(n_take)})
    return np.sort(np.concatenate(selected)), pd.DataFrame(rows)


def ordered_labels(df, color_col, normal_label=None, label_order=None):
    if label_order is None:
        labels = df[color_col].value_counts().index.tolist()
    else:
        present = set(df[color_col].tolist())
        labels = [label for label in label_order if label in present]
        labels.extend(
            label for label in df[color_col].value_counts().index
            if label not in labels
        )
    if normal_label is not None and normal_label in labels:
        labels = [normal_label] + [label for label in labels if label != normal_label]
    return labels


def make_color_map(labels, normal_label=None):
    color_map, color_idx = {}, 0
    for label in labels:
        if normal_label is not None and label == normal_label:
            color_map[label] = NORMAL_COLOR
        else:
            color_map[label] = RAINBOW_COLORS[color_idx % len(RAINBOW_COLORS)]
            color_idx += 1
    return color_map


def save_scatter(df, x_col, y_col, color_col, path, title, normal_label=None,
                 label_order=None, point_size=2.0, alpha=0.22):
    labels = ordered_labels(df, color_col, normal_label, label_order)
    color_map = make_color_map(labels, normal_label)
    fig, ax = plt.subplots(figsize=(12, 9))
    for label in labels:
        sub = df[df[color_col] == label]
        ax.scatter(sub[x_col], sub[y_col], s=point_size, alpha=alpha,
                   color=color_map[label], label=label, linewidths=0,
                   rasterized=True)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(markerscale=2.0, fontsize=8, ncol=1, bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_centroid_distances(embedding_df, label_col, feature_cols, out_dir, tag):
    centers = embedding_df.groupby(label_col)[feature_cols].mean()
    dists = pairwise_distances(centers.values, metric="euclidean")
    dist_df = pd.DataFrame(dists, index=centers.index, columns=centers.index)
    dist_df.to_csv(os.path.join(out_dir, f"{tag}_centroid_distance_matrix.csv"))
    return centers, dists


def save_class_dispersion(embedding_df, feature_cols, centers, out_dir, normal_label):
    values = embedding_df[feature_cols].values
    center_lookup = centers.to_dict(orient="index")
    normal_center = np.array(
        [center_lookup[normal_label][col] for col in feature_cols]
    ) if normal_label in center_lookup else None
    rows = []
    for name, group in embedding_df.groupby("class"):
        own_center = np.array([center_lookup[name][col] for col in feature_cols])
        own_values = group[feature_cols].values
        own_dispersion = float(
            np.linalg.norm(own_values - own_center, axis=1).mean()
        )
        row = {
            "class": name, "n": int(len(group)),
            "mean_distance_to_own_centroid": own_dispersion,
        }
        if normal_center is not None:
            row["distance_to_benign_centroid"] = float(
                np.linalg.norm(own_center - normal_center)
            )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("class").reset_index(drop=True)
    frame.to_csv(os.path.join(out_dir, "class_dispersion.csv"), index=False)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/nfv3_energy_suite_uncapped_scenarios.pkl"
    )
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--expert-config", default="configs/nfv3_experts.json"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=5000,
        help="0 or negative means use all rows for that class",
    )
    parser.add_argument("--run-tsne", action="store_true")
    parser.add_argument("--max-tsne-samples", type=int, default=12000)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", default="results")
    args = parser.parse_args()

    with open(args.data, "rb") as handle:
        suite = pickle.load(handle)
    datasets = np.asarray(suite["dataset_names"], dtype=object)
    families_all = np.asarray(suite["families"], dtype=object)
    target_indices = np.flatnonzero(datasets == args.target)
    if not len(target_indices):
        raise ValueError(f"Target {args.target!r} absent from {args.data}")
    families = families_all[target_indices]
    group_map = load_group_map(args.expert_config, args.target)

    selected_local, selection_summary = select_by_class(
        families, args.max_per_class, args.seed
    )
    selected_idx = target_indices[selected_local]
    selection_summary["group"] = selection_summary["class"].map(group_map).fillna("unassigned")

    out_dir = make_out_dir(args.out_root)
    selection_summary.to_csv(
        os.path.join(out_dir, "class_count_summary.csv"), index=False
    )
    class_order = selection_summary.sort_values(
        "available", ascending=False
    )["class"].tolist()
    group_order = (
        selection_summary.groupby("group", as_index=False)["available"].sum()
        .sort_values("available", ascending=False)["group"].tolist()
    )

    X = np.asarray(suite["X"], dtype=np.float32)
    X_selected = X[selected_idx]
    class_selected = families[selected_local]
    group_selected = np.array(
        [group_map.get(name, "unassigned") for name in class_selected],
        dtype=object,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    n_components = min(args.pca_components, X_scaled.shape[0], X_scaled.shape[1])
    pca_model = PCA(n_components=n_components, random_state=args.seed)
    pca_features = pca_model.fit_transform(X_scaled)
    pca_cols = [f"pca{i + 1}" for i in range(n_components)]

    df = pd.DataFrame(pca_features, columns=pca_cols)
    df.insert(0, "index", selected_idx)
    df.insert(1, "class", class_selected)
    df.insert(2, "group", group_selected)
    df.to_csv(os.path.join(out_dir, "embedding_points_pca.csv"), index=False)
    pd.DataFrame({
        "component": np.arange(1, n_components + 1),
        "explained_variance_ratio": pca_model.explained_variance_ratio_,
    }).to_csv(os.path.join(out_dir, "pca_explained_variance.csv"), index=False)

    save_scatter(
        df, "pca1", "pca2", "class",
        os.path.join(out_dir, "pca_by_class.png"),
        f"PCA by class ({args.target}, max_per_class={args.max_per_class})",
        normal_label="benign", label_order=class_order,
    )
    save_scatter(
        df, "pca1", "pca2", "group",
        os.path.join(out_dir, "pca_by_group.png"),
        f"PCA by expert-ownership group ({args.target})",
        normal_label="benign", label_order=group_order,
    )

    class_centers, _ = save_centroid_distances(
        df, "class", pca_cols, out_dir, "pca_class"
    )
    save_centroid_distances(df, "group", pca_cols, out_dir, "pca_group")
    save_class_dispersion(df, pca_cols, class_centers, out_dir, "benign")

    if not args.run_tsne:
        print(f"Saved embedding analysis to {out_dir}")
        print("Skipped t-SNE. Re-run with --run-tsne to generate t-SNE plots.")
        return

    if len(df) > args.max_tsne_samples:
        rng = np.random.default_rng(args.seed + 17)
        tsne_rows = np.sort(
            rng.choice(len(df), size=args.max_tsne_samples, replace=False)
        )
    else:
        tsne_rows = np.arange(len(df))
    tsne = TSNE(
        n_components=2, perplexity=30, init="pca", learning_rate="auto",
        random_state=args.seed,
    )
    tsne_xy = tsne.fit_transform(pca_features[tsne_rows])
    tsne_df = df.iloc[tsne_rows].copy()
    tsne_df["tsne1"] = tsne_xy[:, 0]
    tsne_df["tsne2"] = tsne_xy[:, 1]
    tsne_df.to_csv(os.path.join(out_dir, "embedding_points_tsne.csv"), index=False)
    save_scatter(
        tsne_df, "tsne1", "tsne2", "class",
        os.path.join(out_dir, "tsne_by_class.png"),
        f"t-SNE by class ({args.target}, n={len(tsne_df)})",
        normal_label="benign", label_order=class_order,
    )
    save_scatter(
        tsne_df, "tsne1", "tsne2", "group",
        os.path.join(out_dir, "tsne_by_group.png"),
        f"t-SNE by expert-ownership group ({args.target}, n={len(tsne_df)})",
        normal_label="benign", label_order=group_order,
    )
    print(f"Saved embedding analysis to {out_dir}")


if __name__ == "__main__":
    main()
