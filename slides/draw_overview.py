"""
Overview figure: Long-tail MoE + TTA + OOD pipeline.
Saves slides/overview.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
EXPERT_COLORS = [
    "#4878CF",  # Exp1 Benign     – blue
    "#E84545",  # Exp2 DoS        – red
    "#2CA02C",  # Exp3 DDoS/Scan  – green
    "#FF7F0E",  # Exp4 BruteForce – orange
    "#9467BD",  # Exp5 WebBot     – purple
    "#8C564B",  # Exp6 Tail       – brown
]
FADED_ALPHA  = 0.18
ACTIVE_ALPHA = 0.90
BG           = "#F7F7F7"

# ── class sizes (approximate CIC-2017 proportions, log-scaled for visual) ────
class_names = [
    "Benign", "DoS-Hulk", "PortScan", "DDoS",
    "DoS-GoldenEye", "FTP-Patator", "SSH-Patator", "DoS-Slowloris",
    "DoS-SlowHTTP", "Bot", "Web-BF", "Web-XSS",
    "Infiltration", "Web-SQLi", "Heartbleed",
]
counts_raw = np.array([
    2273097, 230124, 158930, 128027,
    10293,   7938,   5897,   5796,
    5499,    1966,   1507,   652,
    36,      21,     11,
], dtype=float)
counts = counts_raw / counts_raw.max()   # normalise to [0,1]
n_cls = len(class_names)

# expert assignment (mirrors PARTITIONS_2017 in code)
# index → expert index
ASSIGN = {
    0:  0,   # Benign
    1:  1, 4:  1, 7:  1, 8:  1,   # DoS family
    2:  2, 3:  2,                  # DDoS / PortScan
    5:  3, 6:  3,                  # BruteForce
    9:  4, 10: 4, 11: 4,           # WebBot
    12: 5, 13: 5, 14: 5,           # Tail
}
n_exp = 6
exp_labels = ["Exp 1\n(Benign)", "Exp 2\n(DoS)",
              "Exp 3\n(DDoS/Scan)", "Exp 4\n(BruteForce)",
              "Exp 5\n(WebBot)", "Exp 6\n(Tail)"]

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8), facecolor="white")

# four columns: [bar chart | experts | TTA box | OOD note]
ax_bar  = fig.add_axes([0.02, 0.12, 0.26, 0.78])   # long-tail distribution
ax_void = fig.add_axes([0.30, 0.00, 0.36, 1.00])   # expert boxes (axis off)
ax_tta  = fig.add_axes([0.68, 0.00, 0.17, 1.00])   # TTA box (axis off)
ax_ood  = fig.add_axes([0.86, 0.00, 0.13, 1.00])   # OOD note (axis off)
ax_void.axis("off")
ax_tta.axis("off")
ax_ood.axis("off")

# ── (A) Long-tail bar chart ───────────────────────────────────────────────────
ax_bar.set_facecolor(BG)
x = np.arange(n_cls)
for i in range(n_cls):
    exp_idx = ASSIGN[i]
    color   = EXPERT_COLORS[exp_idx]
    ax_bar.bar(i, counts[i], color=color, alpha=ACTIVE_ALPHA,
               edgecolor="white", linewidth=0.4)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(class_names, rotation=55, ha="right", fontsize=6.5)
ax_bar.set_ylabel("Relative frequency (log)", fontsize=8)
ax_bar.set_yscale("log")
ax_bar.set_ylim(1e-4, 2)
ax_bar.set_title("Data\n(Long-tail distribution)", fontsize=9, fontweight="bold")
ax_bar.spines[["top","right"]].set_visible(False)

# expert color legend inside bar chart
legend_patches = [mpatches.Patch(color=EXPERT_COLORS[e], alpha=ACTIVE_ALPHA,
                                  label=exp_labels[e].replace("\n", " "))
                  for e in range(n_exp)]
ax_bar.legend(handles=legend_patches, fontsize=6.2, loc="upper right",
              framealpha=0.85, title="Expert", title_fontsize=6.5)

# ── (B) Expert boxes (ax_void) ────────────────────────────────────────────────
# draw n_exp boxes vertically
box_w, box_h = 0.42, 0.10
box_x = 0.05          # left edge inside ax_void
gap   = (1.0 - n_exp * box_h) / (n_exp + 1)

for e in range(n_exp):
    y_bot = 1.0 - (e + 1) * (box_h + gap)
    fc = EXPERT_COLORS[e]

    fancy = FancyBboxPatch((box_x, y_bot), box_w, box_h,
                            boxstyle="round,pad=0.01",
                            linewidth=1.2,
                            edgecolor=fc, facecolor=fc + "33",
                            transform=ax_void.transAxes, zorder=2)
    ax_void.add_patch(fancy)

    # mini long-tail inside each expert box
    sub_x = np.linspace(box_x + 0.02, box_x + box_w - 0.02, n_cls)
    bar_w = 0.012
    for i in range(n_cls):
        is_own = ASSIGN[i] == e
        alpha  = ACTIVE_ALPHA if is_own else FADED_ALPHA
        h      = counts[i] * (box_h * 0.65)
        rect   = plt.Rectangle((sub_x[i] - bar_w / 2, y_bot + 0.01),
                                bar_w, h,
                                color=EXPERT_COLORS[e], alpha=alpha,
                                transform=ax_void.transAxes, zorder=3)
        ax_void.add_patch(rect)

    # label
    ax_void.text(box_x + box_w + 0.03,
                 y_bot + box_h / 2,
                 exp_labels[e],
                 ha="left", va="center", fontsize=8.5, fontweight="bold",
                 color=EXPERT_COLORS[e],
                 transform=ax_void.transAxes)

ax_void.set_title("Expert Partition\n(taxonomy-based, non-overlapping)",
                  fontsize=9, fontweight="bold", pad=4)

# arrow: bar chart → expert boxes  (in figure coords)
fig.patches.append(FancyArrowPatch(
    (0.285, 0.50), (0.305, 0.50),
    arrowstyle="->", mutation_scale=18,
    color="#444", linewidth=1.8,
    transform=fig.transFigure, zorder=10))

# ── (C) TTA aggregation box ───────────────────────────────────────────────────
tta_box = FancyBboxPatch((0.07, 0.25), 0.80, 0.50,
                          boxstyle="round,pad=0.02",
                          linewidth=2, edgecolor="#444",
                          facecolor="#EEF4FF",
                          transform=ax_tta.transAxes, zorder=2)
ax_tta.add_patch(tta_box)

ax_tta.text(0.47, 0.82, "Aggregation", ha="center", va="center",
            fontsize=10, fontweight="bold", transform=ax_tta.transAxes)
ax_tta.text(0.47, 0.68, "(TTA)", ha="center", va="center",
            fontsize=9, color="#2255AA", transform=ax_tta.transAxes)

# stability formula
ax_tta.text(0.47, 0.54,
            r"$s_e = \frac{1}{1+\mathrm{JS}(P_e(x)\|P_e(x'))}$",
            ha="center", va="center", fontsize=8,
            transform=ax_tta.transAxes)
ax_tta.text(0.47, 0.41,
            r"$w_e = \frac{s_e}{\sum_{e'} s_{e'}}$",
            ha="center", va="center", fontsize=8,
            transform=ax_tta.transAxes)
ax_tta.text(0.47, 0.29,
            r"$\hat{y} = \arg\max \sum_e w_e P_e^{\mathrm{global}}(x)$",
            ha="center", va="center", fontsize=7.5,
            transform=ax_tta.transAxes)

ax_tta.text(0.47, 0.10,
            "⚠ weights collapse\nto uniform (≈1/E)",
            ha="center", va="center", fontsize=7.5,
            color="#CC3300", style="italic",
            transform=ax_tta.transAxes)

ax_tta.set_title("Aggregation\n(TTA)", fontsize=9, fontweight="bold", pad=4)

# arrow: expert → TTA
fig.patches.append(FancyArrowPatch(
    (0.665, 0.50), (0.685, 0.50),
    arrowstyle="->", mutation_scale=18,
    color="#444", linewidth=1.8,
    transform=fig.transFigure, zorder=10))

# ── (D) OOD overconfidence note ───────────────────────────────────────────────
ood_box = FancyBboxPatch((0.04, 0.20), 0.90, 0.60,
                          boxstyle="round,pad=0.02",
                          linewidth=2, edgecolor="#CC3300",
                          facecolor="#FFF3F0",
                          transform=ax_ood.transAxes, zorder=2)
ax_ood.add_patch(ood_box)

ax_ood.text(0.50, 0.90, "OOD", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#CC3300",
            transform=ax_ood.transAxes)
ax_ood.text(0.50, 0.78, "Overconfidence", ha="center", va="center",
            fontsize=8, color="#CC3300",
            transform=ax_ood.transAxes)

ax_ood.text(0.50, 0.63,
            "Non-owning expert:\nwrong class,\nhigh confidence",
            ha="center", va="center", fontsize=7.5,
            transform=ax_ood.transAxes)

ax_ood.text(0.50, 0.44, "→ Plan:", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#555",
            transform=ax_ood.transAxes)
ax_ood.text(0.50, 0.33,
            "Train experts to\nrecognize OOD inputs\n& express uncertainty",
            ha="center", va="center", fontsize=7,
            color="#333", transform=ax_ood.transAxes)

ax_ood.text(0.50, 0.13, "(to be attempted)",
            ha="center", va="center", fontsize=7,
            style="italic", color="#888",
            transform=ax_ood.transAxes)

ax_ood.set_title("OOD\n(planned)", fontsize=9, fontweight="bold",
                 color="#CC3300", pad=4)

# dashed arrow: expert → OOD (diagonal)
fig.patches.append(FancyArrowPatch(
    (0.66, 0.72), (0.87, 0.72),
    arrowstyle="->", mutation_scale=14,
    color="#CC3300", linewidth=1.4, linestyle="dashed",
    transform=fig.transFigure, zorder=10))

# ── global title ──────────────────────────────────────────────────────────────
fig.text(0.50, 0.97,
         "Overview: MoE + TTA + OOD for Long-tail Network Intrusion Detection",
         ha="center", va="top", fontsize=12, fontweight="bold")

out = "/home/user/Desktop/imbal_cic/slides/overview.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
