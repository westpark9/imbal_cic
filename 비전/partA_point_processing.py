# -*- coding: utf-8 -*-
"""
Homework 1 (Computer Vision) - Part A
Point Processing and Histogram-Based Image Enhancement  (from scratch)

Core image-processing operations (color-space conversion, histogram / equalization,
contrast stretching, gamma, AHE, CLAHE, bit-plane slicing) are implemented from
scratch, i.e. without built-in functions that directly perform the operation.
Allowed built-ins: numpy (basic array ops), PIL (image loading), matplotlib (display).

Run:
    python partA_point_processing.py
-> all figures are saved to output/ as A1..A9, and quantitative results are printed.

Parameters (summary):
  - RGB->YUV matrix: BT.601-style (given in the assignment)
  - histogram levels L = 256 (8-bit)
  - contrast stretching: 2nd / 98th percentiles
  - gamma: gamma = 0.5, 1.0, 2.0 (c = 1)
  - AHE / CLAHE tiles: 8x8,  CLAHE clip = 0.01, 0.05 (fraction of tile pixel count)
  - bit-plane: reconstruct from the 4 most significant planes (bit 4..7)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless: save to files only (no window)
import matplotlib.pyplot as plt
from PIL import Image

IMG_PATH = os.path.join("images", "point_processing_input_rgb.png")
OUT_DIR = "output"
L = 256

# RGB->YUV matrix given in the assignment (BT.601-style)
M_FWD = np.array([[ 0.299,    0.587,    0.114  ],
                  [-0.14713, -0.28886,  0.436  ],
                  [ 0.615,   -0.51499, -0.10001]])
M_INV = np.linalg.inv(M_FWD)          # inverse transform (basic array op)


# =====================================================================
# Common utilities
# =====================================================================
def show(ax, img, title="", cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def to_uint8(img):
    """For display only: clip to [0,255] and cast to uint8 (not used in computation)."""
    return np.clip(np.round(img), 0, 255).astype(np.uint8)

def quantize8(img):
    return np.clip(np.round(img), 0, 255).astype(np.int64)

def save(fig, tag):
    path = os.path.join(OUT_DIR, tag + ".png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path

def mse(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def psnr(a, b, peak=255.0):
    m = mse(a, b)
    return float("inf") if m == 0 else 10.0 * np.log10(peak * peak / m)


# =====================================================================
# Core operations (from scratch)
# =====================================================================
def rgb_to_yuv(img):
    """img:(H,W,3) float -> Y,U,V each (H,W) float."""
    flat = img.reshape(-1, 3).T
    yuv = (M_FWD @ flat).T.reshape(img.shape)
    return yuv[..., 0], yuv[..., 1], yuv[..., 2]

def yuv_to_rgb(Y, U, V):
    yuv = np.stack([Y, U, V], axis=-1)
    flat = yuv.reshape(-1, 3).T
    return (M_INV @ flat).T.reshape(yuv.shape)

def compute_hist(q):
    """No built-in histogram: count per level directly. h[k] = #(pixels == k)."""
    h = np.zeros(256, dtype=np.float64)
    for k in range(256):
        h[k] = np.sum(q == k)
    return h

def hist_equalize(q):
    """s_k = (L-1)*CDF(r_k). Returns (equalized image, LUT, CDF)."""
    h = compute_hist(q)
    cdf = np.cumsum(h) / h.sum()
    lut = np.round((L - 1) * cdf)
    return lut[q], lut, cdf

def contrast_stretch(img, p_low=2, p_high=98):
    """Piecewise-linear stretch with r_min,r_max at the 2nd/98th percentiles."""
    rmin = np.percentile(img, p_low)
    rmax = np.percentile(img, p_high)
    s = (img - rmin) / (rmax - rmin) * (L - 1)
    return np.clip(s, 0, L - 1), rmin, rmax

def gamma_correct(img, gamma):
    """s = 255 * (r/255)^gamma, c = 1."""
    rn = np.clip(img / (L - 1), 0, 1)
    return np.power(rn, gamma) * (L - 1)

def pad_to_multiple(img, nt):
    """Edge-replicate pad so the image divides evenly into nt tiles. Returns (padded, H0, W0)."""
    H, W = img.shape
    Hs = int(np.ceil(H / nt) * nt); Ws = int(np.ceil(W / nt) * nt)
    return np.pad(img, ((0, Hs - H), (0, Ws - W)), mode="edge"), H, W

def ahe(img, nt=8):
    """Adaptive HE: independent HE per tile (no interpolation)."""
    q, H0, W0 = pad_to_multiple(quantize8(img), nt)
    H, W = q.shape; th, tw = H // nt, W // nt
    out = np.zeros_like(q, dtype=np.float64)
    for ti in range(nt):
        for tj in range(nt):
            tile = q[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw]
            h = compute_hist(tile)
            lut = np.round((L - 1) * (np.cumsum(h) / h.sum()))
            out[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw] = lut[tile]
    return out[:H0, :W0]

def clahe(img, nt=8, clip=0.01):
    """CLAHE: clip + redistribute tile histogram -> local CDF -> bilinear interpolation."""
    q, H0, W0 = pad_to_multiple(quantize8(img), nt)
    H, W = q.shape; th, tw = H // nt, W // nt
    npix = th * tw
    clip_count = max(1, int(clip * npix))

    # per-tile LUT
    luts = np.zeros((nt, nt, 256))
    for ti in range(nt):
        for tj in range(nt):
            tile = q[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw]
            h = compute_hist(tile)
            excess = np.maximum(h - clip_count, 0).sum()      # total clipped amount
            h = np.minimum(h, clip_count)                     # clip
            h += excess / 256.0                               # redistribute uniformly
            luts[ti, tj] = np.round((L - 1) * (np.cumsum(h) / h.sum()))

    # bilinear interpolation across tile centers (vectorized)
    cy = (np.arange(nt) + 0.5) * th
    cx = (np.arange(nt) + 0.5) * tw
    yy = np.clip((np.arange(H) - cy[0]) / th, 0, nt - 1)
    xx = np.clip((np.arange(W) - cx[0]) / tw, 0, nt - 1)
    i0 = np.clip(np.floor(yy).astype(int), 0, nt - 1); i1 = np.clip(i0 + 1, 0, nt - 1)
    j0 = np.clip(np.floor(xx).astype(int), 0, nt - 1); j1 = np.clip(j0 + 1, 0, nt - 1)
    wy = (yy - i0)[:, None]; wx = (xx - j0)[None, :]
    I0 = np.broadcast_to(i0[:, None], (H, W)); I1 = np.broadcast_to(i1[:, None], (H, W))
    J0 = np.broadcast_to(j0[None, :], (H, W)); J1 = np.broadcast_to(j1[None, :], (H, W))
    A = luts[I0, J0, q]; B = luts[I0, J1, q]; Cc = luts[I1, J0, q]; D = luts[I1, J1, q]
    out = (1 - wy) * ((1 - wx) * A + wx * B) + wy * ((1 - wx) * Cc + wx * D)
    return out[:H0, :W0]


# =====================================================================
# Run experiments + save figures + collect quantitative results
# =====================================================================
def run_all(img_path=IMG_PATH, make_a9_thumbs=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
    Mh, Nh = rgb.shape[:2]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    res = {"figures": {}, "metrics": {}}
    met = res["metrics"]
    met["shape"] = (Mh, Nh)

    # ---- A1. RGB -> YUV ----
    Y, U, V = rgb_to_yuv(rgb)
    met["Y_range"] = (float(Y.min()), float(Y.max()))
    met["U_range"] = (float(U.min()), float(U.max()))
    met["V_range"] = (float(V.min()), float(V.max()))
    met["corr_U_BmY"] = float(np.corrcoef(U.ravel(), (B - Y).ravel())[0, 1])
    met["corr_V_RmY"] = float(np.corrcoef(V.ravel(), (R - Y).ravel())[0, 1])
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    show(ax[0], to_uint8(rgb), "Original RGB", cmap=None)
    show(ax[1], Y, "Y (luminance)", vmin=0, vmax=255)
    show(ax[2], U, "U (blue-diff, B-Y)")
    show(ax[3], V, "V (red-diff, R-Y)")
    res["figures"]["A1"] = save(fig, "A1_rgb_yuv")

    # ---- A2. Histogram ----
    Yq = quantize8(Y)
    hist = compute_hist(Yq)
    p = hist / (Mh * Nh)
    met["hist_sum"] = float(hist.sum()); met["p_sum"] = float(p.sum())
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    show(ax[0], Yq, "Input luminance Y", vmin=0, vmax=255)
    ax[1].bar(np.arange(256), hist, width=1.0); ax[1].set_title("Histogram h(r_k)=n_k"); ax[1].set_xlabel("r_k")
    ax[2].bar(np.arange(256), p, width=1.0, color="C1"); ax[2].set_title("Normalized p(r_k)=n_k/MN"); ax[2].set_xlabel("r_k")
    res["figures"]["A2"] = save(fig, "A2_histogram")

    # ---- A3. Histogram Equalization ----
    Y_eq, lut_he, cdf_he = hist_equalize(Yq)
    hist_eq = compute_hist(quantize8(Y_eq))
    met["HE_mean_before"] = float(Yq.mean()); met["HE_mean_after"] = float(Y_eq.mean())
    met["HE_std_before"] = float(Yq.std()); met["HE_std_after"] = float(Y_eq.std())
    rgb_eq = yuv_to_rgb(Y_eq, U, V)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    show(ax[0, 0], Yq, "Original Y", vmin=0, vmax=255)
    ax[0, 1].bar(np.arange(256), hist, width=1.0); ax[0, 1].set_title("Original histogram")
    show(ax[1, 0], Y_eq, "Equalized Y", vmin=0, vmax=255)
    ax[1, 1].bar(np.arange(256), hist_eq, width=1.0, color="C2"); ax[1, 1].set_title("Equalized histogram")
    res["figures"]["A3a"] = save(fig, "A3_1_Y_equalization")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    show(ax[0], to_uint8(rgb), "Original RGB", cmap=None)
    show(ax[1], to_uint8(rgb_eq), "HE-enhanced RGB (Y only)", cmap=None)
    ax[2].plot(np.arange(256), lut_he); ax[2].plot(np.arange(256), np.arange(256), "k--", lw=0.8)
    ax[2].set_title("HE transform s=T(r)"); ax[2].set_xlabel("r"); ax[2].set_ylabel("s"); ax[2].grid(True)
    res["figures"]["A3b"] = save(fig, "A3_2_RGB_compare")

    # ---- A4. Contrast Stretching ----
    Y_cs, rmin, rmax = contrast_stretch(Y, 2, 98)
    met["cs_rmin"] = float(rmin); met["cs_rmax"] = float(rmax)
    met["cs_gain"] = float((L - 1) / (rmax - rmin))
    hist_cs = compute_hist(quantize8(Y_cs))
    rgb_cs = yuv_to_rgb(Y_cs, U, V)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    show(ax[0, 0], to_uint8(rgb), "Original RGB", cmap=None)
    show(ax[0, 1], to_uint8(rgb_cs), "Contrast-stretched RGB", cmap=None)
    ax[1, 0].bar(np.arange(256), hist, width=1.0); ax[1, 0].set_title("Histogram (before)")
    ax[1, 1].bar(np.arange(256), hist_cs, width=1.0, color="C3"); ax[1, 1].set_title("Histogram (after)")
    res["figures"]["A4"] = save(fig, "A4_contrast_stretch")

    # ---- A5. Gamma Correction ----
    gammas = [0.5, 1.0, 2.0]
    fig, ax = plt.subplots(2, len(gammas), figsize=(13, 8))
    rr = np.arange(256)
    for i, g in enumerate(gammas):
        Yg = gamma_correct(Y, g)
        rgbg = yuv_to_rgb(Yg, U, V)
        show(ax[0, i], to_uint8(rgbg), "gamma = %.1f" % g, cmap=None)
        ax[1, i].plot(rr, 255 * (rr / 255.0) ** g); ax[1, i].plot(rr, rr, "k--", lw=0.8)
        ax[1, i].set_title("s=T(r), gamma=%.1f" % g); ax[1, i].set_xlabel("r"); ax[1, i].set_ylabel("s"); ax[1, i].grid(True)
    res["figures"]["A5"] = save(fig, "A5_gamma")

    # ---- A6. AHE ----
    Y_ahe = ahe(Y, nt=8)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    show(ax[0], Yq, "Original Y", vmin=0, vmax=255)
    show(ax[1], Y_eq, "Global HE", vmin=0, vmax=255)
    show(ax[2], Y_ahe, "AHE (8x8 tiles)", vmin=0, vmax=255)
    res["figures"]["A6"] = save(fig, "A6_AHE")

    # ---- A7. CLAHE ----
    Y_clahe1 = clahe(Y, nt=8, clip=0.01)
    Y_clahe2 = clahe(Y, nt=8, clip=0.05)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    show(ax[0, 0], Yq, "Original Y", vmin=0, vmax=255)
    show(ax[0, 1], Y_ahe, "AHE", vmin=0, vmax=255)
    show(ax[0, 2], Y_clahe1, "CLAHE clip=0.01", vmin=0, vmax=255)
    ax[1, 0].bar(np.arange(256), compute_hist(quantize8(Y_ahe)), width=1.0); ax[1, 0].set_title("Hist: AHE")
    ax[1, 1].bar(np.arange(256), compute_hist(quantize8(Y_clahe1)), width=1.0, color="C2"); ax[1, 1].set_title("Hist: CLAHE 0.01")
    show(ax[1, 2], Y_clahe2, "CLAHE clip=0.05", vmin=0, vmax=255)
    res["figures"]["A7"] = save(fig, "A7_CLAHE")

    # CLAHE auxiliary metric: mapping of input 0 in the darkest tile (clip effect)
    nt = 8; th = tw = Mh // nt; npix = th * tw
    cand = [(np.mean(Yq[i*th:(i+1)*th, j*tw:(j+1)*tw] <= 2), i, j) for i in range(nt) for j in range(nt)]
    frac0, ti, tj = max(cand)
    hti = compute_hist(Yq[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw])
    def _lut0(h, cc):
        ex = np.maximum(h - cc, 0).sum()
        hc = np.minimum(h, cc) + ex / 256.0
        return int(np.round((L - 1) * np.cumsum(hc) / hc.sum())[0])
    met["clahe_tile"] = (int(ti), int(tj))
    met["clahe_tile_zero_count"] = int(hti[0])
    met["clahe_tile_zero_frac"] = float(hti[0] / npix)   # Y==0 fraction (same basis as count)
    met["clahe_map0_ahe"] = int(np.round((L - 1) * np.cumsum(hti) / hti.sum())[0])
    met["clahe_map0_clip01"] = _lut0(hti, max(1, int(0.01 * npix)))
    met["clahe_map0_clip05"] = _lut0(hti, max(1, int(0.05 * npix)))

    # ---- A8. Bit-Plane Slicing ----
    Yb = quantize8(Y).astype(np.uint8)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
    for k in range(8):
        bk = (Yb >> k) & 1
        r_, c_ = divmod(7 - k, 4)
        lbl = " (MSB)" if k == 7 else (" (LSB)" if k == 0 else "")
        show(ax[r_, c_], bk, "bit plane %d%s" % (k, lbl), vmin=0, vmax=1)
    res["figures"]["A8a"] = save(fig, "A8_1_bit_planes")
    recon4 = np.zeros_like(Yb, dtype=np.float64)
    for k in range(4, 8):
        recon4 += ((Yb >> k) & 1).astype(np.float64) * (2 ** k)
    met["bitplane_MSE"] = mse(Yb, recon4)
    met["bitplane_PSNR"] = psnr(Yb, recon4, 255)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    show(ax[0], Yb, "Original Y (8 bits)", vmin=0, vmax=255)
    show(ax[1], recon4, "Reconstructed (4 MSB planes)", vmin=0, vmax=255)
    res["figures"]["A8b"] = save(fig, "A8_2_reconstruction_4MSB")

    # ---- A9. comparison thumbnails (result image + histogram) ----
    if make_a9_thumbs:
        Y_g05 = gamma_correct(Y, 0.5)
        thumbs = [("orig", Yq), ("stretch", Y_cs), ("he", Y_eq),
                  ("ahe", Y_ahe), ("clahe", Y_clahe1), ("gamma", Y_g05)]
        res["a9_thumbs"] = {}
        for key, img in thumbs:
            fig = plt.figure(figsize=(2.0, 2.0))
            plt.imshow(img, cmap="gray", vmin=0, vmax=255); plt.axis("off")
            pth_i = os.path.join(OUT_DIR, "A9_%s_img.png" % key)
            fig.savefig(pth_i, dpi=90, bbox_inches="tight", pad_inches=0); plt.close(fig)
            fig = plt.figure(figsize=(2.6, 1.9))
            plt.bar(np.arange(256), compute_hist(quantize8(img)), width=1.0, color="C0")
            plt.xlim(0, 255); plt.yticks([]); plt.xticks([0, 128, 255], fontsize=7)
            pth_h = os.path.join(OUT_DIR, "A9_%s_hist.png" % key)
            fig.savefig(pth_h, dpi=90, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
            res["a9_thumbs"][key] = (pth_i, pth_h)

    return res


def _print_metrics(m):
    print("=" * 60)
    print("Part A  Quantitative results")
    print("=" * 60)
    print("input size       : %dx%d" % m["shape"])
    print("Y range          : [%.2f, %.2f]" % m["Y_range"])
    print("U range          : [%.2f, %.2f]" % m["U_range"])
    print("V range          : [%.2f, %.2f]" % m["V_range"])
    print("corr(U, B-Y)     : %.3f   corr(V, R-Y): %.3f" % (m["corr_U_BmY"], m["corr_V_RmY"]))
    print("sum(h)=%.0f  sum(p)=%.4f" % (m["hist_sum"], m["p_sum"]))
    print("HE  Y mean %.1f -> %.1f ,  std %.1f -> %.1f"
          % (m["HE_mean_before"], m["HE_mean_after"], m["HE_std_before"], m["HE_std_after"]))
    print("Contrast stretch : r_min=%.1f r_max=%.1f  gain=%.3f" % (m["cs_rmin"], m["cs_rmax"], m["cs_gain"]))
    print("Bit-plane(4 MSB) : MSE=%.2f  PSNR=%.2f dB" % (m["bitplane_MSE"], m["bitplane_PSNR"]))
    print("CLAHE darkest tile %s: Y=0 count %d (%.0f%%),  input 0 -> output  AHE %d / clip0.01 %d / clip0.05 %d"
          % (m["clahe_tile"], m["clahe_tile_zero_count"], 100 * m["clahe_tile_zero_frac"],
             m["clahe_map0_ahe"], m["clahe_map0_clip01"], m["clahe_map0_clip05"]))
    print("=" * 60)


if __name__ == "__main__":
    result = run_all()
    _print_metrics(result["metrics"])
    print("figures saved to '%s/' :" % OUT_DIR)
    for k, v in result["figures"].items():
        print("  [%s] %s" % (k, v))
