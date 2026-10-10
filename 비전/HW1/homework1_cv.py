# -*- coding: utf-8 -*-
"""Computer Vision Homework 1: Parts A, B and C, implemented from scratch.

Run all experiments: python homework1_cv.py
Run one part:        python homework1_cv.py --part B

Inputs: images/ beside this file.
Figures: output/; measurements: results_A.json, results_B.json, results_C.json.
Only basic NumPy array operations, FFT/IFFT, image loading and plotting are used.
Part-specific helper prefixes keep the three implementations easy to locate.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = str(BASE_DIR / "output")


# =====================================================================
# PART A
# =====================================================================


A_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "point_processing_input_rgb.png")
A_L = 256

# RGB->YUV matrix given in the assignment (BT.601-style)
A_M_FWD = np.array([[ 0.299,    0.587,    0.114  ],
                  [-0.14713, -0.28886,  0.436  ],
                  [ 0.615,   -0.51499, -0.10001]])
A_M_INV = np.linalg.inv(A_M_FWD)          # inverse transform (basic array op)


# =====================================================================
# Common utilities
# =====================================================================
def a_show(ax, img, title="", cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def a_to_uint8(img):
    """For display only: clip to [0,255] and cast to uint8 (not used in computation)."""
    return np.clip(np.round(img), 0, 255).astype(np.uint8)

def a_quantize8(img):
    return np.clip(np.round(img), 0, 255).astype(np.int64)

def a_save(fig, tag):
    path = os.path.join(OUT_DIR, tag + ".png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path

def a_mse(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def a_psnr(a, b, peak=255.0):
    m = a_mse(a, b)
    return float("inf") if m == 0 else 10.0 * np.log10(peak * peak / m)


# =====================================================================
# Core operations (from scratch)
# =====================================================================
def a_rgb_to_yuv(img):
    """img:(H,W,3) float -> Y,U,V each (H,W) float."""
    flat = img.reshape(-1, 3).T
    yuv = (A_M_FWD @ flat).T.reshape(img.shape)
    return yuv[..., 0], yuv[..., 1], yuv[..., 2]

def a_yuv_to_rgb(Y, U, V):
    yuv = np.stack([Y, U, V], axis=-1)
    flat = yuv.reshape(-1, 3).T
    return (A_M_INV @ flat).T.reshape(yuv.shape)

def a_compute_hist(q):
    """No built-in histogram: count per level directly. h[k] = #(pixels == k)."""
    h = np.zeros(256, dtype=np.float64)
    for k in range(256):
        h[k] = np.sum(q == k)
    return h

def a_hist_equalize(q):
    """s_k = (L-1)*CDF(r_k). Returns (equalized image, LUT, CDF)."""
    h = a_compute_hist(q)
    cdf = np.cumsum(h) / h.sum()
    lut = np.round((A_L - 1) * cdf)
    return lut[q], lut, cdf

def a_contrast_stretch(img, p_low=2, p_high=98):
    """Choose percentile levels from the 8-bit luminance histogram CDF."""
    if not 0 <= p_low < p_high <= 100:
        raise ValueError("Require 0 <= p_low < p_high <= 100")
    h = a_compute_hist(a_quantize8(img)); cdf = np.cumsum(h) / h.sum()
    rmin = float(np.flatnonzero(cdf >= p_low / 100.0)[0])
    rmax = float(np.flatnonzero(cdf >= p_high / 100.0)[0])
    if rmax <= rmin:
        return img.astype(np.float64).copy(), rmin, rmax
    return np.clip((img-rmin)/(rmax-rmin)*(A_L-1), 0, A_L-1), rmin, rmax

def a_gamma_correct(img, gamma):
    """s = 255 * (r/255)^gamma, c = 1."""
    rn = np.clip(img / (A_L - 1), 0, 1)
    return np.power(rn, gamma) * (A_L - 1)

def a_pad_to_multiple(img, nt):
    """Edge-replicate pad so the image divides evenly into nt tiles. Returns (padded, H0, W0)."""
    H, W = img.shape
    Hs = int(np.ceil(H / nt) * nt); Ws = int(np.ceil(W / nt) * nt)
    return np.pad(img, ((0, Hs - H), (0, Ws - W)), mode="edge"), H, W

def a_ahe(img, nt=8):
    """Adaptive HE: independent HE per tile (no interpolation)."""
    q, H0, W0 = a_pad_to_multiple(a_quantize8(img), nt)
    H, W = q.shape; th, tw = H // nt, W // nt
    out = np.zeros_like(q, dtype=np.float64)
    for ti in range(nt):
        for tj in range(nt):
            tile = q[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw]
            h = a_compute_hist(tile)
            lut = np.round((A_L - 1) * (np.cumsum(h) / h.sum()))
            out[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw] = lut[tile]
    return out[:H0, :W0]

def a_clahe(img, nt=8, clip=0.01):
    """CLAHE: clip + redistribute tile histogram -> local CDF -> bilinear interpolation."""
    q, H0, W0 = a_pad_to_multiple(a_quantize8(img), nt)
    H, W = q.shape; th, tw = H // nt, W // nt
    npix = th * tw
    clip_count = max(1, int(clip * npix))

    # per-tile LUT
    luts = np.zeros((nt, nt, 256))
    for ti in range(nt):
        for tj in range(nt):
            tile = q[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw]
            h = a_compute_hist(tile)
            excess = np.maximum(h - clip_count, 0).sum()      # total clipped amount
            h = np.minimum(h, clip_count)                     # clip
            h += excess / 256.0                               # redistribute uniformly
            luts[ti, tj] = np.round((A_L - 1) * (np.cumsum(h) / h.sum()))

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
def run_part_a(img_path=A_IMG_PATH, make_a9_thumbs=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
    Mh, Nh = rgb.shape[:2]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    res = {"figures": {}, "metrics": {}}
    met = res["metrics"]
    met["shape"] = (Mh, Nh)

    # ---- A1. RGB -> YUV ----
    Y, U, V = a_rgb_to_yuv(rgb)
    met["Y_range"] = (float(Y.min()), float(Y.max()))
    met["U_range"] = (float(U.min()), float(U.max()))
    met["V_range"] = (float(V.min()), float(V.max()))
    met["corr_U_BmY"] = float(np.corrcoef(U.ravel(), (B - Y).ravel())[0, 1])
    met["corr_V_RmY"] = float(np.corrcoef(V.ravel(), (R - Y).ravel())[0, 1])
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    a_show(ax[0], a_to_uint8(rgb), "Original RGB", cmap=None)
    a_show(ax[1], Y, "Y (luminance)", vmin=0, vmax=255)
    a_show(ax[2], U, "U (blue-diff, B-Y)")
    a_show(ax[3], V, "V (red-diff, R-Y)")
    res["figures"]["A1"] = a_save(fig, "A1_rgb_yuv")

    # ---- A2. Histogram ----
    Yq = a_quantize8(Y)
    hist = a_compute_hist(Yq)
    p = hist / (Mh * Nh)
    met["hist_sum"] = float(hist.sum()); met["p_sum"] = float(p.sum())
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    a_show(ax[0], Yq, "Input luminance Y", vmin=0, vmax=255)
    ax[1].bar(np.arange(256), hist, width=1.0); ax[1].set_title("Histogram h(r_k)=n_k"); ax[1].set_xlabel("r_k")
    ax[2].bar(np.arange(256), p, width=1.0, color="C1"); ax[2].set_title("Normalized p(r_k)=n_k/MN"); ax[2].set_xlabel("r_k")
    res["figures"]["A2"] = a_save(fig, "A2_histogram")

    # ---- A3. Histogram Equalization ----
    Y_eq, lut_he, cdf_he = a_hist_equalize(Yq)
    hist_eq = a_compute_hist(a_quantize8(Y_eq))
    met["HE_mean_before"] = float(Yq.mean()); met["HE_mean_after"] = float(Y_eq.mean())
    met["HE_std_before"] = float(Yq.std()); met["HE_std_after"] = float(Y_eq.std())
    rgb_eq = a_yuv_to_rgb(Y_eq, U, V)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a_show(ax[0, 0], Yq, "Original Y", vmin=0, vmax=255)
    ax[0, 1].bar(np.arange(256), hist, width=1.0); ax[0, 1].set_title("Original histogram")
    a_show(ax[1, 0], Y_eq, "Equalized Y", vmin=0, vmax=255)
    ax[1, 1].bar(np.arange(256), hist_eq, width=1.0, color="C2"); ax[1, 1].set_title("Equalized histogram")
    res["figures"]["A3a"] = a_save(fig, "A3_1_Y_equalization")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    a_show(ax[0], a_to_uint8(rgb), "Original RGB", cmap=None)
    a_show(ax[1], a_to_uint8(rgb_eq), "HE-enhanced RGB (Y only)", cmap=None)
    ax[2].plot(np.arange(256), lut_he); ax[2].plot(np.arange(256), np.arange(256), "k--", lw=0.8)
    ax[2].set_title("HE transform s=T(r)"); ax[2].set_xlabel("r"); ax[2].set_ylabel("s"); ax[2].grid(True)
    res["figures"]["A3b"] = a_save(fig, "A3_2_RGB_compare")

    # ---- A4. Contrast Stretching ----
    Y_cs, rmin, rmax = a_contrast_stretch(Y, 2, 98)
    met["cs_rmin"] = float(rmin); met["cs_rmax"] = float(rmax)
    met["cs_gain"] = float((A_L - 1) / (rmax - rmin))
    hist_cs = a_compute_hist(a_quantize8(Y_cs))
    rgb_cs = a_yuv_to_rgb(Y_cs, U, V)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a_show(ax[0, 0], a_to_uint8(rgb), "Original RGB", cmap=None)
    a_show(ax[0, 1], a_to_uint8(rgb_cs), "Contrast-stretched RGB", cmap=None)
    ax[1, 0].bar(np.arange(256), hist, width=1.0); ax[1, 0].set_title("Histogram (before)")
    ax[1, 1].bar(np.arange(256), hist_cs, width=1.0, color="C3"); ax[1, 1].set_title("Histogram (after)")
    res["figures"]["A4"] = a_save(fig, "A4_contrast_stretch")

    # ---- A5. Gamma Correction ----
    gammas = [0.5, 1.0, 2.0]
    fig, ax = plt.subplots(2, len(gammas), figsize=(13, 8))
    rr = np.arange(256)
    for i, g in enumerate(gammas):
        Yg = a_gamma_correct(Y, g)
        rgbg = a_yuv_to_rgb(Yg, U, V)
        a_show(ax[0, i], a_to_uint8(rgbg), "gamma = %.1f" % g, cmap=None)
        ax[1, i].plot(rr, 255 * (rr / 255.0) ** g); ax[1, i].plot(rr, rr, "k--", lw=0.8)
        ax[1, i].set_title("s=T(r), gamma=%.1f" % g); ax[1, i].set_xlabel("r"); ax[1, i].set_ylabel("s"); ax[1, i].grid(True)
    res["figures"]["A5"] = a_save(fig, "A5_gamma")

    # ---- A6. AHE ----
    Y_ahe = a_ahe(Y, nt=8)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    a_show(ax[0], Yq, "Original Y", vmin=0, vmax=255)
    a_show(ax[1], Y_eq, "Global HE", vmin=0, vmax=255)
    a_show(ax[2], Y_ahe, "AHE (8x8 tiles)", vmin=0, vmax=255)
    res["figures"]["A6"] = a_save(fig, "A6_AHE")

    # ---- A7. CLAHE ----
    Y_clahe1 = a_clahe(Y, nt=8, clip=0.01)
    Y_clahe2 = a_clahe(Y, nt=8, clip=0.05)
    images = [Yq, Y_ahe, Y_clahe1, Y_clahe2]
    titles = ["Original Y", "AHE", "CLAHE clip=0.01", "CLAHE clip=0.05"]
    hs = [a_compute_hist(a_quantize8(v)) for v in images]
    ymax = 1.05 * max(h.max() for h in hs)
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for j, (v, title, h) in enumerate(zip(images, titles, hs)):
        a_show(ax[0, j], v, title, vmin=0, vmax=255)
        ax[1, j].bar(np.arange(256), h, width=1)
        ax[1, j].set(xlim=(0,255), ylim=(0,ymax), title="Histogram: " + title,
                     xlabel="Intensity", ylabel="Pixel count")
    fig.tight_layout()
    res["figures"]["A7"] = a_save(fig, "A7_CLAHE")

    # Descriptive contrast statistics; these do not isolate image noise.
    met["enhancement_stats"] = []
    variants = [("Original", Y), ("Stretch", Y_cs), ("HE", Y_eq),
                ("Gamma 0.5", a_gamma_correct(Y,0.5)), ("Gamma 1.0", a_gamma_correct(Y,1.0)),
                ("Gamma 2.0", a_gamma_correct(Y,2.0)), ("AHE", Y_ahe),
                ("CLAHE 0.01", Y_clahe1), ("CLAHE 0.05", Y_clahe2)]
    for label, v in variants:
        tiles = [v[i*64:(i+1)*64,j*64:(j+1)*64] for i in range(8) for j in range(8)]
        met["enhancement_stats"].append([label, float(v.mean()), float(v.std()),
                                          float(np.mean([t.std() for t in tiles]))])

    # CLAHE auxiliary metric: mapping of input 0 in the darkest tile (clip effect)
    nt = 8; th = tw = Mh // nt; npix = th * tw
    cand = [(np.mean(Yq[i*th:(i+1)*th, j*tw:(j+1)*tw] <= 2), i, j) for i in range(nt) for j in range(nt)]
    frac0, ti, tj = max(cand)
    hti = a_compute_hist(Yq[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw])
    def _lut0(h, cc):
        ex = np.maximum(h - cc, 0).sum()
        hc = np.minimum(h, cc) + ex / 256.0
        return int(np.round((A_L - 1) * np.cumsum(hc) / hc.sum())[0])
    met["clahe_tile"] = (int(ti), int(tj))
    met["clahe_tile_zero_count"] = int(hti[0])
    met["clahe_tile_zero_frac"] = float(hti[0] / npix)   # Y==0 fraction (same basis as count)
    met["clahe_map0_ahe"] = int(np.round((A_L - 1) * np.cumsum(hti) / hti.sum())[0])
    met["clahe_map0_clip01"] = _lut0(hti, max(1, int(0.01 * npix)))
    met["clahe_map0_clip05"] = _lut0(hti, max(1, int(0.05 * npix)))

    # ---- A8. Bit-Plane Slicing ----
    Yb = a_quantize8(Y).astype(np.uint8)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
    for k in range(8):
        bk = (Yb >> k) & 1
        r_, c_ = divmod(7 - k, 4)
        lbl = " (MSB)" if k == 7 else (" (LSB)" if k == 0 else "")
        a_show(ax[r_, c_], bk, "bit plane %d%s" % (k, lbl), vmin=0, vmax=1)
    res["figures"]["A8a"] = a_save(fig, "A8_1_bit_planes")
    recon4 = np.zeros_like(Yb, dtype=np.float64)
    for k in range(4, 8):
        recon4 += ((Yb >> k) & 1).astype(np.float64) * (2 ** k)
    met["bitplane_MSE"] = a_mse(Yb, recon4)
    met["bitplane_PSNR"] = a_psnr(Yb, recon4, 255)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    a_show(ax[0], Yb, "Original Y (8 bits)", vmin=0, vmax=255)
    a_show(ax[1], recon4, "Reconstructed (4 MSB planes)", vmin=0, vmax=255)
    res["figures"]["A8b"] = a_save(fig, "A8_2_reconstruction_4MSB")

    # ---- A9. comparison thumbnails (result image + histogram) ----
    if make_a9_thumbs:
        Y_g05 = a_gamma_correct(Y, 0.5)
        thumbs = [("orig", Yq), ("stretch", Y_cs), ("he", Y_eq),
                  ("ahe", Y_ahe), ("clahe", Y_clahe1), ("gamma", Y_g05)]
        res["a9_thumbs"] = {}
        for key, img in thumbs:
            fig = plt.figure(figsize=(2.0, 2.0))
            plt.imshow(img, cmap="gray", vmin=0, vmax=255); plt.axis("off")
            pth_i = os.path.join(OUT_DIR, "A9_%s_img.png" % key)
            fig.savefig(pth_i, dpi=90, bbox_inches="tight", pad_inches=0); plt.close(fig)
            fig = plt.figure(figsize=(2.6, 1.9))
            plt.bar(np.arange(256), a_compute_hist(a_quantize8(img)), width=1.0, color="C0")
            plt.xlim(0, 255); plt.yticks([]); plt.xticks([0, 128, 255], fontsize=7)
            pth_h = os.path.join(OUT_DIR, "A9_%s_hist.png" % key)
            fig.savefig(pth_h, dpi=90, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
            res["a9_thumbs"][key] = (pth_i, pth_h)

    return res


def print_part_a(m):
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
    print("Enhancement statistics: method / mean / global std / mean tile std")
    for row in m["enhancement_stats"]:
        print("%-12s %.3f %.3f %.3f" % tuple(row))
    print("=" * 60)


# =====================================================================
# PART B
# =====================================================================


B_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "spatial_frequency_filtering_input.png")
B_SPEC = "magma"   # colormap for spectra


# =====================================================================
# From-scratch helpers
# =====================================================================
def b_conv2d(f, h):
    """Linear convolution, zero-padded, 'same' output (kernel flipped)."""
    f = f.astype(np.float64); h = h.astype(np.float64)
    kh, kw = h.shape
    ph, pw = kh // 2, kw // 2
    hf = h[::-1, ::-1]
    fp = np.pad(f, ((ph, ph), (pw, pw)))
    out = np.zeros_like(f)
    for i in range(kh):
        for j in range(kw):
            out += hf[i, j] * fp[i:i+f.shape[0], j:j+f.shape[1]]
    return out

def b_conv2d_fft(f, h, return_spectra=False):
    """Zero-extended linear convolution. Unshifted kernel, then centered 'same' crop.

    F, H and G use the identical full-convolution grid. A top-left kernel origin
    adds a phase delay; cropping by its radius aligns the odd-kernel output.
    """
    f = np.asarray(f, dtype=np.float64); h = np.asarray(h, dtype=np.float64)
    M, N = f.shape; m, n = h.shape
    if m % 2 == 0 or n % 2 == 0:
        raise ValueError("This centered implementation requires odd kernel dimensions")
    shape = (M+m-1, N+n-1)
    F = np.fft.fft2(f, shape); H = np.fft.fft2(h, shape); G = F * H
    full = np.fft.ifft2(G).real
    si, sj = m//2, n//2
    same = full[si:si+M, sj:sj+N]
    return (same, F, H, G) if return_spectra else same

def b_logmag(X):
    """Centered log-magnitude spectrum: log(1 + |fftshift(X)|)."""
    return np.log1p(np.abs(np.fft.fftshift(X)))

def b_spectrum_of_image(img):
    return b_logmag(np.fft.fft2(img))

def b_spectrum_of_kernel(h, shape):
    """Display-only: pad kernel to image size to show H(u,v)."""
    return b_logmag(np.fft.fft2(h, s=shape))

def b_gaussian_kernel(sigma):
    rad = max(1, int(np.ceil(3 * sigma)))
    ax = np.arange(-rad, rad + 1)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g / g.sum()

# metrics (from scratch)
def b_mse(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def b_psnr(a, b, peak=255.0):
    m = b_mse(a, b)
    return float("inf") if m == 0 else 10.0 * np.log10(peak * peak / m)

def b_gauss_win(size=11, sigma=1.5):
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2)); g /= g.sum()
    return np.outer(g, g)

def b_ssim(a, b, peak=255.0):
    a = a.astype(np.float64); b = b.astype(np.float64)
    w = b_gauss_win(11, 1.5)
    C1 = (0.01 * peak) ** 2; C2 = (0.03 * peak) ** 2
    mu_a = b_conv2d(a, w); mu_b = b_conv2d(b, w)
    va = b_conv2d(a*a, w) - mu_a**2
    vb = b_conv2d(b*b, w) - mu_b**2
    vab = b_conv2d(a*b, w) - mu_a*mu_b
    smap = ((2*mu_a*mu_b + C1) * (2*vab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (va + vb + C2))
    return float(smap.mean())


# =====================================================================
# Display helpers
# =====================================================================
def b_show(ax, img, title="", cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def b_save(fig, tag):
    path = os.path.join(OUT_DIR, tag + ".png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


b_hb = np.ones((3, 3)) / 9.0
b_hs = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)


def b_overview_fig(f, h, hname, tag, clip_spatial):
    g_s = b_conv2d(f, h)
    g_f, F, H, G = b_conv2d_fft(f, h, return_spectra=True)
    diff = np.abs(g_s-g_f)
    sf, sh, sg = b_logmag(F), b_logmag(H), b_logmag(G)
    top = max(sf.max(), sg.max())
    fig, ax = plt.subplots(2,4,figsize=(16,8),layout="constrained")
    b_show(ax[0,0], f, "Input f", vmin=0, vmax=255)
    b_show(ax[0,1], sf, "log(1+|F|), padded grid", cmap=B_SPEC, vmin=0, vmax=top)
    b_show(ax[0,2], h, "Impulse response " + hname, vmin=min(0,h.min()), vmax=h.max())
    for (i,j), v in np.ndenumerate(h):
        ax[0,2].text(j,i,"%.3f" % v,ha="center",va="center",color="red",fontsize=11)
    b_show(ax[0,3], sh, "log(1+|H|), padded grid", cmap=B_SPEC, vmin=0, vmax=sh.max())
    b_show(ax[1,0], np.clip(g_s,0,255), "Spatial result (display clipped)", vmin=0,vmax=255)
    b_show(ax[1,1], np.clip(g_f,0,255), "FFT result (same crop)", vmin=0,vmax=255)
    b_show(ax[1,2], sg, "log(1+|G|), G=H F before crop", cmap=B_SPEC,vmin=0,vmax=top)
    b_show(ax[1,3], diff/1e-12, "Absolute difference / 1e-12",cmap="magma",vmin=0,vmax=1.2)
    fig.colorbar(ax[1,3].images[0],ax=ax[1,3],shrink=.8,label="Error in units of 1e-12")
    fig.suptitle("F and G share a log scale; max absolute error = %.3e" % diff.max())
    return b_save(fig,tag), g_s, g_f


def run_part_b(img_path=B_IMG_PATH):
    os.makedirs(OUT_DIR, exist_ok=True)
    f = np.array(Image.open(img_path).convert("L"), dtype=np.float64)
    res = {"figures": {}, "metrics": {}}
    F = res["figures"]; met = res["metrics"]
    met["shape"] = f.shape

    # ---- B1 Blur ----
    F["B1"], gb_s, gb_f = b_overview_fig(f, b_hb, "h_b", "B1_blur", clip_spatial=False)
    met["blur_mse"] = b_mse(gb_s, gb_f); met["blur_psnr"] = b_psnr(gb_s, gb_f, 255); met["blur_ssim"] = b_ssim(gb_s, gb_f, 255)

    # ---- B2 Sharpen ----
    F["B2"], gs_s, gs_f = b_overview_fig(f, b_hs, "h_s", "B2_sharpen", clip_spatial=True)
    met["sharp_mse"] = b_mse(gs_s, gs_f); met["sharp_psnr"] = b_psnr(gs_s, gs_f, 255); met["sharp_ssim"] = b_ssim(gs_s, gs_f, 255)

    # ---- B3 Frequency analysis (|F|,|H|,|G|) + response cross-section ----
    for name, h, tag in [("Blur h_b", b_hb, "blur"), ("Sharpen h_s", b_hs, "sharpen")]:
        _, Fs, Hs, Gs = b_conv2d_fft(f, h, return_spectra=True)
        spectra = [b_logmag(Fs),b_logmag(Hs),b_logmag(Gs)]
        vmax = max(spectra[0].max(),spectra[2].max())
        fig, ax = plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
        for j,title in enumerate(["log(1+|F|)","log(1+|H|)","log(1+|G|), G=H F"]):
            b_show(ax[j],spectra[j],title,cmap=B_SPEC,vmin=0,
                 vmax=spectra[1].max() if j==1 else vmax)
            fig.colorbar(ax[j].images[0],ax=ax[j],shrink=.8)
        fig.suptitle(name + " - same padded grid; F and G share the scale")
        F["B3_"+tag] = b_save(fig,"B3_spectra_"+tag)

    # ---- B4 Impulse-response verification ----
    sz = 31
    delta = np.zeros((sz, sz)); delta[sz//2, sz//2] = 1.0
    c = sz // 2; r = 4  # crop half-size for visibility
    for name, h, tag in [("Blur", b_hb, "blur"), ("Sharpen", b_hs, "sharpen")]:
        out = b_conv2d(delta, h)
        out_fft = b_conv2d_fft(delta, h)
        met["impulse_fft_err_"+tag] = float(np.max(np.abs(out-out_fft)))
        kh, kw = h.shape
        patch = out[c-kh//2:c-kh//2+kh, c-kw//2:c-kw//2+kw]
        err = float(np.max(np.abs(patch - h)))
        fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
        b_show(ax[0], delta, "delta(x,y)")
        # |FFT(delta)| = 1 at every frequency -> log(1+1)=0.693 constant.
        # Fix vmin/vmax so the panel reads as genuinely FLAT (otherwise imshow
        # auto-scales ~1e-16 float noise to full contrast and looks textured).
        b_show(ax[1], b_logmag(np.fft.fft2(delta)), "log|FFT(delta)|  (FLAT = all freqs equal)",
             cmap=B_SPEC, vmin=0.0, vmax=1.0)
        b_show(ax[2], out[c-r:c+r+1, c-r:c+r+1], "output h*delta  (center 9x9)")
        b_show(ax[3], h, "impulse response h", vmin=min(0,h.min()), vmax=h.max())
        for (hi,hj), value in np.ndenumerate(h):
            ax[3].text(hj,hi,"%.3f" % value,ha="center",va="center",color="red",fontsize=10)
        b_show(ax[4], b_spectrum_of_kernel(h, (sz, sz)), "log|H(u,v)|", cmap=B_SPEC)
        fig.suptitle("%s :  max|output_center - h| = %.2e" % (name, err), y=1.05)
        F["B4_" + tag] = b_save(fig, "B4_impulse_" + tag)
        met["impulse_err_" + tag] = err

    # ---- B5 Gaussian-based unsharp masking ----
    COLS = ["original f", "Gaussian kernel", "blurred f_L", "high-freq f_H", "sharpened g"]

    def components(sigma):
        gk = b_gaussian_kernel(sigma); fL = b_conv2d(f, gk); fH = f - fL
        return gk, fL, fH

    def b5_panel(variants, title, tag):
        # variants: list of (label, gk, fL, fH, g). 2 rows per variant: image row + |F| spectrum row.
        nrows = 2 * len(variants)
        fig, ax = plt.subplots(nrows, 5, figsize=(16, 3.05 * nrows))
        cmaps = ["gray", "viridis", "gray", "gray", "gray"]
        for vi, (lbl, gk, fL, fH, g) in enumerate(variants):
            ri = 2 * vi
            imgs = [f, gk, fL, fH, np.clip(g, 0, 255)]
            for c in range(5):
                b_show(ax[ri, c], imgs[c], "%s  [%s]" % (COLS[c], lbl), cmap=cmaps[c])
            specs = [b_spectrum_of_image(f), b_spectrum_of_kernel(gk, f.shape),
                     b_spectrum_of_image(fL), b_spectrum_of_image(fH), b_spectrum_of_image(g)]
            for c in range(5):
                b_show(ax[ri + 1, c], specs[c], "|F| of " + COLS[c], cmap=B_SPEC)
        fig.suptitle(title, y=1.0)
        plt.tight_layout()
        return b_save(fig, tag)

    # (A) vary sigma (k fixed) : rows = [sigma=1 image, sigma=1 |F|, sigma=3 image, sigma=3 |F|]
    k_fix = 2.0
    var_sigma = []
    for s in [1.0, 3.0]:
        gk, fL, fH = components(s)
        var_sigma.append(("sigma=%.0f, k=%.0f" % (s, k_fix), gk, fL, fH, (1 + k_fix) * f - k_fix * fL))
    F["B5_sigma"] = b5_panel(var_sigma, "B5 - vary sigma (k=%.0f fixed);  each variant: image row + |F| spectrum row" % k_fix, "B5_1_vary_sigma")

    # (B) vary k (sigma fixed = 3)
    s_fix = 3.0
    gk3, fL3, fH3 = components(s_fix)
    var_k = [("sigma=%.0f, k=%.0f" % (s_fix, k), gk3, fL3, fH3, (1 + k) * f - k * fL3) for k in [1.0, 2.0]]
    F["B5_k"] = b5_panel(var_k, "B5 - vary k (sigma=%.0f fixed);  each variant: image row + |F| spectrum row" % s_fix, "B5_2_vary_k")

    # (C) compare with the fixed 3x3 sharpening kernel h_s (use sigma=3 for coarse-structure sharpening)
    g_unsharp = (1 + 1.0) * f - 1.0 * fL3        # sigma=3, k=1
    g_fixed = b_conv2d(f, b_hs)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    b_show(ax[0], f, "original f")
    b_show(ax[1], np.clip(g_fixed, 0, 255), "fixed 3x3 kernel h_s")
    b_show(ax[2], np.clip(g_unsharp, 0, 255), "unsharp sigma=3, k=1")
    fig.suptitle("B5 comparison: fixed frequency gain vs adjustable Gaussian scale and strength", y=1.02)
    F["B5_vs_fixed"] = b_save(fig, "B5_3_vs_fixed")

    # B5 verification in both domains, for every sigma/k combination.
    met["unsharp_comparison"] = []
    fig, ax = plt.subplots(4,3,figsize=(12,14),layout="constrained")
    for row,(sigma,k) in enumerate([(1.,1.),(1.,2.),(3.,1.),(3.,2.)]):
        gk = b_gaussian_kernel(sigma)
        fLs = b_conv2d(f,gk); fLf = b_conv2d_fft(f,gk)
        us = (1+k)*f-k*fLs; uf = (1+k)*f-k*fLf
        met["unsharp_comparison"].append([sigma,k,gk.shape[0],b_mse(us,uf),
                                           b_psnr(us,uf,255),b_ssim(us,uf,255),
                                           float(np.max(np.abs(us-uf))),
                                           float(100*np.mean((us<0)|(us>255)))])
        label = "sigma=%.0f, k=%.0f" % (sigma,k)
        b_show(ax[row,0],np.clip(us,0,255),"Spatial: "+label,vmin=0,vmax=255)
        b_show(ax[row,1],np.clip(uf,0,255),"FFT: "+label,vmin=0,vmax=255)
        b_show(ax[row,2],np.abs(us-uf)/1e-12,"Difference / 1e-12",cmap=B_SPEC,vmin=0,vmax=1.2)
        fig.colorbar(ax[row,2].images[0],ax=ax[row,2],shrink=.8)
    F["B5_domains"] = b_save(fig,"B5_4_domains")
    met["mean_input"] = float(f.mean())
    met["mean_sharpen"] = float(gs_s.mean())
    return res


def print_part_b(m):
    print("=" * 60); print("Part B  Quantitative results"); print("=" * 60)
    print("input size            : %dx%d" % m["shape"])
    print("[Blur]    spatial vs freq:  MSE=%.3e  PSNR=%.1f dB  SSIM=%.6f" % (m["blur_mse"], m["blur_psnr"], m["blur_ssim"]))
    print("[Sharpen] spatial vs freq:  MSE=%.3e  PSNR=%.1f dB  SSIM=%.6f" % (m["sharp_mse"], m["sharp_psnr"], m["sharp_ssim"]))
    print("impulse h*delta=h error:  blur=%.1e  sharpen=%.1e" % (m["impulse_err_blur"], m["impulse_err_sharpen"]))
    print("B5: sigma k size MSE PSNR SSIM max_error out_of_range_percent")
    for row in m["unsharp_comparison"]:
        print("%.0f %.0f %d %.3e %.2f %.8f %.3e %.3f" % tuple(row))
    print("=" * 60)


# =====================================================================
# PART C
# =====================================================================


C_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "wiener_filter_input.png")   # alt: wiener_filter_input_2 / _rocket
C_RMS_TARGET = 0.03
C_KS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]


# ---------- from-scratch helpers ----------
def c_conv2d(a, h):
    a = a.astype(np.float64); h = h.astype(np.float64)
    kh, kw = h.shape; ph, pw = kh // 2, kw // 2
    hf = h[::-1, ::-1]; fp = np.pad(a, ((ph, ph), (pw, pw)))
    out = np.zeros_like(a)
    for i in range(kh):
        for j in range(kw):
            out += hf[i, j] * fp[i:i+a.shape[0], j:j+a.shape[1]]
    return out

def c_mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

def c_psnr(a, b, peak=1.0):
    m = c_mse(a, b)
    return float("inf") if m == 0 else 10.0 * np.log10(peak * peak / m)

def c_gw(size=11, sigma=1.5):
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2)); g /= g.sum()
    return np.outer(g, g)

def c_ssim(a, b, peak=1.0):
    a = a.astype(np.float64); b = b.astype(np.float64); w = c_gw(11, 1.5)
    C1 = (0.01 * peak) ** 2; C2 = (0.03 * peak) ** 2
    mu_a = c_conv2d(a, w); mu_b = c_conv2d(b, w)
    va = c_conv2d(a*a, w) - mu_a**2; vb = c_conv2d(b*b, w) - mu_b**2; vab = c_conv2d(a*b, w) - mu_a*mu_b
    smap = ((2*mu_a*mu_b + C1) * (2*vab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (va + vb + C2))
    return float(smap.mean())

def c_gaussian_psf(size=15, sigma=2.5):
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return h / h.sum()

def c_psf2otf(psf, shape):
    """Pad PSF to image size, shift its center to the origin (zero-phase) -> OTF H(u,v)."""
    p = np.zeros(shape); kh, kw = psf.shape
    p[:kh, :kw] = psf
    p = np.roll(p, -(kh // 2), axis=0)
    p = np.roll(p, -(kw // 2), axis=1)
    return np.fft.fft2(p)

def c_wiener_restore(g, H, K):
    G = np.fft.fft2(g)
    Wf = np.conj(H) / (np.abs(H)**2 + K)
    return np.real(np.fft.ifft2(Wf * G))

def c_show(ax, img, title="", cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.axis("off")

def c_save(fig, tag):
    path = os.path.join(OUT_DIR, tag + ".png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


# input images to test (first is the primary, shown in full)
C_INPUTS = [
    ("primary", "wiener_filter_input.png",   "cameraman"),
    ("img2",    "wiener_filter_input_2.png", "input 2"),
    ("rocket",  "wiener_filter_input_rocket.png", "rocket"),
]


def c_run_one(path, key, label, psf, rng, full=False):
    """Run the full Wiener experiment on one image. Returns a dict with figures + table."""
    f1 = np.array(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    H = c_psf2otf(psf, f1.shape)
    gb = np.real(np.fft.ifft2(H * np.fft.fft2(f1)))           # C1 blur
    n = rng.normal(0.0, 1.0, f1.shape); n = n - n.mean()
    n = n * (C_RMS_TARGET / np.sqrt(np.mean(n**2)))             # C2 noise (exact RMS)
    g = gb + n
    out = {"key": key, "label": label, "figs": {}, "noise_rms": float(np.sqrt(np.mean(n**2)))}
    figs = out["figs"]

    if full:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        c_show(ax[0], f1, "Original f(x,y)", vmin=0, vmax=1)
        c_show(ax[1], psf, "PSF h(x,y) 15x15 sigma=2.5", cmap="viridis")
        c_show(ax[2], gb, "Blurred g_b = h*f", vmin=0, vmax=1)
        figs["blur"] = c_save(fig, "C_%s_blur" % key)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        c_show(ax[0], gb, "Blurred g_b", vmin=0, vmax=1)
        c_show(ax[1], g, "Blurred + noisy g", vmin=0, vmax=1)
        figs["noise"] = c_save(fig, "C_%s_noise" % key)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        c_show(ax[0], f1, "Original", vmin=0, vmax=1)
        c_show(ax[1], g, "Degraded g", vmin=0, vmax=1)
        c_show(ax[2], np.clip(c_wiener_restore(g, H, 1e-2), 0, 1), "Wiener restored (K=1e-2)", vmin=0, vmax=1)
        figs["restore"] = c_save(fig, "C_%s_restore" % key)
    else:
        # compact degradation panel for the extra inputs
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        c_show(ax[0], f1, "Original (%s)" % label, vmin=0, vmax=1)
        c_show(ax[1], gb, "Blurred", vmin=0, vmax=1)
        c_show(ax[2], g, "Blurred + noisy", vmin=0, vmax=1)
        figs["deg"] = c_save(fig, "C_%s_degraded" % key)

    # K sweep + table
    rows = []
    raw_rows = []
    fig, ax = plt.subplots(1, len(C_KS), figsize=(18, 4))
    for i, K in enumerate(C_KS):
        restored = c_wiener_restore(g, H, K)
        rc = np.clip(restored, 0, 1)
        raw_rows.append((K,c_mse(f1,restored),c_psnr(f1,restored,1.0),c_ssim(f1,restored,1.0),float(100*np.mean((restored<0)|(restored>1)))))
        mm = c_mse(f1, rc); ps = c_psnr(f1, rc, 1.0); ss = c_ssim(f1, rc, 1.0)
        rows.append((K, mm, ps, ss))
        c_show(ax[i], rc, "K=%.0e\nPSNR=%.2f" % (K, ps), vmin=0, vmax=1)
    fig.suptitle("K sweep - %s" % label, y=1.02)
    figs["sweep"] = c_save(fig, "C_%s_Ksweep" % key)
    out["table"] = rows
    out["raw_table"] = raw_rows
    out["noise_mean"] = float(n.mean())
    out["degraded_metrics"] = [c_mse(f1,g),c_psnr(f1,g,1.0),c_ssim(f1,g,1.0)]
    out["best_psnr_K"] = max(rows, key=lambda t: t[2])[0]
    out["best_ssim_K"] = max(rows, key=lambda t: t[3])[0]
    return out


def run_part_c():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    psf = c_gaussian_psf(15, 2.5)
    results = []
    for i, (key, name, label) in enumerate(C_INPUTS):
        path = os.path.join(BASE_DIR, "images", name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        results.append(c_run_one(path, key, label, psf, rng, full=(i == 0)))
    return {"psf_sum": float(psf.sum()), "inputs": results}


def print_part_c(res):
    print("=" * 60); print("Part C  Quantitative results"); print("=" * 60)
    print("PSF sum = %.6f ; noise RMS target = %.3f ; inputs = %d"
          % (res["psf_sum"], C_RMS_TARGET, len(res["inputs"])))
    for r in res["inputs"]:
        print("\n[%s]  noise RMS=%.5f" % (r["label"], r["noise_rms"]))
        print("  Metrics after clipping restored values to [0, 1]")
        print("  %-10s  %-9s  %-9s  %-7s" % ("K", "MSE", "PSNR(dB)", "SSIM"))
        for K, mm, ps, ss in r["table"]:
            print("  %-10.0e  %-9.5f  %-9.2f  %-7.4f" % (K, mm, ps, ss))
        print("  best PSNR at K=%.0e , best SSIM at K=%.0e" % (r["best_psnr_K"], r["best_ssim_K"]))
    print("=" * 60)


def portable_results(value):
    """Save relative figure paths so the complete folder can be moved."""
    if isinstance(value, dict):
        return {k: portable_results(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_results(v) for v in value]
    if isinstance(value, str) and value.endswith(".png"):
        return "output/" + Path(value).name
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", nargs="+", choices=["A", "B", "C"],
                        default=["A", "B", "C"], help="Parts to execute (default: A B C)")
    args = parser.parse_args(argv)
    runners = {"A": run_part_a, "B": run_part_b, "C": run_part_c}
    printers = {"A": print_part_a, "B": print_part_b, "C": print_part_c}
    for part in dict.fromkeys(args.part):
        result = runners[part]()
        with (BASE_DIR / f"results_{part}.json").open("w", encoding="utf-8") as stream:
            json.dump(portable_results(result), stream, indent=2, default=float)
        printers[part](result if part == "C" else result["metrics"])
    print(f"Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
