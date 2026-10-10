# -*- coding: utf-8 -*-
"""
Homework 1 (Computer Vision) - Part C
Wiener Filter (Image Restoration)  (from scratch)

Degradation model:  g = h * f + n
  - PSF h: 15x15 Gaussian, sigma = 2.5, normalized so sum(h) = 1
  - noise n: zero-mean Gaussian, RMS = 0.03 (rescaled to the exact RMS)
Wiener restoration:  F_hat = conj(H) / (|H|^2 + K) * G,  restored = IFFT(F_hat).
K is swept over {1e-6, 1e-4, 1e-3, 1e-2, 1e-1} and compared by MSE / PSNR / SSIM.

Allowed built-ins: numpy (+ numpy.fft), PIL (loading), matplotlib (display).
Run:  python partC_wiener.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMG_PATH = os.path.join("images", "wiener_filter_input.png")   # alt: wiener_filter_input_2 / _rocket
OUT_DIR = "output"
RMS_TARGET = 0.03
KS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]


# ---------- from-scratch helpers ----------
def conv2d(a, h):
    a = a.astype(np.float64); h = h.astype(np.float64)
    kh, kw = h.shape; ph, pw = kh // 2, kw // 2
    hf = h[::-1, ::-1]; fp = np.pad(a, ((ph, ph), (pw, pw)))
    out = np.zeros_like(a)
    for i in range(kh):
        for j in range(kw):
            out += hf[i, j] * fp[i:i+a.shape[0], j:j+a.shape[1]]
    return out

def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

def psnr(a, b, peak=1.0):
    m = mse(a, b)
    return float("inf") if m == 0 else 10.0 * np.log10(peak * peak / m)

def _gw(size=11, sigma=1.5):
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2)); g /= g.sum()
    return np.outer(g, g)

def ssim(a, b, peak=1.0):
    a = a.astype(np.float64); b = b.astype(np.float64); w = _gw(11, 1.5)
    C1 = (0.01 * peak) ** 2; C2 = (0.03 * peak) ** 2
    mu_a = conv2d(a, w); mu_b = conv2d(b, w)
    va = conv2d(a*a, w) - mu_a**2; vb = conv2d(b*b, w) - mu_b**2; vab = conv2d(a*b, w) - mu_a*mu_b
    smap = ((2*mu_a*mu_b + C1) * (2*vab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (va + vb + C2))
    return float(smap.mean())

def gaussian_psf(size=15, sigma=2.5):
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return h / h.sum()

def psf2otf(psf, shape):
    """Pad PSF to image size, shift its center to the origin (zero-phase) -> OTF H(u,v)."""
    p = np.zeros(shape); kh, kw = psf.shape
    p[:kh, :kw] = psf
    p = np.roll(p, -(kh // 2), axis=0)
    p = np.roll(p, -(kw // 2), axis=1)
    return np.fft.fft2(p)

def wiener_restore(g, H, K):
    G = np.fft.fft2(g)
    Wf = np.conj(H) / (np.abs(H)**2 + K)
    return np.real(np.fft.ifft2(Wf * G))

def show(ax, img, title="", cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.axis("off")

def save(fig, tag):
    path = os.path.join(OUT_DIR, tag + ".png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def run_all(img_path=IMG_PATH):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)           # fixed seed -> reproducible noise
    f1 = np.array(Image.open(img_path).convert("L"), dtype=np.float64) / 255.0
    res = {"figures": {}, "metrics": {}}
    F = res["figures"]; met = res["metrics"]
    met["shape"] = f1.shape

    psf = gaussian_psf(15, 2.5)
    H = psf2otf(psf, f1.shape)
    met["psf_sum"] = float(psf.sum())

    # ---- C1: blur ----
    gb = np.real(np.fft.ifft2(H * np.fft.fft2(f1)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    show(ax[0], f1, "Original f(x,y)", vmin=0, vmax=1)
    show(ax[1], psf, "PSF h(x,y) 15x15 sigma=2.5", cmap="viridis")
    show(ax[2], gb, "Blurred g_b = h*f", vmin=0, vmax=1)
    F["C1"] = save(fig, "C1_blur")

    # ---- C2: add noise RMS=0.03 ----
    n = rng.normal(0.0, 1.0, f1.shape)
    n = n - n.mean()
    n = n * (RMS_TARGET / np.sqrt(np.mean(n**2)))
    met["noise_rms"] = float(np.sqrt(np.mean(n**2)))
    g = gb + n
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    show(ax[0], gb, "Blurred g_b", vmin=0, vmax=1)
    show(ax[1], g, "Blurred + noisy g", vmin=0, vmax=1)
    F["C2"] = save(fig, "C2_noise")

    # ---- C3: Wiener restore (representative K) ----
    restored_demo = wiener_restore(g, H, 1e-2)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    show(ax[0], f1, "Original", vmin=0, vmax=1)
    show(ax[1], g, "Degraded g", vmin=0, vmax=1)
    show(ax[2], np.clip(restored_demo, 0, 1), "Wiener restored (K=1e-2)", vmin=0, vmax=1)
    F["C3"] = save(fig, "C3_wiener_demo")

    # ---- C4: K sweep + metrics table ----
    rows = []
    fig, ax = plt.subplots(1, len(KS), figsize=(18, 4))
    for i, K in enumerate(KS):
        rc = np.clip(wiener_restore(g, H, K), 0, 1)
        m = mse(f1, rc); ps = psnr(f1, rc, 1.0); ss = ssim(f1, rc, 1.0)
        rows.append((K, m, ps, ss))
        show(ax[i], rc, "K=%.0e\nPSNR=%.2f" % (K, ps), vmin=0, vmax=1)
    F["C4"] = save(fig, "C4_K_sweep")
    met["table"] = rows
    met["best_psnr_K"] = max(rows, key=lambda t: t[2])[0]
    met["best_ssim_K"] = max(rows, key=lambda t: t[3])[0]
    return res


def _print(m):
    print("=" * 56); print("Part C  Quantitative results"); print("=" * 56)
    print("input %dx%d | PSF sum=%.6f | noise RMS=%.5f (target %.3f)"
          % (m["shape"][0], m["shape"][1], m["psf_sum"], m["noise_rms"], RMS_TARGET))
    print("%-10s | %-9s | %-9s | %-7s" % ("K", "MSE", "PSNR(dB)", "SSIM"))
    print("-" * 46)
    for K, mm, ps, ss in m["table"]:
        print("%-10.0e | %-9.5f | %-9.2f | %-7.4f" % (K, mm, ps, ss))
    print("best PSNR at K=%.0e ; best SSIM at K=%.0e" % (m["best_psnr_K"], m["best_ssim_K"]))
    print("=" * 56)


if __name__ == "__main__":
    r = run_all()
    _print(r["metrics"])
    print("figures saved to '%s/':" % OUT_DIR)
    for k, v in r["figures"].items():
        print("  [%s] %s" % (k, v))
