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
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "wiener_filter_input.png")   # alt: wiener_filter_input_2 / _rocket
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = str(BASE_DIR / "output")
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


# input images to test (first is the primary, shown in full)
INPUTS = [
    ("primary", "wiener_filter_input.png",   "cameraman"),
    ("img2",    "wiener_filter_input_2.png", "input 2"),
    ("rocket",  "wiener_filter_input_rocket.png", "rocket"),
]


def run_one(path, key, label, psf, rng, full=False):
    """Run the full Wiener experiment on one image. Returns a dict with figures + table."""
    f1 = np.array(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    H = psf2otf(psf, f1.shape)
    gb = np.real(np.fft.ifft2(H * np.fft.fft2(f1)))           # C1 blur
    n = rng.normal(0.0, 1.0, f1.shape); n = n - n.mean()
    n = n * (RMS_TARGET / np.sqrt(np.mean(n**2)))             # C2 noise (exact RMS)
    g = gb + n
    out = {"key": key, "label": label, "figs": {}, "noise_rms": float(np.sqrt(np.mean(n**2)))}
    figs = out["figs"]

    if full:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        show(ax[0], f1, "Original f(x,y)", vmin=0, vmax=1)
        show(ax[1], psf, "PSF h(x,y) 15x15 sigma=2.5", cmap="viridis")
        show(ax[2], gb, "Blurred g_b = h*f", vmin=0, vmax=1)
        figs["blur"] = save(fig, "C_%s_blur" % key)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        show(ax[0], gb, "Blurred g_b", vmin=0, vmax=1)
        show(ax[1], g, "Blurred + noisy g", vmin=0, vmax=1)
        figs["noise"] = save(fig, "C_%s_noise" % key)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        show(ax[0], f1, "Original", vmin=0, vmax=1)
        show(ax[1], g, "Degraded g", vmin=0, vmax=1)
        show(ax[2], np.clip(wiener_restore(g, H, 1e-2), 0, 1), "Wiener restored (K=1e-2)", vmin=0, vmax=1)
        figs["restore"] = save(fig, "C_%s_restore" % key)
    else:
        # compact degradation panel for the extra inputs
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        show(ax[0], f1, "Original (%s)" % label, vmin=0, vmax=1)
        show(ax[1], gb, "Blurred", vmin=0, vmax=1)
        show(ax[2], g, "Blurred + noisy", vmin=0, vmax=1)
        figs["deg"] = save(fig, "C_%s_degraded" % key)

    # K sweep + table
    rows = []
    raw_rows = []
    fig, ax = plt.subplots(1, len(KS), figsize=(18, 4))
    for i, K in enumerate(KS):
        restored = wiener_restore(g, H, K)
        rc = np.clip(restored, 0, 1)
        raw_rows.append((K,mse(f1,restored),psnr(f1,restored,1.0),ssim(f1,restored,1.0),float(100*np.mean((restored<0)|(restored>1)))))
        mm = mse(f1, rc); ps = psnr(f1, rc, 1.0); ss = ssim(f1, rc, 1.0)
        rows.append((K, mm, ps, ss))
        show(ax[i], rc, "K=%.0e\nPSNR=%.2f" % (K, ps), vmin=0, vmax=1)
    fig.suptitle("K sweep - %s" % label, y=1.02)
    figs["sweep"] = save(fig, "C_%s_Ksweep" % key)
    out["table"] = rows
    out["raw_table"] = raw_rows
    out["noise_mean"] = float(n.mean())
    out["degraded_metrics"] = [mse(f1,g),psnr(f1,g,1.0),ssim(f1,g,1.0)]
    out["best_psnr_K"] = max(rows, key=lambda t: t[2])[0]
    out["best_ssim_K"] = max(rows, key=lambda t: t[3])[0]
    return out


def run_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    psf = gaussian_psf(15, 2.5)
    results = []
    for i, (key, name, label) in enumerate(INPUTS):
        path = os.path.join(BASE_DIR, "images", name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        results.append(run_one(path, key, label, psf, rng, full=(i == 0)))
    return {"psf_sum": float(psf.sum()), "inputs": results}


def _print(res):
    print("=" * 60); print("Part C  Quantitative results"); print("=" * 60)
    print("PSF sum = %.6f ; noise RMS target = %.3f ; inputs = %d"
          % (res["psf_sum"], RMS_TARGET, len(res["inputs"])))
    for r in res["inputs"]:
        print("\n[%s]  noise RMS=%.5f" % (r["label"], r["noise_rms"]))
        print("  %-10s  %-9s  %-9s  %-7s" % ("K", "MSE", "PSNR(dB)", "SSIM"))
        for K, mm, ps, ss in r["table"]:
            print("  %-10.0e  %-9.5f  %-9.2f  %-7.4f" % (K, mm, ps, ss))
        print("  Raw restoration: K MSE PSNR SSIM clipped_percent")
        for row in r["raw_table"]:
            print("  %.0e %.6f %.2f %.4f %.3f" % tuple(row))
        print("  best PSNR at K=%.0e , best SSIM at K=%.0e" % (r["best_psnr_K"], r["best_ssim_K"]))
    print("=" * 60)


def portable_results(value):
    """Keep cached experiment paths valid when the project folder is moved."""
    if isinstance(value, dict):
        return {k: portable_results(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_results(v) for v in value]
    if isinstance(value, str) and value.endswith(".png"):
        return "output/" + Path(value).name
    return value


if __name__ == "__main__":
    r = run_all()
    with open(BASE_DIR / "results_C.json", "w", encoding="utf-8") as stream:
        json.dump(portable_results(r), stream, indent=2, default=float)
    _print(r)
