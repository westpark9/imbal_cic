# -*- coding: utf-8 -*-
"""
Homework 1 (Computer Vision) - Part B
Spatial- and Frequency-Domain Filtering  (from scratch)

Each filter is applied in both the spatial domain (direct convolution) and the
frequency domain (FFT multiply with zero-padding for LINEAR convolution), and the
two are compared via MSE / PSNR / SSIM. Also: frequency-response analysis,
impulse-response verification, and Gaussian-based unsharp masking.

Allowed built-ins: numpy (+ numpy.fft), PIL (loading), matplotlib (display).

Run:  python partB_filtering.py
-> figures saved to output/ as B1..B6, quantitative results printed.

Parameters:
  - blur kernel h_b = (1/9) * ones(3,3)
  - sharpen kernel h_s = [[0,-1,0],[-1,5,-1],[0,-1,0]]
  - Fourier display: centered log-magnitude  log(1 + |fftshift(F)|)
  - frequency filtering: zero-pad to (M+m-1, N+n-1) => linear convolution
  - unsharp masking: sigma = 1.0, 3.0 ; k = 1.0, 2.0 ; Gaussian radius = ceil(3*sigma)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMG_PATH = os.path.join("images", "spatial_frequency_filtering_input.png")
OUT_DIR = "output"
SPEC = "magma"   # colormap for spectra


# =====================================================================
# From-scratch helpers
# =====================================================================
def conv2d(f, h):
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

def conv2d_fft(f, h):
    """Frequency-domain LINEAR convolution: zero-pad to (M+m-1,N+n-1), multiply, crop 'same'."""
    f = f.astype(np.float64); h = h.astype(np.float64)
    M, N = f.shape; m, n = h.shape
    P, Q = M + m - 1, N + n - 1            # <-- zero-padding that makes it LINEAR (not circular)
    F = np.fft.fft2(f, (P, Q)); H = np.fft.fft2(h, (P, Q))
    g = np.real(np.fft.ifft2(F * H))
    si, sj = (m - 1) // 2, (n - 1) // 2
    return g[si:si+M, sj:sj+N]

def logmag(X):
    """Centered log-magnitude spectrum: log(1 + |fftshift(X)|)."""
    return np.log1p(np.abs(np.fft.fftshift(X)))

def spectrum_of_image(img):
    return logmag(np.fft.fft2(img))

def spectrum_of_kernel(h, shape):
    """Display-only: pad kernel to image size to show H(u,v)."""
    return logmag(np.fft.fft2(h, s=shape))

def gaussian_kernel(sigma):
    rad = max(1, int(np.ceil(3 * sigma)))
    ax = np.arange(-rad, rad + 1)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g / g.sum()

# metrics (from scratch)
def mse(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def psnr(a, b, peak=255.0):
    m = mse(a, b)
    return float("inf") if m == 0 else 10.0 * np.log10(peak * peak / m)

def _gauss_win(size=11, sigma=1.5):
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2)); g /= g.sum()
    return np.outer(g, g)

def ssim(a, b, peak=255.0):
    a = a.astype(np.float64); b = b.astype(np.float64)
    w = _gauss_win(11, 1.5)
    C1 = (0.01 * peak) ** 2; C2 = (0.03 * peak) ** 2
    mu_a = conv2d(a, w); mu_b = conv2d(b, w)
    va = conv2d(a*a, w) - mu_a**2
    vb = conv2d(b*b, w) - mu_b**2
    vab = conv2d(a*b, w) - mu_a*mu_b
    smap = ((2*mu_a*mu_b + C1) * (2*vab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (va + vb + C2))
    return float(smap.mean())


# =====================================================================
# Display helpers
# =====================================================================
def show(ax, img, title="", cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def save(fig, tag):
    path = os.path.join(OUT_DIR, tag + ".png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


hb = np.ones((3, 3)) / 9.0
hs = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)


def _overview_fig(f, h, hname, tag, clip_spatial):
    """B1/B2 shared layout: row1 inputs/responses, row2 results + difference."""
    g_s = conv2d(f, h)          # spatial
    g_f = conv2d_fft(f, h)      # frequency (linear conv)
    diff = np.abs(g_s - g_f)
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    show(ax[0, 0], f, "1) input f(x,y)")
    show(ax[0, 1], spectrum_of_image(f), "2) log|F(u,v)|", cmap=SPEC)
    show(ax[0, 2], h, "3) impulse response %s" % hname)
    show(ax[0, 3], spectrum_of_kernel(h, f.shape), "4) log|H(u,v)|", cmap=SPEC)
    disp_s = np.clip(g_s, 0, 255) if clip_spatial else g_s
    disp_f = np.clip(g_f, 0, 255) if clip_spatial else g_f
    show(ax[1, 0], disp_s, "spatial  g_s = h * f")
    show(ax[1, 1], disp_f, "frequency  g_f = F^-1{H F}")
    show(ax[1, 2], spectrum_of_image(g_f), "log(1+|G_f|) of output", cmap=SPEC)
    show(ax[1, 3], diff, "|g_s - g_f|  (max=%.1e)" % diff.max())
    path = save(fig, tag)
    return path, g_s, g_f


def run_all(img_path=IMG_PATH):
    os.makedirs(OUT_DIR, exist_ok=True)
    f = np.array(Image.open(img_path).convert("L"), dtype=np.float64)
    res = {"figures": {}, "metrics": {}}
    F = res["figures"]; met = res["metrics"]
    met["shape"] = f.shape

    # ---- B1 Blur ----
    F["B1"], gb_s, gb_f = _overview_fig(f, hb, "h_b", "B1_blur", clip_spatial=False)
    met["blur_mse"] = mse(gb_s, gb_f); met["blur_psnr"] = psnr(gb_s, gb_f, 255); met["blur_ssim"] = ssim(gb_s, gb_f, 255)

    # ---- B2 Sharpen ----
    F["B2"], gs_s, gs_f = _overview_fig(f, hs, "h_s", "B2_sharpen", clip_spatial=True)
    met["sharp_mse"] = mse(gs_s, gs_f); met["sharp_psnr"] = psnr(gs_s, gs_f, 255); met["sharp_ssim"] = ssim(gs_s, gs_f, 255)

    # ---- B3 Frequency analysis (|F|,|H|,|G|) + response cross-section ----
    F_sp = spectrum_of_image(f)
    for name, h, tag in [("Blur h_b", hb, "blur"), ("Sharpen h_s", hs, "sharpen")]:
        H = np.fft.fft2(h, s=f.shape)
        G = H * np.fft.fft2(f)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        show(ax[0], F_sp, "|F(u,v)|  input", cmap=SPEC)
        show(ax[1], logmag(H), "|H(u,v)|  (%s)" % name, cmap=SPEC)
        show(ax[2], logmag(G), "|G|=|H F|  output", cmap=SPEC)
        fig.suptitle(name, y=1.02)
        F["B3_" + tag] = save(fig, "B3_spectra_" + tag)

    # central horizontal cross-section of |H| (linear), proves low-pass vs high-pass
    Hb = np.abs(np.fft.fftshift(np.fft.fft2(hb, s=f.shape)))
    Hs = np.abs(np.fft.fftshift(np.fft.fft2(hs, s=f.shape)))
    cx = f.shape[0] // 2
    u = np.arange(f.shape[1]) - f.shape[1] // 2
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(u, Hb[cx, :]); ax[0].axhline(1, color="k", ls="--", lw=0.7)
    ax[0].set_title("|H_b| central row (blur): peak at center, decays -> LOW-PASS")
    ax[0].set_xlabel("frequency (u, 0=DC)"); ax[0].set_ylabel("|H_b|"); ax[0].grid(alpha=0.3)
    ax[1].plot(u, Hs[cx, :], color="C3"); ax[1].axhline(1, color="k", ls="--", lw=0.7)
    ax[1].set_title("|H_s| central row (sharpen): grows to the edges -> HIGH-PASS")
    ax[1].set_xlabel("frequency (u, 0=DC)"); ax[1].set_ylabel("|H_s|"); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    F["B3_profile"] = save(fig, "B3_response_profile")
    met["Hb_dc"] = float(Hb[cx, cx]); met["Hb_edge"] = float(Hb[cx, 0])
    met["Hs_dc"] = float(Hs[cx, cx]); met["Hs_edge"] = float(Hs[cx, 0])

    # ---- B4 Impulse-response verification ----
    sz = 31
    delta = np.zeros((sz, sz)); delta[sz//2, sz//2] = 1.0
    c = sz // 2; r = 4  # crop half-size for visibility
    for name, h, tag in [("Blur", hb, "blur"), ("Sharpen", hs, "sharpen")]:
        out = conv2d(delta, h)
        kh, kw = h.shape
        patch = out[c-kh//2:c-kh//2+kh, c-kw//2:c-kw//2+kw]
        err = float(np.max(np.abs(patch - h)))
        fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
        show(ax[0], delta, "delta(x,y)")
        # |FFT(delta)| = 1 at every frequency -> log(1+1)=0.693 constant.
        # Fix vmin/vmax so the panel reads as genuinely FLAT (otherwise imshow
        # auto-scales ~1e-16 float noise to full contrast and looks textured).
        show(ax[1], logmag(np.fft.fft2(delta)), "log|FFT(delta)|  (FLAT = all freqs equal)",
             cmap=SPEC, vmin=0.0, vmax=1.0)
        show(ax[2], out[c-r:c+r+1, c-r:c+r+1], "output h*delta  (center 9x9)")
        show(ax[3], h, "impulse response h")
        show(ax[4], spectrum_of_kernel(h, (sz, sz)), "log|H(u,v)|", cmap=SPEC)
        fig.suptitle("%s :  max|output_center - h| = %.2e" % (name, err), y=1.05)
        F["B4_" + tag] = save(fig, "B4_impulse_" + tag)
        met["impulse_err_" + tag] = err

    # ---- B5 Gaussian-based unsharp masking ----
    sigmas = [1.0, 3.0]; ks = [1.0, 2.0]
    fLs = {s: conv2d(f, gaussian_kernel(s)) for s in sigmas}

    # (1) components per sigma (depend on sigma only; k not involved)
    fig, ax = plt.subplots(len(sigmas), 3, figsize=(12, 7.5))
    for i, s in enumerate(sigmas):
        gk = gaussian_kernel(s); fL = fLs[s]; fH = f - fL
        show(ax[i, 0], gk, "Gaussian kernel  sigma=%.1f" % s, cmap="viridis")
        show(ax[i, 1], fL, "blurred f_L  sigma=%.1f" % s)
        show(ax[i, 2], fH, "high-freq f_H = f - f_L  sigma=%.1f" % s)
    fig.suptitle("B5 components (depend on sigma only; k is not used here)", y=1.0)
    F["B5_1"] = save(fig, "B5_1_components")

    # (2) sharpened results: original + sigma across columns, k across rows
    fig, ax = plt.subplots(len(ks), len(sigmas) + 1, figsize=(13, 8))
    for ri, k in enumerate(ks):
        show(ax[ri, 0], f, "original f  (k=%.1f row)" % k)
        for ci, s in enumerate(sigmas):
            g = (1 + k) * f - k * fLs[s]
            show(ax[ri, ci + 1], np.clip(g, 0, 255), "sharpened  sigma=%.1f, k=%.1f" % (s, k))
    fig.suptitle("B5 sharpened g=(1+k)f - k f_L   (sigma -> columns, k -> rows)", y=1.0)
    F["B5_2"] = save(fig, "B5_2_sharpened")

    # (3) spectra for a representative setting
    s0, k0 = 3.0, 2.0
    fL = fLs[s0]; fH = f - fL; g = (1 + k0) * f - k0 * fL
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    show(ax[0], spectrum_of_image(f), "|F| original", cmap=SPEC)
    show(ax[1], spectrum_of_image(fL), "|F| blurred f_L", cmap=SPEC)
    show(ax[2], spectrum_of_image(fH), "|F| high-freq f_H", cmap=SPEC)
    show(ax[3], spectrum_of_image(g), "|F| sharpened g", cmap=SPEC)
    fig.suptitle("B5 spectra  (representative: sigma=%.1f, k=%.1f)" % (s0, k0), y=1.02)
    F["B5_3"] = save(fig, "B5_3_spectra")

    # (4) compare unsharp masking vs the fixed sharpening kernel h_s
    g_unsharp = (1 + 1.0) * f - 1.0 * fLs[1.0]   # sigma=1, k=1
    g_fixed = conv2d(f, hs)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    show(ax[0], f, "original f")
    show(ax[1], np.clip(g_fixed, 0, 255), "fixed kernel h_s (B2)")
    show(ax[2], np.clip(g_unsharp, 0, 255), "unsharp  sigma=1.0, k=1.0")
    fig.suptitle("B5 vs fixed kernel: unsharp masking lets sigma(band) and k(strength) be tuned independently", y=1.02)
    F["B5_4"] = save(fig, "B5_4_vs_fixed")

    return res


def _print(m):
    print("=" * 60); print("Part B  Quantitative results"); print("=" * 60)
    print("input size            : %dx%d" % m["shape"])
    print("[Blur]    spatial vs freq:  MSE=%.3e  PSNR=%.1f dB  SSIM=%.6f" % (m["blur_mse"], m["blur_psnr"], m["blur_ssim"]))
    print("[Sharpen] spatial vs freq:  MSE=%.3e  PSNR=%.1f dB  SSIM=%.6f" % (m["sharp_mse"], m["sharp_psnr"], m["sharp_ssim"]))
    print("|H_b|: DC=%.3f edge=%.3f  (decays -> low-pass)" % (m["Hb_dc"], m["Hb_edge"]))
    print("|H_s|: DC=%.3f edge=%.3f  (grows  -> high-pass)" % (m["Hs_dc"], m["Hs_edge"]))
    print("impulse h*delta=h error:  blur=%.1e  sharpen=%.1e" % (m["impulse_err_blur"], m["impulse_err_sharpen"]))
    print("=" * 60)


if __name__ == "__main__":
    r = run_all()
    _print(r["metrics"])
    print("figures saved to '%s/':" % OUT_DIR)
    for k, v in r["figures"].items():
        print("  [%s] %s" % (k, v))
