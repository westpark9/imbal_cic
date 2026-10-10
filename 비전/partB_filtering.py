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
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "spatial_frequency_filtering_input.png")
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = str(BASE_DIR / "output")
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

def conv2d_fft(f, h, return_spectra=False):
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
    g_s = conv2d(f, h)
    g_f, F, H, G = conv2d_fft(f, h, return_spectra=True)
    diff = np.abs(g_s-g_f)
    sf, sh, sg = logmag(F), logmag(H), logmag(G)
    top = max(sf.max(), sg.max())
    fig, ax = plt.subplots(2,4,figsize=(16,8),layout="constrained")
    show(ax[0,0], f, "Input f", vmin=0, vmax=255)
    show(ax[0,1], sf, "log(1+|F|), padded grid", cmap=SPEC, vmin=0, vmax=top)
    show(ax[0,2], h, "Impulse response " + hname, vmin=min(0,h.min()), vmax=h.max())
    for (i,j), v in np.ndenumerate(h):
        ax[0,2].text(j,i,"%.3f" % v,ha="center",va="center",color="red",fontsize=11)
    show(ax[0,3], sh, "log(1+|H|), padded grid", cmap=SPEC, vmin=0, vmax=sh.max())
    show(ax[1,0], np.clip(g_s,0,255), "Spatial result (display clipped)", vmin=0,vmax=255)
    show(ax[1,1], np.clip(g_f,0,255), "FFT result (same crop)", vmin=0,vmax=255)
    show(ax[1,2], sg, "log(1+|G|), G=H F before crop", cmap=SPEC,vmin=0,vmax=top)
    show(ax[1,3], diff/1e-12, "Absolute difference / 1e-12",cmap="magma",vmin=0,vmax=1.2)
    fig.colorbar(ax[1,3].images[0],ax=ax[1,3],shrink=.8,label="Error in units of 1e-12")
    fig.suptitle("F and G share a log scale; max absolute error = %.3e" % diff.max())
    return save(fig,tag), g_s, g_f


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
    for name, h, tag in [("Blur h_b", hb, "blur"), ("Sharpen h_s", hs, "sharpen")]:
        _, Fs, Hs, Gs = conv2d_fft(f, h, return_spectra=True)
        spectra = [logmag(Fs),logmag(Hs),logmag(Gs)]
        vmax = max(spectra[0].max(),spectra[2].max())
        fig, ax = plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
        for j,title in enumerate(["log(1+|F|)","log(1+|H|)","log(1+|G|), G=H F"]):
            show(ax[j],spectra[j],title,cmap=SPEC,vmin=0,
                 vmax=spectra[1].max() if j==1 else vmax)
            fig.colorbar(ax[j].images[0],ax=ax[j],shrink=.8)
        fig.suptitle(name + " - same padded grid; F and G share the scale")
        F["B3_"+tag] = save(fig,"B3_spectra_"+tag)

    # ---- B4 Impulse-response verification ----
    sz = 31
    delta = np.zeros((sz, sz)); delta[sz//2, sz//2] = 1.0
    c = sz // 2; r = 4  # crop half-size for visibility
    for name, h, tag in [("Blur", hb, "blur"), ("Sharpen", hs, "sharpen")]:
        out = conv2d(delta, h)
        out_fft = conv2d_fft(delta, h)
        met["impulse_fft_err_"+tag] = float(np.max(np.abs(out-out_fft)))
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
        show(ax[3], h, "impulse response h", vmin=min(0,h.min()), vmax=h.max())
        for (hi,hj), value in np.ndenumerate(h):
            ax[3].text(hj,hi,"%.3f" % value,ha="center",va="center",color="red",fontsize=10)
        show(ax[4], spectrum_of_kernel(h, (sz, sz)), "log|H(u,v)|", cmap=SPEC)
        fig.suptitle("%s :  max|output_center - h| = %.2e" % (name, err), y=1.05)
        F["B4_" + tag] = save(fig, "B4_impulse_" + tag)
        met["impulse_err_" + tag] = err

    # ---- B5 Gaussian-based unsharp masking ----
    COLS = ["original f", "Gaussian kernel", "blurred f_L", "high-freq f_H", "sharpened g"]

    def components(sigma):
        gk = gaussian_kernel(sigma); fL = conv2d(f, gk); fH = f - fL
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
                show(ax[ri, c], imgs[c], "%s  [%s]" % (COLS[c], lbl), cmap=cmaps[c])
            specs = [spectrum_of_image(f), spectrum_of_kernel(gk, f.shape),
                     spectrum_of_image(fL), spectrum_of_image(fH), spectrum_of_image(g)]
            for c in range(5):
                show(ax[ri + 1, c], specs[c], "|F| of " + COLS[c], cmap=SPEC)
        fig.suptitle(title, y=1.0)
        plt.tight_layout()
        return save(fig, tag)

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
    g_fixed = conv2d(f, hs)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    show(ax[0], f, "original f")
    show(ax[1], np.clip(g_fixed, 0, 255), "fixed 3x3 kernel h_s")
    show(ax[2], np.clip(g_unsharp, 0, 255), "unsharp sigma=3, k=1")
    fig.suptitle("B5 comparison: fixed frequency gain vs adjustable Gaussian scale and strength", y=1.02)
    F["B5_vs_fixed"] = save(fig, "B5_3_vs_fixed")

    # B5 verification in both domains, for every sigma/k combination.
    met["unsharp_comparison"] = []
    fig, ax = plt.subplots(4,3,figsize=(12,14),layout="constrained")
    for row,(sigma,k) in enumerate([(1.,1.),(1.,2.),(3.,1.),(3.,2.)]):
        gk = gaussian_kernel(sigma)
        fLs = conv2d(f,gk); fLf = conv2d_fft(f,gk)
        us = (1+k)*f-k*fLs; uf = (1+k)*f-k*fLf
        met["unsharp_comparison"].append([sigma,k,gk.shape[0],mse(us,uf),
                                           psnr(us,uf,255),ssim(us,uf,255),
                                           float(np.max(np.abs(us-uf))),
                                           float(100*np.mean((us<0)|(us>255)))])
        label = "sigma=%.0f, k=%.0f" % (sigma,k)
        show(ax[row,0],np.clip(us,0,255),"Spatial: "+label,vmin=0,vmax=255)
        show(ax[row,1],np.clip(uf,0,255),"FFT: "+label,vmin=0,vmax=255)
        show(ax[row,2],np.abs(us-uf)/1e-12,"Difference / 1e-12",cmap=SPEC,vmin=0,vmax=1.2)
        fig.colorbar(ax[row,2].images[0],ax=ax[row,2],shrink=.8)
    F["B5_domains"] = save(fig,"B5_4_domains")
    met["mean_input"] = float(f.mean())
    met["mean_sharpen"] = float(gs_s.mean())
    return res


def _print(m):
    print("=" * 60); print("Part B  Quantitative results"); print("=" * 60)
    print("input size            : %dx%d" % m["shape"])
    print("[Blur]    spatial vs freq:  MSE=%.3e  PSNR=%.1f dB  SSIM=%.6f" % (m["blur_mse"], m["blur_psnr"], m["blur_ssim"]))
    print("[Sharpen] spatial vs freq:  MSE=%.3e  PSNR=%.1f dB  SSIM=%.6f" % (m["sharp_mse"], m["sharp_psnr"], m["sharp_ssim"]))
    print("impulse h*delta=h error:  blur=%.1e  sharpen=%.1e" % (m["impulse_err_blur"], m["impulse_err_sharpen"]))
    print("B5: sigma k size MSE PSNR SSIM max_error out_of_range_percent")
    for row in m["unsharp_comparison"]:
        print("%.0f %.0f %d %.3e %.2f %.8f %.3e %.3f" % tuple(row))
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
    with open(BASE_DIR / "results_B.json", "w", encoding="utf-8") as stream:
        json.dump(portable_results(r), stream, indent=2, default=float)
    _print(r["metrics"])
    print("figures saved to '%s/':" % OUT_DIR)
    for k, v in r["figures"].items():
        print("  [%s] %s" % (k, v))
