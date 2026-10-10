# -*- coding: utf-8 -*-
"""Build the reviewed notebook. Execution outputs are generated separately."""
from pathlib import Path
import argparse
import nbformat as nbf

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

# Cell 1
md(r"""
# Homework 1 — Computer Vision

제출 코드는 `homework1_cv.py`를 참고하며, 실행 결과 그림은 `output/` 폴더에 저장된다. 이 노트북은 A/B/C 결과를 확인하기 위한 검토용 자료이다.
""")

# Cell 2
code(r"""
# ---- Common imports & global settings ----
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, time

# 그림을 노트북 안에 인라인으로 표시
%matplotlib inline
rng = np.random.default_rng(0)   # 재현성(reproducibility)을 위한 고정 시드 (Part C 잡음 생성에 사용)

IMG_DIR = "images"
print("numpy", np.__version__)
print("files:", os.listdir(IMG_DIR))
""")

# Cell 3
code(r"""
# ---- Common helper: 이미지 표시/저장 & 정량지표(metrics) from scratch ----

OUT_DIR = "output"                       # display된 그림(이미지/히스토그램)을 문제번호별로 저장
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(tag):
    # 현재 figure를 output/<문제번호>_<이름>.png 로 저장 (plt.show() 직전에 호출)
    plt.savefig(os.path.join(OUT_DIR, tag + ".png"), dpi=130, bbox_inches='tight')

def show(ax, img, title="", cmap='gray', vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def to_uint8(img):
    # 표시용: [0,255]로 클리핑 후 uint8 캐스팅 (중간계산엔 사용하지 않음)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)

# ----- 2D convolution (zero-padding, 'same' output) : from scratch -----
def conv2d(f, h):
    # 선형 합성곱(linear convolution): 커널을 뒤집어(flip) 미끄러뜨림
    f = f.astype(np.float64); h = h.astype(np.float64)
    kh, kw = h.shape
    ph, pw = kh // 2, kw // 2
    hf = h[::-1, ::-1]                      # convolution => kernel flip
    fp = np.pad(f, ((ph, ph), (pw, pw)))   # zero padding
    out = np.zeros_like(f)
    for i in range(kh):
        for j in range(kw):
            out += hf[i, j] * fp[i:i+f.shape[0], j:j+f.shape[1]]
    return out

# ----- 정량 지표: MSE / PSNR / SSIM (from scratch) -----
def mse(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def psnr(a, b, L):
    m = mse(a, b)
    return float('inf') if m == 0 else 10.0 * np.log10((L ** 2) / m)

def _gauss_window(size=11, sigma=1.5):
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2)); g /= g.sum()
    return np.outer(g, g)

def ssim(a, b, L):
    # Wang et al.(2004) SSIM: 11x11 Gaussian 가중 국소통계 사용
    a = a.astype(np.float64); b = b.astype(np.float64)
    w = _gauss_window(11, 1.5)
    C1 = (0.01 * L) ** 2; C2 = (0.03 * L) ** 2
    mu_a = conv2d(a, w); mu_b = conv2d(b, w)
    mua2, mub2, muab = mu_a**2, mu_b**2, mu_a*mu_b
    va = conv2d(a*a, w) - mua2
    vb = conv2d(b*b, w) - mub2
    vab = conv2d(a*b, w) - muab
    smap = ((2*muab + C1) * (2*vab + C2)) / ((mua2 + mub2 + C1) * (va + vb + C2))
    return float(smap.mean())

# 동작 확인(sanity check)
_t = np.array(Image.open(os.path.join(IMG_DIR, "spatial_frequency_filtering_input.png")).convert('L'), dtype=np.float64)
print("SSIM(x,x) =", ssim(_t, _t, 255))
""")

# Cell 4
md(r"""
# Part A. Point Processing and Histogram-Based Image Enhancement
핵심 연산은 아래 셀에서 직접 구현한다. Y를 변경하고 U,V를 유지하며, 최종 RGB 범위 조정 때문에 색상·채도까지 완전히 보존되는 것은 아니다. 히스토그램/비트평면용 양자화에는 정수, 필요한 산술에는 float64를 사용한다.

""")

# Cell 5
code(r"""
# ---- Part A 입력 로드 ----
rgb = np.array(Image.open(os.path.join(IMG_DIR, "point_processing_input_rgb.png")).convert('RGB'), dtype=np.float64)  # [0,255] float
print("RGB shape:", rgb.shape, "range:", rgb.min(), rgb.max())
M, N = rgb.shape[:2]
R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
""")

# Cell 6
md(r"""
## A-1. RGB → YUV Conversion
주어진 행렬(BT.601 계열)을 **직접** 적용한다.

$$\begin{bmatrix}Y\\U\\V\end{bmatrix}=
\begin{bmatrix}0.299&0.587&0.114\\-0.14713&-0.28886&0.436\\0.615&-0.51499&-0.10001\end{bmatrix}
\begin{bmatrix}R\\G\\B\end{bmatrix}$$

- 입력 $R,G,B\in[0,255]$ 이므로 가중치 합이 1인 첫 행에 의해 **$Y\in[0,255]$** (휘도, luminance).
- $U,V$ 는 색차(chrominance)로 **음수 가능** → 클리핑/8bit 캐스팅 없이 float로 유지한다(지시사항).
""")

# Cell 7
code(r"""
M_fwd = np.array([[ 0.299,    0.587,    0.114  ],
                  [-0.14713, -0.28886,  0.436  ],
                  [ 0.615,   -0.51499, -0.10001]])
M_inv = np.linalg.inv(M_fwd)   # 역변환 행렬 (기본 배열연산 = 허용)

def rgb_to_yuv(img):
    # img: (H,W,3) float  ->  Y,U,V 각각 (H,W) float
    flat = img.reshape(-1, 3).T          # (3, HW)
    yuv = M_fwd @ flat                   # (3, HW)
    yuv = yuv.T.reshape(img.shape)
    return yuv[..., 0], yuv[..., 1], yuv[..., 2]

def yuv_to_rgb(Y, U, V):
    yuv = np.stack([Y, U, V], axis=-1)
    flat = yuv.reshape(-1, 3).T
    rgb = (M_inv @ flat).T.reshape(yuv.shape)
    return rgb

Y, U, V = rgb_to_yuv(rgb)
print("Y range [%.2f, %.2f]  (기대: 0~255)" % (Y.min(), Y.max()))
print("U range [%.2f, %.2f]  (음수 포함)" % (U.min(), U.max()))
print("V range [%.2f, %.2f]  (음수 포함)" % (V.min(), V.max()))

fig, ax = plt.subplots(1, 4, figsize=(15, 4))
show(ax[0], to_uint8(rgb), "Original RGB", cmap=None)
show(ax[1], Y, "Y (luminance)", vmin=0, vmax=255)
show(ax[2], U, "U = B-Y (chroma)")   # 자동 스케일로 부호 표시
show(ax[3], V, "V = R-Y (chroma)")
plt.tight_layout(); savefig("A1_rgb_yuv"); plt.show()
""")

# Cell 8
md(r"""
**해석**
Y는 밝기 구조를, U와 V는 각각 B−Y와 R−Y에 거의 비례하는 색차를 나타낸다. 계수는 반올림되어 있으며 상관계수 1.000은 표시 정밀도에서의 결과이다. U·V는 음수도 포함하고, 시각화의 밝기는 자동 스케일된 색차 크기를 뜻한다.

""")

# Cell 9
md(r"""
## A-2. Histogram Computation
휘도 $Y$ 를 8-bit로 양자화($r_k\in\{0,\dots,255\}$) 후 **직접** 계수한다.
$$h(r_k)=n_k,\qquad p(r_k)=\frac{n_k}{MN}$$
- `np.histogram` 미사용. 각 레벨 $k$ 에 대해 `np.sum(Yq==k)` 로 직접 집계.
""")

# Cell 10
code(r"""
def quantize8(img):
    return np.clip(np.round(img), 0, 255).astype(np.int64)

def compute_hist(q):
    # built-in histogram 미사용: 레벨별 직접 집계
    h = np.zeros(256, dtype=np.float64)
    for k in range(256):
        h[k] = np.sum(q == k)
    return h

Yq = quantize8(Y)
hist = compute_hist(Yq)
p = hist / (M * N)                 # normalized histogram
print("sum(h) =", hist.sum(), " (= MN =", M*N, ")")
print("sum(p) =", round(p.sum(), 6))

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
show(ax[0], Yq, "Input luminance Y", vmin=0, vmax=255)
ax[1].bar(np.arange(256), hist, width=1.0); ax[1].set_title("Histogram  h(r_k)=n_k"); ax[1].set_xlabel("intensity r_k")
ax[2].bar(np.arange(256), p, width=1.0, color='C1'); ax[2].set_title("Normalized histogram  p(r_k)=n_k/MN"); ax[2].set_xlabel("intensity r_k")
plt.tight_layout(); savefig("A2_histogram"); plt.show()
""")

# Cell 11
md(r"""
**해석**
- **코드-수식 대응**: 반복문의 `k` 가 곧 강도 $r_k=k\ (k=0,\dots,255)$ 이고, `hist[k]=np.sum(q==k)` 가 그 강도를 갖는 화소 수 $n_k$ 다. 검증: $\sum_k h(r_k)=MN=262144$ (전체 화소수).
- **왜 정규화 전후 히스토그램 모양이 같은가**: 정규화는 모든 막대를 **같은 상수 $MN$ 으로 나누는 것**뿐이다. 모든 값에 동일 상수를 곱(÷)하면 막대들의 **상대 비율은 그대로**이고 **세로축 눈금만** 바뀐다(왼쪽: 화소 수 $0\sim$수천, 오른쪽: 확률 $0\sim0.0x$, 합=1). 즉 $p(r_k)=n_k/MN$ 은 히스토그램을 **확률분포(PDF)** 로 재해석한 것일 뿐 분포의 형태는 동일하다.
- 분포가 특정 밝기 구간에 **몰려 있을수록 대비가 낮다**. 이 분포 형태가 이후 평활화/스트레칭의 대상이 된다.
""")

# Cell 12
md(r"""
## A-3. Histogram Equalization (HE)
누적분포 $CDF(r_k)=\sum_{j=0}^{k} p(r_j)$ 를 변환함수로 사용한다.
$$s_k=(L-1)\,CDF(r_k),\quad L=256$$
이는 분포를 **전역적으로(globally)** 균일(uniform)에 가깝게 펴서 대비를 늘린다.
HE된 Y를 원본 U,V와 결합해 RGB로 역변환하여 색을 보존한 향상 결과를 얻는다.
""")

# Cell 13
code(r"""
def hist_equalize(q):
    h = compute_hist(q)
    cdf = np.cumsum(h) / h.sum()          # CDF in [0,1]
    lut = np.round((256 - 1) * cdf)       # s_k = (L-1)*CDF
    return lut[q], lut, cdf

Y_eq, lut_he, cdf_he = hist_equalize(Yq)
hist_eq = compute_hist(quantize8(Y_eq))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
show(ax[0,0], Yq, "Original Y", vmin=0, vmax=255)
ax[0,1].bar(np.arange(256), hist, width=1.0); ax[0,1].set_title("Original histogram")
show(ax[1,0], Y_eq, "Equalized Y", vmin=0, vmax=255)
ax[1,1].bar(np.arange(256), hist_eq, width=1.0, color='C2'); ax[1,1].set_title("Equalized histogram")
plt.tight_layout(); savefig("A3_1_Y_equalization"); plt.show()

# 평활화된 Y + 원본 U,V -> RGB 역변환
rgb_eq = yuv_to_rgb(Y_eq, U, V)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
show(ax[0], to_uint8(rgb), "Original RGB", cmap=None)
show(ax[1], to_uint8(rgb_eq), "HE-enhanced RGB (Y only)", cmap=None)
ax[2].plot(np.arange(256), lut_he); ax[2].set_title("HE transform  s=T(r)"); ax[2].set_xlabel("r"); ax[2].set_ylabel("s"); ax[2].grid(True)
plt.tight_layout(); savefig("A3_2_RGB_compare"); plt.show()
""")

# Cell 14
md(r"""
**해석**
HE는 입력 빈도가 높은 밝기 구간을 더 넓게 매핑한다. 정수 LUT의 건너뜀과 여러 입력 레벨의 병합 때문에 출력 히스토그램은 완전히 균일하지 않다. 양자화 전 레벨당 증가량은 255·p(k)이다.
이번 영상의 평균은 115.4→129.5, 표준편차는 75.1→71.4이다. 밝기는 증가했지만 전체 표준편차는 줄었다. 순흑 입력은 약 28의 어두운 회색으로 올라간다. U,V를 유지한 처리이며 최종 RGB clipping까지 고려하면 색상·채도 불변을 보장하지 않는다.

""")

# Cell 15
md(r"""
## A-4. Contrast Stretching (piecewise-linear)
$r_{\min},r_{\max}$ 를 휘도 히스토그램의 **2·98 백분위수(percentile)** 로 선택(이상치에 강건).
$$s=\begin{cases}0,&r<r_{\min}\\[2pt]\dfrac{r-r_{\min}}{r_{\max}-r_{\min}}(L-1),&r_{\min}\le r\le r_{\max}\\[6pt]L-1,&r>r_{\max}\end{cases}$$
**파라미터**: 2nd/98th percentile.
""")

# Cell 16
code(r'''
L = 256
def contrast_stretch(img, p_low=2, p_high=98):
    """Choose percentile levels from the 8-bit luminance histogram CDF."""
    if not 0 <= p_low < p_high <= 100:
        raise ValueError("Require 0 <= p_low < p_high <= 100")
    h = compute_hist(quantize8(img)); cdf = np.cumsum(h) / h.sum()
    rmin = float(np.flatnonzero(cdf >= p_low / 100.0)[0])
    rmax = float(np.flatnonzero(cdf >= p_high / 100.0)[0])
    if rmax <= rmin:
        return img.astype(np.float64).copy(), rmin, rmax
    return np.clip((img-rmin)/(rmax-rmin)*(L-1), 0, L-1), rmin, rmax

Y_cs, rmin, rmax = contrast_stretch(Y, 2, 98)
print("r_min(2%%) = %.2f,  r_max(98%%) = %.2f" % (rmin, rmax))
hist_cs = compute_hist(quantize8(Y_cs))
rgb_cs = yuv_to_rgb(Y_cs, U, V)

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
show(ax[0,0], to_uint8(rgb), "Original RGB", cmap=None)
show(ax[0,1], to_uint8(rgb_cs), "Contrast-stretched RGB", cmap=None)
ax[1,0].bar(np.arange(256), hist, width=1.0); ax[1,0].set_title("Histogram (before)")
ax[1,1].bar(np.arange(256), hist_cs, width=1.0, color='C3'); ax[1,1].set_title("Histogram (after stretch)")
plt.tight_layout(); savefig("A4_contrast_stretch"); plt.show()
''')

# Cell 17
md(r"""
**해석**
양자화된 휘도 히스토그램의 CDF가 0.02, 0.98에 처음 도달하는 레벨을 선택한다. 이번 입력은 r_min=0, r_max=235, 선형 이득=255/235≈1.085이다. 중앙 범위는 선형 확장되고 양 끝은 포화되므로 전체 분포 모양이 정확히 그대로 보존되는 것은 아니다. HE는 밝기 구간마다 이득이 다른 CDF 매핑이다.

""")

# Cell 18
md(r"""
## A-5. Gamma Correction (power-law)
일반형은 $s=c\,r^{\gamma}$ 이다. 여기서 **$c$ 는 전체 출력 크기를 키우거나 줄이는 스케일(gain) 상수**로,
$c=1$ 이면 추가 스케일링 없이 정규화 입력 $[0,1]$ 을 그대로 $[0,1]$ 로 사상한다(출력 범위를 $[0,255]$ 로만 되돌림). $c\ne1$ 이면 전체가 비례해 더 밝거나 어두워진다. 본 과제는 $c=1$.
$$s=255\left(\frac{r}{255}\right)^{\gamma}$$
**파라미터**: $\gamma\in\{0.5,\,1.0,\,2.0\}$. Y에 적용 후 RGB 재구성.
""")

# Cell 19
code(r"""
def gamma_correct(img, gamma, L=256):
    rn = img / (L - 1)                    # normalize to [0,1]
    sn = np.power(np.clip(rn, 0, 1), gamma)
    return sn * (L - 1)

gammas = [0.5, 1.0, 2.0]
fig, ax = plt.subplots(2, len(gammas), figsize=(13, 8))
rr = np.arange(256)
for i, g in enumerate(gammas):
    Yg = gamma_correct(Y, g)
    rgbg = yuv_to_rgb(Yg, U, V)
    show(ax[0, i], to_uint8(rgbg), "gamma = %.1f" % g, cmap=None)
    ax[1, i].plot(rr, 255 * (rr / 255.0) ** g); ax[1, i].plot(rr, rr, 'k--', lw=0.8)
    ax[1, i].set_title("transform s=T(r), gamma=%.1f" % g); ax[1, i].set_xlabel("r"); ax[1, i].set_ylabel("s"); ax[1, i].grid(True)
plt.tight_layout(); savefig("A5_gamma"); plt.show()

def print_luminance_stats(items):
    print("Method          Mean        Global std  Mean tile std")
    for name,v in items:
        tile_std = [v[i*64:(i+1)*64,j*64:(j+1)*64].std() for i in range(8) for j in range(8)]
        print("%-15s %10.3f %10.3f %13.3f" % (name,v.mean(),v.std(),np.mean(tile_std)))

print_luminance_stats([("Original",Y)]+[("Gamma %.1f"%v,gamma_correct(Y,v)) for v in [0.5,1.,2.]])

""")

# Cell 20
md(r"""
**해석**
정규화 좌표에서는 s_n=r_n^γ, c=1이다. γ<1은 암부를 밝게 하며 암부의 차이와 기존 잡음도 증폭할 수 있다. γ=1은 항등이고, γ>1은 전체를 어둡게 하면서 밝은 구간의 차이를 확대할 수 있다. 원래 강도 단위의 기울기는 γ(r/255)^(γ−1)이다.
통계표는 평균, 전체 표준편차, 64×64 타일 64개의 표준편차 평균이다. 타일 표준편차는 구조·대비·잡음을 함께 반영하므로 잡음만의 지표는 아니다.

""")

# Cell 21
md(r"""
## A-6. Adaptive Histogram Equalization (AHE)
영상을 타일(tile)로 나누어 **각 타일마다 독립적으로 HE** 를 수행한다(국소 대비 향상).
**파라미터**: 타일 격자 8×8 (각 타일 64×64).
""")

# Cell 22
code(r"""
def pad_to_multiple(img, nt):
    # 타일로 정확히 나뉘도록 가장자리 복제(edge replicate) 패딩
    H, W = img.shape
    Hs = int(np.ceil(H / nt) * nt); Ws = int(np.ceil(W / nt) * nt)
    return np.pad(img, ((0, Hs - H), (0, Ws - W)), mode='edge'), H, W

def ahe(img, nt=8):
    q, H0, W0 = pad_to_multiple(quantize8(img), nt)
    H, W = q.shape; th, tw = H // nt, W // nt
    out = np.zeros_like(q, dtype=np.float64)
    for ti in range(nt):
        for tj in range(nt):
            tile = q[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw]
            h = compute_hist(tile)
            lut = np.round((256 - 1) * (np.cumsum(h) / h.sum()))
            out[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw] = lut[tile]
    return out[:H0, :W0]

Y_ahe = ahe(Y, nt=8)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
show(ax[0], Yq, "Original Y", vmin=0, vmax=255)
show(ax[1], Y_eq, "Global HE", vmin=0, vmax=255)
show(ax[2], Y_ahe, "AHE (8x8 tiles)", vmin=0, vmax=255)
plt.tight_layout(); savefig("A6_AHE"); plt.show()
print_luminance_stats([("Original",Y),("HE",Y_eq),("AHE",Y_ahe)])

""")

# Cell 23
md(r"""
**해석**
각 타일의 국소 CDF로 대비가 커지지만 좁은 강도 구간의 잡음·미세 변동도 확대된다. 독립 타일을 보간 없이 처리하므로 타일 경계에 불연속이 생긴다. 표의 타일 표준편차 증가는 국소 변동의 증가를 나타내며, 잡음과 영상 구조를 분리한 측정은 아니다.

""")

# Cell 24
md(r"""
## A-7. CLAHE (Contrast Limited AHE)
AHE의 잡음 증폭/블로킹을 완화: (1) 타일 히스토그램을 **클립 한계(clip limit)** 로 자른 뒤,
(2) 잘린 양을 **균등 재분배(redistribute)**, (3) 국소 CDF로 매핑, (4) 타일 중심 간 **쌍선형 보간(bilinear interpolation)** 으로 경계 아티팩트 제거.
**파라미터**: 타일 8×8, clip limit 두 값 비교(타일 화소수 대비 비율 `clip∈{0.01, 0.05}`).
""")

# Cell 25
code(r"""
def clahe(img, nt=8, clip=0.01):
    q, H0, W0 = pad_to_multiple(quantize8(img), nt)
    H, W = q.shape; th, tw = H // nt, W // nt
    npix = th * tw
    clip_count = max(1, int(clip * npix))     # 절대 클립 한계(개수)

    # 1~5단계: 타일별 LUT(매핑) 계산
    luts = np.zeros((nt, nt, 256))
    for ti in range(nt):
        for tj in range(nt):
            tile = q[ti*th:(ti+1)*th, tj*tw:(tj+1)*tw]
            h = compute_hist(tile)
            excess = np.maximum(h - clip_count, 0).sum()   # 잘린 총량
            h = np.minimum(h, clip_count)                  # clip
            h += excess / 256.0                            # 균등 재분배
            cdf = np.cumsum(h) / h.sum()
            luts[ti, tj] = np.round((256 - 1) * cdf)

    # 7단계: 타일 중심 기준 쌍선형 보간 (vectorized)
    cy = (np.arange(nt) + 0.5) * th
    cx = (np.arange(nt) + 0.5) * tw
    yy = np.clip((np.arange(H) - cy[0]) / th, 0, nt - 1)
    xx = np.clip((np.arange(W) - cx[0]) / tw, 0, nt - 1)
    i0 = np.clip(np.floor(yy).astype(int), 0, nt - 1); i1 = np.clip(i0 + 1, 0, nt - 1)
    j0 = np.clip(np.floor(xx).astype(int), 0, nt - 1); j1 = np.clip(j0 + 1, 0, nt - 1)
    wy = (yy - i0)[:, None]; wx = (xx - j0)[None, :]
    I0 = np.broadcast_to(i0[:, None], (H, W)); I1 = np.broadcast_to(i1[:, None], (H, W))
    J0 = np.broadcast_to(j0[None, :], (H, W)); J1 = np.broadcast_to(j1[None, :], (H, W))
    A = luts[I0, J0, q]; Bb = luts[I0, J1, q]; Cc = luts[I1, J0, q]; D = luts[I1, J1, q]
    out = (1-wy)*((1-wx)*A + wx*Bb) + wy*((1-wx)*Cc + wx*D)
    return out[:H0, :W0]

t0 = time.time()
Y_clahe1 = clahe(Y, nt=8, clip=0.01)
Y_clahe2 = clahe(Y, nt=8, clip=0.05)
print("CLAHE done in %.2fs" % (time.time() - t0))

images = [Yq, Y_ahe, Y_clahe1, Y_clahe2]
titles = ["Original Y", "AHE", "CLAHE clip=0.01", "CLAHE clip=0.05"]
hists = [compute_hist(quantize8(v)) for v in images]
ymax = 1.05*max(h.max() for h in hists)
fig, ax = plt.subplots(2,4,figsize=(16,8))
for j,(v,title,h) in enumerate(zip(images,titles,hists)):
    show(ax[0,j],v,title,vmin=0,vmax=255)
    ax[1,j].bar(np.arange(256),h,width=1)
    ax[1,j].set(xlim=(0,255),ylim=(0,ymax),title="Histogram: "+title,xlabel="Intensity",ylabel="Pixel count")
plt.tight_layout(); savefig("A7_CLAHE"); plt.show()
print_luminance_stats([("Original",Y),("AHE",Y_ahe),("CLAHE 0.01",Y_clahe1),("CLAHE 0.05",Y_clahe2)])

""")

# Cell 26
md(r"""
**해석**
CLAHE는 타일 히스토그램 clipping으로 과도한 국소 CDF 기울기를 완화하고, 타일 중심 사이의 보간으로 경계 불연속을 줄인다. clip_count는 40 또는 204이다. 초과량 E를 한 번 균등 재분배하여 h_final=min(h,clip_count)+E/256으로 쓰므로 재분배 후에는 원래 count를 넘을 수 있다.
원본·AHE·두 CLAHE 결과의 전역 히스토그램을 동일한 세로축으로 비교한다. 타일 입력 히스토그램의 clipping은 최종 전역 출력 히스토그램의 봉우리를 제한하지 않는다. 큰 clip은 보간된 AHE에 접근하며, 본 실험의 보간 없는 AHE와 완전히 동일해지는 것은 아니다.

""")

# Cell 27
md(r"""
## A-8. Bit-Plane Slicing
8-bit 휘도를 비트평면으로 분해: $Y(x,y)=\sum_{k=0}^{7} b_k(x,y)\,2^{k}$.
상위 4개(MSB) 평면만으로 재구성하여 원본과 비교한다.
""")

# Cell 28
code(r"""
Yb = quantize8(Y).astype(np.uint8)
fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
planes = []
for k in range(8):
    bk = (Yb >> k) & 1                 # k번째 비트평면 (0/1)
    planes.append(bk)
    r, c = divmod(7 - k, 4)            # MSB(k=7)를 좌상단부터
    show(ax[r, c], bk, "bit plane %d%s" % (k, " (MSB)" if k==7 else (" (LSB)" if k==0 else "")), vmin=0, vmax=1)
plt.tight_layout(); savefig("A8_1_bit_planes"); plt.show()

# 상위 4개 평면(k=4..7)으로 재구성
recon4 = np.zeros_like(Yb, dtype=np.float64)
for k in range(4, 8):
    recon4 += ((Yb >> k) & 1).astype(np.float64) * (2 ** k)
print("MSE(Y, recon4) = %.2f,  PSNR = %.2f dB" % (mse(Yb, recon4), psnr(Yb, recon4, 255)))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
show(ax[0], Yb, "Original Y (8 bits)", vmin=0, vmax=255)
show(ax[1], recon4, "Reconstructed (4 MSB planes)", vmin=0, vmax=255)
plt.tight_layout(); savefig("A8_2_reconstruction_4MSB"); plt.show()
""")

# Cell 29
md(r"""
**해석**
- **그림 관찰**: bit 7~5 평면만으로도 얼굴·우주복·헬멧이 또렷이 식별되고 bit 4는 질감을 보강한다. 반면 bit 3부터 점점 무작위 점처럼 변하고 bit 0(LSB)은 거의 순수 잡음이다.
- **평면별 역할 (단순히 상위/하위가 아니라)**:
  - **bit 7 (MSB, 가중치 128)**: 영상을 밝음/어두움 두 덩어리로 나눈 가장 거친 구조 — 얼굴·우주복·배경의 큰 윤곽.
  - **bit 6, 5 (가중치 64, 32)**: 주요 형태와 명암 단계 — 여기까지면 무엇인지 거의 알아볼 수 있다.
  - **bit 4 (가중치 16)**: 중간 톤의 음영·질감이 보강됨.
  - **bit 3~2 (가중치 8, 4)**: 미세한 질감/그라데이션 디테일.
  - **bit 1~0 (LSB, 가중치 2, 1)**: 기여가 극히 작아 **거의 무작위(잡음)** 로 보인다 — 미세 텍스처·센서 잡음.
- **"상위 4개(bit 4~7)가 가장 중요"를 어떻게 아는가**: 두 근거로 판단한다.
  1. **정의/가중치**: "most significant"는 가중치가 큰 비트를 뜻하며, bit 4~7의 가중치 합 $16\!+\!32\!+\!64\!+\!128=240$ 은 최대값 255의 약 **94%** 를 차지한다(=값의 대부분을 상위 4비트가 결정).
  2. **정량 검증**: 상위 4비트만으로 복원했을 때 **MSE $\approx67.2$, PSNR $\approx29.9$ dB** (위 출력)로 원본과 수치적으로 가깝고, 육안으로도 거의 구분되지 않는다. 즉 가정이 아니라 지표로 확인된다.
- 다만 하위 4비트를 버려 매끄러운 그라데이션에 계단 모양 **윤곽선(false contour)** 이 약하게 생긴다. (비트평면 기반 영상 압축의 기본 원리.)
""")

# Cell 30
md(r"""
## A-9. Comparison and Discussion

앞의 결과 영상과 히스토그램을 바탕으로 처리 범위, 대비, 잡음, 계산 복잡도와 적용 대상을 비교한다.
""")

# Cell 31
code(r"""
# ---- A-9 방법별 결과 영상과 히스토그램을 개별 파일로 저장 ----
Y_g05 = gamma_correct(Y, 0.5)
methods = [
    ("orig",    "Original Y",       Yq),
    ("stretch", "Contrast stretch", Y_cs),
    ("he",      "Histogram Eq",     Y_eq),
    ("ahe",     "AHE",              Y_ahe),
    ("clahe",   "CLAHE 0.01",       Y_clahe1),
    ("gamma",   "Gamma 0.5",        Y_g05),
]
for key, name, img in methods:
    # 결과 영상 썸네일
    fig = plt.figure(figsize=(2.0, 2.0))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255); plt.axis('off')
    plt.savefig(os.path.join(OUT_DIR, "A9_%s_img.png" % key), dpi=90, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    # 히스토그램 썸네일
    fig = plt.figure(figsize=(2.6, 1.9))
    plt.bar(np.arange(256), compute_hist(quantize8(img)), width=1.0, color='C0')
    plt.xlim(0, 255); plt.yticks([]); plt.xticks([0, 128, 255], fontsize=7)
    plt.savefig(os.path.join(OUT_DIR, "A9_%s_hist.png" % key), dpi=90, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
print("saved A9 result images and histograms to", OUT_DIR)
""")

# Cell 32
md(r"""
**비교 및 논의**
스트레칭·HE·감마는 전역 처리이고 AHE·CLAHE는 국소 처리이다. 스트레칭은 중앙 강도 구간의 일정 이득, HE는 CDF에 따른 구간별 이득, 감마는 밝기에 따른 이득을 제공한다. γ<1은 암부 잡음을 증폭할 수 있다. AHE는 좁은 타일 분포의 잡음과 타일 경계 불연속에 민감하고, CLAHE는 빈도 제한과 보간으로 이를 완화한다.
현재 compute_hist는 L개 레벨마다 전체 화소를 비교하여 O(LMN), 타일 방법은 O(LMN+TL)이다. L=256,T=64를 고정하면 영상 화소수에 선형이다. 단일 패스 계수 구현은 O(MN+L)로 개선 가능하다. 스트레칭은 좁은 동적범위, HE는 전역 대비, 감마는 톤 조정, AHE/CLAHE는 불균일한 국소 대비에 적합하다.

""")

# Cell 33
md(r"""
# Part B. Spatial and Frequency Domain Filtering
두 영역의 zero 경계 선형 합성곱을 비교한다. 구현의 단일 기준은 제출용 `homework1_cv.py`이며, 이 노트북은 같은 함수를 호출해 그림과 지표를 표시한다.
FFT 크기는 (M+m−1,N+n−1)이다. 커널을 좌상단에 배치하고 역변환한 전체 결과의 (m//2,n//2)부터 M×N 크롭하여 정렬한다. F,H,G는 같은 패딩 격자의 배열이고 G=HF는 크롭 전 스펙트럼이다.
SSIM: 11×11 Gaussian 창, σ=1.5, C1=(0.01L)²,C2=(0.03L)², zero-padding과 경계 포함 전체 평균. PSNR peak/SSIM L은 B=255,C=1이다. B는 clipping 전 float 결과끼리 비교한다.

""")

# Cell 34
code(r"""
import homework1_cv as hw
from IPython.display import display, Image as DisplayImage
import importlib
importlib.reload(hw)
b_results = hw.run_part_b()
bm = b_results["metrics"]
print("Input:",bm["shape"],"; kernel sizes 3x3, 7x7, 19x19")
print("Core implementations: homework1_cv.py (b_conv2d, b_conv2d_fft, b_ssim)")
def show_b(key):
    display(DisplayImage(filename=b_results["figures"][key]))

""")

# Cell 35
md(r"""
## B-1. Blur Filtering
h_b는 모든 원소가 1/9인 3×3 커널이다. 입력·커널·공간/FFT 결과와 실제 패딩 격자의 F,H,G 스펙트럼을 표시한다. F와 G의 로그 표시 범위는 동일하다.

""")

# Cell 36
code(r"""
show_b("B1")
print("Blur: MSE=%.3e, PSNR=%.2f dB, SSIM=%.8f" % (bm["blur_mse"],bm["blur_psnr"],bm["blur_ssim"]))

""")

# Cell 37
md(r"""
**해석**
동전 에지와 질감이 두 결과에서 동일하게 약해진다. 차이맵은 오차를 10⁻¹² 단위로 확대 표시하므로 무늬가 보일 수 있다. 최대 차이는 약 3.13×10⁻¹³이고, MSE≈3.61×10⁻²⁷로 반올림 수준이다. 박스 커널은 DC 이득이 1인 저역통과 필터이며 영점과 측엽이 있어 주파수 응답이 단조 감소하지 않는다.

""")

# Cell 38
md(r"""
## B-2. Sharpening Filtering
h_s=[[0,−1,0],[−1,5,−1],[0,−1,0]]이다. 계수합이 1이므로 DC 이득은 1이며 상수 내부 영역을 보존한다. zero 경계와 same 크롭에서는 전체 평균이 달라질 수 있다. 출력 clipping은 표시용이다.

""")

# Cell 39
code(r"""
show_b("B2")
print("Sharpen: MSE=%.3e, PSNR=%.2f dB, SSIM=%.8f" % (bm["sharp_mse"],bm["sharp_psnr"],bm["sharp_ssim"]))
print("Mean input / sharpen:",bm["mean_input"],bm["mean_sharpen"])

""")

# Cell 40
md(r"""
**그림 관찰**: 결과에서 동전 경계·각인 질감이 또렷해지고 에지 주변이 밝게 강조된다(오버슈트). 여기서도 차이맵은 ≈0으로 공간·주파수 결과가 일치.

**해석**: 샤프닝도 공간/주파수 결과가 동일(반올림 제외, 차이맵 참조). 출력은 에지에서 오버/언더슈트로 $[0,255]$를 벗어나 **표시할 때만 클리핑**한다(지표는 원본 float로 계산).
""")

# Cell 41
md(r"""
## B-3. Frequency-Domain Analysis
두 필터에 대해 $|F|,\,|H|,\,|G|=|HF|$ 를 비교한다.
""")

# Cell 42
code(r"""
show_b("B3_blur")
show_b("B3_sharpen")

""")

# Cell 43
md(r"""
**해석**
중심 정렬 좌표에서 H_b=(1+2cosωx)(1+2cosωy)/9이다. DC를 통과시키고 고주파를 전반적으로 약화시키며 영점과 측엽을 갖는다.
H_s=5−2cosωx−2cosωy는 DC에서 1, (π,π)에서 9이다. 따라서 순수 고역통과가 아니라 고역강조이다. 중앙의 어두운 색은 0이 아닌 상대적으로 낮은 이득을 뜻한다. 에지는 급격한 밝기 변화이므로 고주파 강조에 따라 더 뚜렷해진다.

""")

# Cell 44
md(r"""
## B-4. Verification of the Impulse Response
임펄스 $\delta(x,y)$ 에 필터를 적용하면 출력이 곧 임펄스응답 $h$ 가 된다: $h*\delta=h$.
""")

# Cell 45
code(r"""
show_b("B4_blur")
show_b("B4_sharpen")
for tag in ["blur","sharpen"]:
    print(tag,"kernel error:",bm["impulse_err_"+tag],"spatial/FFT error:",bm["impulse_fft_err_"+tag])

""")

# Cell 46
md(r"""
**해석**
31×31 배열 중앙을 표시 좌표의 원점으로 둔다. 배열 인덱스에서는 이동한 임펄스이므로 출력은 같은 위치로 이동한 h이다. 위치 이동은 FFT 위상에 영향을 주지만 크기는 모든 주파수에서 1이다. 입력 임펄스에 대한 출력이 h이므로 h를 임펄스응답이라고 부른다. 공간/FFT 출력도 수치적으로 일치한다.

""")

# Cell 47
md(r"""
## B-5. Gaussian based Image Sharpening
f_L=h_G*f, f_H=f−f_L, g=f+kf_H이다. σ∈{1,3}, k∈{1,2}의 네 조합을 두 영역에서 비교한다. 커널 반경은 ceil(3σ), 크기는 2ceil(3σ)+1로 각각 7×7,19×19이다. 이미지 및 FFT 스펙트럼, 고정 커널 비교와 정량표를 제시한다.

""")

# Cell 48
code(r"""
for key in ["B5_sigma","B5_k","B5_vs_fixed","B5_domains"]:
    show_b(key)
print("sigma k size MSE PSNR SSIM max_error out_of_range_percent")
for row in bm["unsharp_comparison"]:
    print("%.0f %.0f %d %.3e %.2f %.8f %.3e %.3f" % tuple(row))

""")

# Cell 49
md(r"""
**해석**
σ가 커지면 공간 커널은 넓어지고 저역통과 통과 대역은 좁아진다. 따라서 잔차 1−H_G가 더 낮은 주파수까지 강조하여 f_H에 굵은 윤곽이 포함된다. 큰 k는 잔차의 강도를 키워 halo·잡음 증폭·범위 밖 출력을 증가시킨다. σ=3에서 k=1→2일 때 범위 밖 화소 비율은 2.673%→7.665%이다.
고정 3×3 커널은 주파수별 이득이 고정되어 있고 Gaussian 방법은 공간 규모와 강도를 독립 조절한다. 고정 커널이 굵은 구조에 전혀 반응하지 않는다는 뜻은 아니다. 네 조합의 공간/FFT 결과는 모두 반올림 수준의 차이만 보인다.

""")

# Cell 50
md(r"""
## B-6. Discussion
합성곱 정리에 따라 공간 합성곱은 각 Fourier 성분에 주파수 응답을 곱하는 것과 같다. 충분한 zero-padding으로 선형 합성곱의 전체 지지영역을 담으면 순환 겹침이 방지된다. 두 경로의 경계조건과 커널 정렬을 맞추어야 하며, 남는 차이는 FFT와 누적합의 유한정밀 반올림 때문이다.
N×N 영상과 k×k 커널에서 직접합성곱은 O(N²k²), FFT 경로는 패딩 크기를 포함해 대략 O(PQ log(PQ))이다.

""")

# Cell 51
md(r"""
# Part C. Wiener Filter
**비너 필터란?** 흐려지고 잡음이 섞인 영상에서 원본을 추정하는 복원 필터이다. 흐림으로 약해진 주파수를 단순히 역증폭하면 잡음도 커지므로, K를 사용해 역복원과 잡음 억제 사이의 균형을 조절한다.
열화는 g=h*f+n이다. PSF는 σ=2.5의 15×15 Gaussian이며 합을 1로 정규화한다. 입력은 [0,1]이다. 열화·복원은 동일한 주기적 경계의 순환 합성곱 모델을 사용한다. 원본 f는 평가에만 사용하며 복원에는 g,H,K만 필요하다. 잡음 시드는 0이고 가산 후 g는 clipping하지 않는다.

""")

# Cell 52
code(r"""
WIENER_INPUT = "wiener_filter_input.png"   # 대체: wiener_filter_input_2.png / wiener_filter_input_rocket.png
f1 = np.array(Image.open(os.path.join(IMG_DIR, WIENER_INPUT)).convert('L'), dtype=np.float64) / 255.0
print("f1 range:", f1.min(), f1.max(), "shape", f1.shape)

def gaussian_psf(size=15, sigma=2.5):
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return h / h.sum()                     # sum = 1 정규화

def psf2otf(psf, shape):
    # PSF를 영상 크기로 zero-pad 후 중심을 원점으로 이동(zero-phase) -> OTF=H(u,v)
    p = np.zeros(shape)
    kh, kw = psf.shape
    p[:kh, :kw] = psf
    p = np.roll(p, -(kh // 2), axis=0)
    p = np.roll(p, -(kw // 2), axis=1)
    return np.fft.fft2(p)

psf = gaussian_psf(15, 2.5)
H = psf2otf(psf, f1.shape)
print("PSF sum = %.6f, shape %s" % (psf.sum(), psf.shape))
""")

# Cell 53
md(r"""
## C-1. Generate a Blurred Image
g_b=h*f를 영상 크기 FFT 곱으로 계산한다. PSF 중심을 배열 원점으로 옮겨 H를 얻으며 순환 합성곱 경계를 사용한다. B의 zero 경계 선형 합성곱과 구분한다. Gaussian blur가 에지와 질감을 약화시키는 것을 확인한다.

""")

# Cell 54
code(r"""
F1 = np.fft.fft2(f1)
gb = np.real(np.fft.ifft2(H * F1))     # blurred (circular convolution)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
show(ax[0], f1, "Original f(x,y)", vmin=0, vmax=1)
show(ax[1], psf, "PSF h(x,y) 15x15 sigma=2.5", cmap='viridis')
show(ax[2], gb, "Blurred g_b = h*f", vmin=0, vmax=1)
plt.tight_layout(); savefig("C1_blur"); plt.show()
""")

# Cell 55
md(r"""
## C-2. Add Zero-mean Gaussian Noise
$g=g_b+n$, 목표 $\mathrm{RMS}_{noise}=0.03$. 생성한 잡음의 실제 RMS를 정확히 0.03으로 맞춘다(재스케일).
**그림 관찰**: 블러+잡음(오른쪽)은 매끄러운 하늘·잔디 영역에 오돌토돌한 그레인이 뚜렷이 얹혀 왼쪽 블러 영상과 구별된다.
""")

# Cell 56
code(r"""
RMS_TARGET = 0.03
n = rng.normal(0.0, 1.0, f1.shape)
n = n - n.mean()                                  # zero-mean 보정
n = n * (RMS_TARGET / np.sqrt(np.mean(n**2)))     # 실제 RMS를 정확히 맞춤
print("noise actual RMS = %.5f (target %.3f)" % (np.sqrt(np.mean(n**2)), RMS_TARGET))
g = gb + n

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
show(ax[0], gb, "Blurred g_b", vmin=0, vmax=1)
show(ax[1], g, "Blurred + noisy g", vmin=0, vmax=1)
plt.tight_layout(); savefig("C2_noise"); plt.show()
""")

# Cell 57
md(r"""
## C-3. Restore with a Wiener Filter
복원은 F_hat=conj(H)G/(|H|²+K)를 역 FFT하여 구한다. conj(H)는 복소켤레이다. K는 주파수별 잡음/신호 전력비를 상수로 근사한 항이다. K=0이면 H≠0에서 1/H에 해당하지만 H=0에서는 역복원이 정의되지 않는다.

""")

# Cell 58
code(r"""
def wiener_restore(g, H, K):
    G = np.fft.fft2(g)
    Wf = np.conj(H) / (np.abs(H)**2 + K)
    return np.real(np.fft.ifft2(Wf * G))

Gspec = np.fft.fft2(g)
restored_demo = wiener_restore(g, H, 1e-2)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
show(ax[0], f1, "Original", vmin=0, vmax=1)
show(ax[1], g, "Degraded g", vmin=0, vmax=1)
show(ax[2], np.clip(restored_demo,0,1), "Wiener restored (K=1e-2)", vmin=0, vmax=1)
plt.tight_layout(); savefig("C3_wiener_demo"); plt.show()
""")

# Cell 59
md(r"""
## C-4. Effect of $K$
$K\in\{10^{-6},10^{-4},10^{-3},10^{-2},10^{-1}\}$ 에 대해 복원 영상과 MSE/PSNR/SSIM(원본 대비)을 비교한다.
""")

# Cell 60
code(r"""
Ks = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
results = []
fig, ax = plt.subplots(1, len(Ks), figsize=(18, 4))
for i, K in enumerate(Ks):
    r = wiener_restore(g, H, K)
    rc = np.clip(r, 0, 1)
    m = mse(f1, rc); ps = psnr(f1, rc, 1.0); ss = ssim(f1, rc, 1.0)
    results.append((K, m, ps, ss))
    show(ax[i], rc, "K=%.0e\nPSNR=%.2f" % (K, ps), vmin=0, vmax=1)
plt.tight_layout(); savefig("C4_K_sweep"); plt.show()

# ----- 결과 표: [0,1] 클리핑 후 원본과 비교 -----
print("%-10s | %-10s | %-10s | %-8s" % ("K", "MSE", "PSNR(dB)", "SSIM"))
print("-" * 46)
best = max(results, key=lambda t: t[2])
for K, m, ps, ss in results:
    mark = "  <-- best" if (K, m, ps, ss) == best else ""
    print("%-10.0e | %-10.5f | %-10.2f | %-8.4f%s" % (K, m, ps, ss, mark))
print("\nBest K = %.0e (max PSNR)" % best[0])

""")

# Cell 61
md(r"""
**해석**
비교표는 [0,1]로 clipping한 복원값의 지표이다. 원시 복원 지표는 통합 스크립트가 results_C.json에 저장한다. K=10⁻⁶에서 원시 PSNR은 −16.37 dB, clipping 후에는 5.04 dB이다. 약 94.06%의 원시 화소가 범위를 벗어나 후처리 영향이 크다.
작은 K는 |H|²≫K인 곳에서 1/H에 접근하여 흐림과 함께 잡음도 크게 역증폭한다. 큰 K는 잡음 증폭을 줄이지만 세부 구조와 밝기도 약화시킬 수 있다(DC 이득 1/(1+K)). 기본 입력의 clipping 후 PSNR 최고는 K=10⁻²(25.04 dB), SSIM 최고는 K=10⁻¹(0.6987)이다. 최고값은 지표와 탐색한 후보에 따라 다르다.

""")

# Cell 62
md(r"""
## C-5. 다른 입력들에 대한 결과
같은 열화 + Wiener 복원을 다른 입력 영상들(`wiener_filter_input_2`, `..._rocket`)에도 반복해, 최적 $K$가 영상 내용에 따라 어떻게 달라지는지 확인한다.
""")

# Cell 63
code(r"""
extra_inputs = ["wiener_filter_input_2.png", "wiener_filter_input_rocket.png"]
for name in extra_inputs:
    fx = np.array(Image.open(os.path.join(IMG_DIR, name)).convert('L'), dtype=np.float64) / 255.0
    Hx = psf2otf(psf, fx.shape)
    gbx = np.real(np.fft.ifft2(Hx * np.fft.fft2(fx)))
    nx = rng.normal(0, 1, fx.shape); nx = nx - nx.mean()
    nx = nx * (RMS_TARGET / np.sqrt(np.mean(nx**2)))
    gx = gbx + nx
    rows = []
    fig, ax = plt.subplots(1, len(Ks)+1, figsize=(20, 3.6))
    show(ax[0], gx, "%s\ndegraded g" % name, vmin=0, vmax=1)
    for i, K in enumerate(Ks):
        rc = np.clip(wiener_restore(gx, Hx, K), 0, 1)
        rows.append((K, mse(fx, rc), psnr(fx, rc, 1.0), ssim(fx, rc, 1.0)))
        show(ax[i+1], rc, "K=%.0e\nPSNR=%.2f" % (K, rows[-1][2]), vmin=0, vmax=1)
    plt.suptitle("K sweep - " + name, y=1.02); plt.tight_layout(); savefig("C5_" + name.split('.')[0]); plt.show()
    bp = max(rows, key=lambda t: t[2]); bs = max(rows, key=lambda t: t[3])
    print("[%s] best PSNR K=%.0e (%.2f dB) , best SSIM K=%.0e (%.4f)" % (name, bp[0], bp[2], bs[0], bs[3]))
""")

# Cell 64
md(r"""
**입력별 비교**
이번 후보와 잡음 시드에서 cameraman과 input 2는 PSNR 기준 K=10⁻², SSIM 기준 K=10⁻¹이 최고이다. rocket은 둘 다 K=10⁻¹이다. 영상의 스펙트럼과 평가 지표에 따른 차이이며 모든 매끄러운 영상에 대해 같은 K가 최적이라는 일반 법칙은 아니다.

""")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().with_name('Homework1_CV_solution.ipynb'))
args = parser.parse_args()
nb = nbf.v4.new_notebook(cells=C, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3'}
})
nbf.write(nb, args.output)
print(f'Notebook written: {len(C)} cells to {args.output}')
