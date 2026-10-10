# -*- coding: utf-8 -*-
"""
Homework 1 report generator (English, full-assignment format).

The report is structured to cover the entire assignment (Part A / B / C). Part A is
filled in now; Parts B and C are placeholders to be completed later. For Part A it runs
partA_point_processing.run_all() to obtain the figures (in output/) and the quantitative
results, and assembles figures + numbers + interpretation into Homework1_Report.docx.

Run:  python build_report.py
"""
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.oxml.ns import qn

import partA_point_processing as A
import partB_filtering as B
import partC_wiener as Cm

DOCX = "Homework1_Report.docx"


# ---------- docx helpers ----------
def set_base_font(doc, font="Calibri", size=10.5):
    st = doc.styles["Normal"]
    st.font.name = font
    st.font.size = Pt(size)

def h(doc, text, level=1):
    return doc.add_heading(text, level=level)

def para(doc, text, bold=False, italic=False, size=None, color=None, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold; run.italic = italic
    if size: run.font.size = Pt(size)
    if color: run.font.color.rgb = RGBColor(*color)
    if align: p.alignment = align
    return p

def bullet(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        r = p.add_run(bold_lead); r.bold = True
    p.add_run(text)
    return p

def img(doc, path, width=6.3, caption=None):
    if path and os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            c = doc.add_paragraph()
            r = c.add_run(caption); r.italic = True; r.font.size = Pt(9)
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER

def quant(doc, lines):
    p = doc.add_paragraph()
    r = p.add_run("Quantitative results"); r.bold = True
    for ln in lines:
        b = doc.add_paragraph(ln)
        b.paragraph_format.left_indent = Inches(0.25)
        b.paragraph_format.space_after = Pt(0)

def interp_head(doc, text="Interpretation (why the result occurs)"):
    para(doc, text, bold=True)


# =====================================================================
def build():
    res = A.run_all()
    m = res["metrics"]; F = res["figures"]; TH = res.get("a9_thumbs", {})

    doc = Document()
    set_base_font(doc)

    # ---------- Title / overview ----------
    doc.add_heading("Homework 1 - Computer Vision", level=0)
    para(doc, "Point Processing | Spatial & Frequency-Domain Filtering | Wiener Restoration",
         italic=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    para(doc, "Overview", bold=True, size=12)
    para(doc, "This report covers the three problems of Homework 1: "
              "(A) Point Processing and Histogram-Based Image Enhancement, "
              "(B) Spatial- and Frequency-Domain Filtering, and "
              "(C) Wiener Filter. All required image-processing operations are implemented "
              "from scratch (no built-in function that directly performs the operation). "
              "Allowed built-ins are numpy (basic array operations), PIL (image loading), "
              "numpy.fft (FFT/IFFT, used in Parts B and C), and matplotlib (visualization). "
              "All intermediate computations use floating-point (float64).")
    para(doc, "Each experiment reports the figures, quantitative results, and a brief "
              "interpretation that explains why the observed result occurs.", italic=True, size=9)
    para(doc, "Document status: Part A is complete below. Parts B and C are outlined as "
              "placeholders and will be added.", italic=True, size=9)

    # =================================================================
    # PART A
    # =================================================================
    h(doc, "Part A. Point Processing and Histogram-Based Image Enhancement", 1)
    para(doc, "Input: a color RGB image (point_processing_input_rgb.png, %dx%d). "
              "Point processing maps each pixel independently; histogram-based methods "
              "redistribute the brightness distribution to improve contrast. All enhancement "
              "is applied to the luminance Y only, so hue/saturation are preserved." % m["shape"])
    para(doc, "Parameters", bold=True)
    for s in [
        "RGB->YUV matrix: given (BT.601-style); inverse transform uses its matrix inverse",
        "Histogram: 8-bit, levels L = 256",
        "Contrast stretching: 2nd / 98th percentiles as r_min, r_max",
        "Gamma: gamma = 0.5, 1.0, 2.0 (c = 1)",
        "AHE / CLAHE tiles: 8x8; CLAHE clip = 0.01, 0.05 (fraction of tile pixel count)",
        "Bit-plane: reconstruct from the 4 most significant planes (bit 4..7)",
    ]:
        bullet(doc, s)
    para(doc, "Code is submitted as partA_point_processing.py; every figure in this section is "
              "produced by that script into output/.", italic=True, size=9)

    # ---- A1 ----
    h(doc, "A-1. RGB -> YUV Conversion", 2)
    para(doc, "The given matrix is applied directly to separate luminance Y from chrominance "
              "U, V. Since R,G,B in [0,255] and the first row sums to 1, Y lies in [0,255] "
              "(luminance). U and V are color differences and can be negative, so they are kept "
              "as floats without clipping or 8-bit casting.")
    img(doc, F["A1"], caption="Figure A1. Original RGB and the Y, U, V components.")
    quant(doc, [
        "Y range = [%.2f, %.2f], U range = [%.2f, %.2f], V range = [%.2f, %.2f]"
        % (m["Y_range"][0], m["Y_range"][1], m["U_range"][0], m["U_range"][1], m["V_range"][0], m["V_range"][1]),
        "corr(U, B-Y) = %.3f,  corr(V, R-Y) = %.3f" % (m["corr_U_BmY"], m["corr_V_RmY"]),
    ])
    interp_head(doc)
    bullet(doc, "the weighted sum reflecting human brightness sensitivity (G has the largest "
                "weight, 0.587). It carries most of the structure and edges.", bold_lead="Y (luminance): ")
    bullet(doc, "the blue-minus-luma and red-minus-luma color differences. The measured "
                "correlations are 1.000, confirming exact proportionality to B-Y and R-Y.",
           bold_lead="U is proportional to (B-Y) and V to (R-Y): ")
    bullet(doc, "blue regions (the star field of the flag, blue parts of the suit) appear bright "
                "in U, and red regions (the red stripes, red parts of the suit) appear bright in V. "
                "imshow auto-scales min..max to black..white, so a blue pixel (large positive B-Y) "
                "becomes white in U.", bold_lead="Reading the U/V images: ")
    bullet(doc, "separating brightness (Y) from color (U,V) lets us enhance contrast on Y alone "
                "while preserving color - the core strategy of Part A.", bold_lead="Why this matters: ")

    # ---- A2 ----
    h(doc, "A-2. Histogram Computation", 2)
    para(doc, "Y is quantized to 8-bit (r_k in {0,...,255}) and counted per level directly: "
              "h(r_k) = n_k, with normalized histogram p(r_k) = n_k / MN.")
    img(doc, F["A2"], caption="Figure A2. Input Y, histogram, and normalized histogram.")
    quant(doc, ["sum(h) = %.0f ( = M*N ),  sum(p) = %.4f" % (m["hist_sum"], m["p_sum"])])
    interp_head(doc)
    bullet(doc, "in the loop, k is the intensity r_k and np.sum(q==k) is the pixel count n_k.")
    bullet(doc, "the normalized histogram has the same shape as the raw one because normalization "
                "divides every bar by the same constant MN; only the y-axis scale changes (counts "
                "vs. probability, sum = 1). A distribution concentrated in a narrow band means low contrast.")

    # ---- A3 ----
    h(doc, "A-3. Histogram Equalization (HE)", 2)
    para(doc, "The CDF, CDF(r_k) = sum_{j<=k} p(r_j), is used as the transform: "
              "s_k = (L-1)*CDF(r_k). This globally spreads the distribution toward uniform, "
              "increasing contrast. The equalized Y is recombined with the original U, V and "
              "converted back to RGB.")
    img(doc, F["A3a"], caption="Figure A3-1. Original/equalized Y and their histograms.")
    img(doc, F["A3b"], caption="Figure A3-2. Original vs. HE RGB, and the transform s=T(r) "
                               "(the curve is auxiliary, not a required item).")
    quant(doc, ["Y mean %.1f -> %.1f,  std %.1f -> %.1f"
                % (m["HE_mean_before"], m["HE_mean_after"], m["HE_std_before"], m["HE_std_after"])])
    interp_head(doc)
    bullet(doc, "the transform slope is dT/dr = (L-1)*p(r): where the histogram is dense (large p), "
                "the slope is steep and those intensities are spread apart, increasing contrast there.")
    bullet(doc, "the equalized histogram becomes comb-like for two reasons: dense regions skip output "
                "levels (empty-level gaps), while sparse regions - whose CDF rises slowly - round "
                "several inputs to the same output (many-to-one merging).")
    bullet(doc, "the mean rises (%.1f -> %.1f), so the image brightens overall. However, with a large "
                "pure-dark region present, the black background is lifted to mid-gray; mid-tone detail "
                "contrast increases while contrast within the dark region may decrease (a limitation of "
                "global HE)." % (m["HE_mean_before"], m["HE_mean_after"]))

    # ---- A4 ----
    h(doc, "A-4. Contrast Stretching", 2)
    para(doc, "A piecewise-linear transform maps [r_min, r_max] linearly onto [0, L-1]. "
              "r_min, r_max are the 2nd/98th percentiles (robust to outliers).")
    img(doc, F["A4"], caption="Figure A4. Original/stretched RGB and before/after histograms.")
    quant(doc, ["r_min(2%%) = %.1f, r_max(98%%) = %.1f, gain = (L-1)/(r_max-r_min) = %.3f"
                % (m["cs_rmin"], m["cs_rmax"], m["cs_gain"])])
    interp_head(doc)
    bullet(doc, "this image already spans nearly the full range (gain ~= %.2f), so the before/after "
                "difference is small. Stretching is most effective on low-dynamic-range (hazy) images."
                % m["cs_gain"])
    bullet(doc, "stretching vs. HE: stretching keeps the distribution shape and only widens the "
                "dynamic range linearly, whereas HE is a nonlinear CDF-based re-distribution. In short, "
                "'widen the range' vs. 'flatten the distribution'.")

    # ---- A5 ----
    h(doc, "A-5. Gamma Correction (power-law)", 2)
    para(doc, "s = 255*(r/255)^gamma. In the general form s = c*r^gamma, c = 1 means no extra scaling. "
              "gamma = 0.5, 1.0, 2.0 are compared.")
    img(doc, F["A5"], caption="Figure A5. Result images and transform curves for each gamma.")
    interp_head(doc)
    bullet(doc, "brightens and expands the dark tones. The slope gamma*r^(gamma-1) tends to infinity "
                "as r->0, so dark values are pulled apart (higher contrast), revealing shadow detail; "
                "the bright region is compressed.", bold_lead="gamma < 1 (0.5): ")
    bullet(doc, "identity transform (unchanged).", bold_lead="gamma = 1: ")
    bullet(doc, "darkens. Here the slope tends to 0 as r->0, so dark tones are compressed (shadows "
                "crushed) while bright tones expand.", bold_lead="gamma > 1 (2.0): ")

    # ---- A6 ----
    h(doc, "A-6. Adaptive Histogram Equalization (AHE)", 2)
    para(doc, "The image is divided into 8x8 tiles and HE is applied independently in each tile "
              "(no interpolation).")
    img(doc, F["A6"], caption="Figure A6. Original Y / global HE / AHE.")
    interp_head(doc)
    bullet(doc, "using a local CDF per tile improves local contrast far more than global HE.")
    bullet(doc, "noise amplification: in nearly uniform tiles (e.g., dark background, smooth parts "
                "of the suit) the narrow brightness range is force-stretched to 0..255, so faint noise "
                "shows up as blotches. Tile-boundary discontinuities also cause blocking artifacts - "
                "both addressed by CLAHE.")

    # ---- A7 ----
    h(doc, "A-7. CLAHE (Contrast Limited AHE)", 2)
    para(doc, "Each tile histogram is clipped at a clip limit, the excess is redistributed uniformly, "
              "a local CDF forms the mapping, and bilinear interpolation between tile centers removes "
              "boundary artifacts. clip = 0.01 and 0.05 are compared.")
    img(doc, F["A7"], caption="Figure A7. Original / AHE / CLAHE with histograms.")
    quant(doc, [
        "Tile size 64x64 = 4096, clip=0.01 -> clip_count = 40",
        "Darkest tile %s: %d pixels with Y=0 (%.0f%%)"
        % (m["clahe_tile"], m["clahe_tile_zero_count"], 100 * m["clahe_tile_zero_frac"]),
        "Input 0 -> output:  AHE (no clip) %d  |  CLAHE clip=0.01 %d  |  clip=0.05 %d"
        % (m["clahe_map0_ahe"], m["clahe_map0_clip01"], m["clahe_map0_clip05"]),
    ])
    interp_head(doc, "Interpretation / Discussion")
    bullet(doc, "in AHE a tile's narrow brightness range makes the CDF steep (runaway contrast gain), "
                "amplifying noise. Clipping caps those peaks, bounding the slope.",
           bold_lead="Why CLAHE amplifies less noise: ")
    bullet(doc, "clip is applied to the tile INPUT histogram (exactly as the assignment states). That "
                "histogram is the material for building the mapping (LUT); it is not a cap on the bars "
                "of the OUTPUT image histogram.", bold_lead="What clip acts on: ")
    bullet(doc, "the %d pixels with value 0 all map to a single output value (point mapping), so they "
                "pile into one output bin regardless of clipping. Clipping only moves where they land "
                "(AHE 0->%d vs. CLAHE 0->%d); it cannot reduce the count. The near-0 spike therefore "
                "reflects a genuinely large dark region, not noise."
                % (m["clahe_tile_zero_count"], m["clahe_map0_ahe"], m["clahe_map0_clip01"]),
           bold_lead="Why the near-0 spike persists: ")
    bullet(doc, "too small ~= original (peaks fully clipped -> near-uniform histogram -> linear CDF -> "
                "near-identity mapping, little enhancement); too large ~= AHE (peaks pass -> steep CDF -> "
                "over-contrast and background-noise amplification).", bold_lead="Clip limit too small / too large: ")

    # ---- A8 ----
    h(doc, "A-8. Bit-Plane Slicing", 2)
    para(doc, "Y(x,y) = sum_k b_k * 2^k is decomposed into 8 bit planes. The image is reconstructed "
              "from the 4 most significant planes (bit 4..7) and compared with the original.")
    img(doc, F["A8a"], caption="Figure A8-1. Eight bit planes (top-left MSB -> bottom-right LSB).")
    img(doc, F["A8b"], width=5.2, caption="Figure A8-2. Original vs. 4-MSB reconstruction.")
    quant(doc, ["4-MSB reconstruction:  MSE = %.2f,  PSNR = %.2f dB" % (m["bitplane_MSE"], m["bitplane_PSNR"])])
    interp_head(doc)
    bullet(doc, "high-order planes (bit 7/6/5) carry large brightness changes - most of the visual "
                "structure/outline. Low-order planes (bit 1/0) have tiny weight and resemble fine "
                "texture/noise.")
    bullet(doc, "two reasons the 4 MSB are the most important: (1) their weights 16+32+64+128 = 240 are "
                "about 94%% of the max value 255; (2) the reconstruction PSNR ~= %.1f dB and is visually "
                "almost identical. Dropping the low bits causes faint false contours in smooth gradients."
                % m["bitplane_PSNR"])

    # ---- A9 (comparison: landscape page, table with embedded images) ----
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    h(doc, "A-9. Comparison and Discussion", 2)
    para(doc, "The result image and histogram of each method are placed inside the table for a direct "
              "comparison of contrast improvement and noise.")

    rows = [
        ("Contrast stretching", "stretch", "Global", "Low (linear dynamic-range widening; shape kept)",
         "Uniform gain across range (linear), gain=(L-1)/(r_max-r_min) - usually small", "O(MN)",
         "Low-dynamic-range (hazy) images, preprocessing"),
        ("Histogram equalization", "he", "Global", "High (distribution flattening)",
         "Selective amplification in dense bands (nonlinear) - stronger than uniform stretching",
         "O(MN+L)", "Images with globally low contrast"),
        ("AHE", "ahe", "Local", "Very high (local)",
         "High - narrow tile ranges stretched -> noise amplification + blocking",
         "Depends on tile count/size", "Images where local contrast matters"),
        ("CLAHE", "clahe", "Local", "High (controlled by clip)",
         "Low - clip caps the slope + interpolation removes blocking",
         "AHE + clipping/interpolation -> depends on implementation/params",
         "Medical, low-light; practical standard"),
        ("Gamma correction", "gamma", "Global (point)", "Tone-curve reshaping (brightness remap)",
         "Low - monotonic point map adds no new noise (can emphasize existing noise for large gamma)",
         "O(MN)", "Display gamma / exposure correction"),
    ]
    headers = ["Method", "Result image", "Scope", "Contrast gain", "Noise sensitivity",
               "Histogram", "Complexity", "Typical use"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for j, htext in enumerate(headers):
        c = table.rows[0].cells[j]
        c.text = ""
        r = c.paragraphs[0].add_run(htext); r.bold = True; r.font.size = Pt(9)
    for (name, key, scope, contrast, noise, cost, app) in rows:
        cells = table.add_row().cells
        cells[0].text = ""; cells[0].paragraphs[0].add_run(name).bold = True
        ip, hp = TH.get(key, (None, None))
        if ip and os.path.exists(ip):
            cells[1].paragraphs[0].add_run().add_picture(ip, width=Inches(1.05))
        if hp and os.path.exists(hp):
            cells[5].paragraphs[0].add_run().add_picture(hp, width=Inches(1.35))
        for idx, txt in [(2, scope), (3, contrast), (4, noise), (6, cost), (7, app)]:
            cells[idx].text = ""
            rr = cells[idx].paragraphs[0].add_run(txt); rr.font.size = Pt(8.5)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)

    doc.add_paragraph()
    para(doc, "Summary", bold=True)
    para(doc, "Global methods (stretching, HE, gamma) are fast and simple but limited in local "
              "contrast; local methods (AHE, CLAHE) are strong on local contrast but cost more and are "
              "noisier. CLAHE balances the two extremes via the clip limit and interpolation, and is the "
              "practical standard.")

    # =================================================================
    # PART B / C placeholders (format covers the whole assignment)
    # =================================================================
    secp = doc.add_section(WD_SECTION.NEW_PAGE)
    secp.orientation = WD_ORIENT.PORTRAIT
    secp.page_width, secp.page_height = secp.page_height, secp.page_width

    resB = B.run_all()
    mB = resB["metrics"]; FB = resB["figures"]

    h(doc, "Part B. Spatial- and Frequency-Domain Filtering", 1)
    para(doc, "Input: a grayscale image f(x,y) (spatial_frequency_filtering_input.png, %dx%d). "
              "Each filter is applied in both the spatial domain (direct convolution) and the "
              "frequency domain (FFT multiply), and the two are compared. Fourier spectra are shown "
              "as the centered log-magnitude log(1 + |fftshift(F)|)." % (mB["shape"][0], mB["shape"][1]))
    para(doc, "Parameters", bold=True)
    for s in [
        "Blur kernel  h_b = (1/9) * ones(3,3)",
        "Sharpen kernel  h_s = [[0,-1,0],[-1,5,-1],[0,-1,0]]",
        "Frequency filtering: zero-pad to (M+m-1, N+n-1) so the FFT performs LINEAR (not circular) convolution; same zero boundary as the spatial method",
        "Unsharp masking: sigma = 1.0, 3.0 ; k = 1.0, 2.0 ; Gaussian radius = ceil(3*sigma)",
    ]:
        bullet(doc, s)
    para(doc, "Code is submitted as partB_filtering.py; every figure below is produced by that script.",
         italic=True, size=9)

    # ---- B1 ----
    h(doc, "B-1. Blur Filtering", 2)
    para(doc, "Top row shows the four required displays: (1) input f, (2) log|F|, (3) impulse response "
              "h_b, (4) log|H_b|. Bottom row shows the spatial result g_s = h_b*f, the frequency result "
              "g_f = F^-1{H_b F}, the output spectrum log(1+|G_f|), and the pixelwise difference |g_s - g_f|.")
    img(doc, FB["B1"], caption="Figure B1. Blur: inputs/responses (top) and results + difference (bottom).")
    quant(doc, ["spatial vs frequency:  MSE = %.2e,  PSNR = %.1f dB,  SSIM = %.6f"
                % (mB["blur_mse"], mB["blur_psnr"], mB["blur_ssim"])])
    interp_head(doc)
    bullet(doc, "the zero-padding that makes the FFT compute LINEAR convolution lives inside conv2d_fft "
                "(pad to (M+m-1, N+n-1), multiply, crop 'same'). spectrum_of_kernel() only pads the kernel "
                "to image size to DISPLAY H(u,v); it is not used for the filtering itself.",
           bold_lead="Where the zero-padding happens: ")
    bullet(doc, "the assignment's numbered 'display' list is items 1-4 (inputs and responses). The spatial "
                "result g_s and frequency result g_f are the quantities to be computed and compared "
                "(MSE/PSNR/SSIM), shown here in the bottom row.", bold_lead="Where g_s appears: ")
    bullet(doc, "by the convolution theorem the two methods compute the SAME linear convolution, so the "
                "results are identical up to floating-point round-off (MSE ~1e-26, PSNR >300 dB, SSIM=1). "
                "The |g_s - g_f| panel confirms it (max ~1e-13).", bold_lead="Why spatial and frequency match: ")

    # ---- B2 ----
    h(doc, "B-2. Sharpening Filtering", 2)
    para(doc, "Same layout for h_s = original + Laplacian-based edge boost (coefficients sum to 1, so mean "
              "brightness is preserved).")
    img(doc, FB["B2"], caption="Figure B2. Sharpen: inputs/responses (top) and results + difference (bottom).")
    quant(doc, ["spatial vs frequency:  MSE = %.2e,  PSNR = %.1f dB,  SSIM = %.6f"
                % (mB["sharp_mse"], mB["sharp_psnr"], mB["sharp_ssim"])])
    interp_head(doc)
    bullet(doc, "again identical (difference ~round-off). The sharpened output overshoots/undershoots at "
                "edges and can leave [0,255], so it is clipped for display only; the metrics use the raw float result.")

    # ---- B3 ----
    h(doc, "B-3. Frequency-Domain Analysis", 2)
    para(doc, "For each filter we compare |F|, |H|, and |G| = |H F|. In the centered spectra the DC/low "
              "frequencies are at the middle and high frequencies toward the edges/corners.")
    img(doc, FB["B3_blur"], width=6.5, caption="Figure B3-1. Blur: |F|, |H_b|, |G|.")
    img(doc, FB["B3_sharpen"], width=6.5, caption="Figure B3-2. Sharpen: |F|, |H_s|, |G|.")
    img(doc, FB["B3_profile"], width=6.6,
        caption="Figure B3-3. Central-row profile of |H| (linear scale): blur decays (low-pass), sharpen grows (high-pass).")
    quant(doc, [
        "|H_b|: DC = %.3f, high-freq edge = %.3f  (multiplier < 1 -> attenuates high freq)" % (mB["Hb_dc"], mB["Hb_edge"]),
        "|H_s|: DC = %.3f, high-freq edge = %.3f  (multiplier > 1 -> amplifies high freq)" % (mB["Hs_dc"], mB["Hs_edge"]),
    ])
    interp_head(doc)
    bullet(doc, "in Figure B3-1 the blur |H_b| is brightest at the center and darkens toward the edges; the "
                "profile (B3-3) makes it exact: |H_b| = 1 at DC and falls to ~0.33 (with nulls). Every high-"
                "frequency component of F is multiplied by a number < 1, so |G| loses its outer (high-freq) "
                "energy and the image blurs. That is literally what a low-pass filter does.",
           bold_lead="How we know blur attenuates high freq: ")
    bullet(doc, "the sharpen |H_s| = 1 at DC and rises to ~5 at the edges, so high frequencies are boosted; "
                "|G| gains outer energy and edges are enhanced.", bold_lead="Why sharpen emphasizes high freq: ")
    bullet(doc, "an edge is an abrupt intensity change, which contains strong high-frequency content. So "
                "boosting high freq sharpens edges; attenuating it smears them.", bold_lead="Edges and high freq: ")

    # ---- B4 ----
    h(doc, "B-4. Verification of the Impulse Response", 2)
    para(doc, "Filtering a discrete impulse delta reproduces the impulse response: h*delta = h.")
    img(doc, FB["B4_blur"], width=6.8, caption="Figure B4-1. Blur: delta, flat FFT of delta, output (center 9x9), h_b, |H_b|.")
    img(doc, FB["B4_sharpen"], width=6.8, caption="Figure B4-2. Sharpen: delta, flat FFT of delta, output (center 9x9), h_s, |H_s|.")
    quant(doc, ["max |output_center - h|:  blur = %.1e,  sharpen = %.1e" % (mB["impulse_err_blur"], mB["impulse_err_sharpen"])])
    interp_head(doc)
    bullet(doc, "it equals h. For blur the center shows a small uniform 3x3 block; for sharpen a bright "
                "center (+5) with a dark cross of -1 neighbors.", bold_lead="What the output h*delta is: ")
    bullet(doc, "a shifted impulse has magnitude 1 at EVERY frequency, so log|FFT(delta)| is flat (uniform) "
                "- IDENTICAL for both filters. (If auto-scaled, ~1e-16 round-off would make it look textured; "
                "we fix the color range so it reads as truly flat.)", bold_lead="Why the delta spectrum is flat: ")
    bullet(doc, "the flat delta excites all frequencies equally, so the output reveals the system's full "
                "frequency response H(u,v); its spatial form is h. The log|H| panels DIFFER between filters "
                "(blur = bright center/low-pass; sharpen = dark center, bright corners/high-pass) precisely "
                "because h differs - that is why h is called the impulse response.",
           bold_lead="Why the delta FFT is the same but |H| differs: ")

    # ---- B5 ----
    h(doc, "B-5. Gaussian-based Image Sharpening (Unsharp Masking)", 2)
    para(doc, "Blur with a Gaussian low-pass to get f_L, take the high-frequency part f_H = f - f_L, and "
              "add it back: g = (1+k)f - k f_L.")
    img(doc, FB["B5_1"], width=6.2, caption="Figure B5-1. Gaussian kernel, blurred f_L, and high-freq f_H for each sigma (these depend on sigma only; k is not used here).")
    img(doc, FB["B5_2"], width=6.6, caption="Figure B5-2. Sharpened results with original for reference (sigma -> columns, k -> rows).")
    img(doc, FB["B5_3"], width=6.8, caption="Figure B5-3. Spectra of original, blurred, high-freq, sharpened for the representative setting sigma=3.0, k=2.0.")
    img(doc, FB["B5_4"], width=6.2, caption="Figure B5-4. Unsharp masking vs. the fixed sharpening kernel h_s.")
    interp_head(doc)
    bullet(doc, "f_L is the low-pass (low-freq) part, so f_H = f - f_L cancels the low freqs and keeps the "
                "high freqs (edges/detail) - a Gaussian-based high-pass.", bold_lead="Why subtracting the blur extracts high freq: ")
    bullet(doc, "larger sigma treats more of the spectrum as 'low' and removes it, so f_H contains a broader "
                "band (thicker edges); smaller sigma extracts only the finest detail (see B5-1).",
           bold_lead="Effect of sigma: ")
    bullet(doc, "k scales how much high-freq is added back: larger k is sharper, but too large produces edge "
                "overshoot (halos/ringing), noise amplification and clipping (see the k=2 row in B5-2).",
           bold_lead="Effect of k: ")
    bullet(doc, "the fixed kernel h_s (B-2) is essentially one special case of unsharp masking with a fixed, "
                "tiny blur and a fixed strength. Unsharp masking generalizes it: sigma sets WHICH band is "
                "treated as detail and k sets the strength, and the two are tuned independently (B5-4). The "
                "fixed h_s gives a single, strong, fine-detail sharpening; unsharp masking with large sigma "
                "sharpens coarser structure, which h_s cannot do.", bold_lead="Comparison with the fixed kernel h_s: ")

    # ---- B6 ----
    h(doc, "B-6. Discussion - Convolution Theorem", 2)
    para(doc, "g(x,y) = h(x,y) * f(x,y)   <->   G(u,v) = H(u,v) F(u,v).")
    bullet(doc, "a complex exponential is an eigenfunction of any linear shift-invariant (LTI) system, so "
                "convolving by h simply multiplies each frequency component of F by H(u,v). The spatial "
                "'slide-and-sum' therefore becomes a per-frequency multiply - and for large kernels the FFT "
                "route (O(N^2 log N)) is cheaper than direct convolution (O(N^2 k^2)).",
           bold_lead="Why convolution = multiplication: ")
    para(doc, "Why the two can differ slightly in practice:", bold=True)
    bullet(doc, "the plain FFT product computes CIRCULAR convolution, which wraps the image around its "
                "edges (the right edge bleeds into the left). Padding to (M+m-1, N+n-1) adds enough zeros "
                "that the wrap-around lands only in the padded region, making the result LINEAR; we then crop "
                "back to 'same'.", bold_lead="Zero-padding / circular convolution: ")
    bullet(doc, "the spatial method assumes a boundary (we used zeros). If the two methods assumed different "
                "boundaries (zero vs replicate vs wrap), the border pixels would differ. We used zeros on both, "
                "so the borders match.", bold_lead="Boundary handling: ")
    bullet(doc, "FFT/IFFT run in finite-precision floating point, so round-off accumulates to ~1e-12..1e-13 "
                "even when the math is identical. That is why the measured MSE is ~1e-26 rather than exactly 0.",
           bold_lead="Numerical precision: ")

    # =================================================================
    # PART C. Wiener Filter
    # =================================================================
    resC = Cm.run_all()
    mC = resC["metrics"]; FC = resC["figures"]

    h(doc, "Part C. Wiener Filter", 1)
    para(doc, "Degradation model: g = h*f + n, where h is the point-spread function (PSF) and n is "
              "additive noise. The PSF is a 15x15 Gaussian (sigma = 2.5) normalized so sum(h) = 1; the "
              "input f is normalized to [0,1]. The Wiener filter is designed from scratch and its "
              "regularization constant K is swept.")
    para(doc, "Parameters", bold=True)
    for s in [
        "PSF: 15x15 Gaussian, sigma = 2.5, normalized to sum = 1",
        "Noise: zero-mean Gaussian, RMS = 0.03 (rescaled to the exact target RMS)",
        "Wiener filter: F_hat = conj(H) / (|H|^2 + K) * G ; K in {1e-6, 1e-4, 1e-3, 1e-2, 1e-1}",
        "Metrics vs the original (data range [0,1]): MSE, PSNR, SSIM",
    ]:
        bullet(doc, s)
    para(doc, "Code is submitted as partC_wiener.py; every figure below is produced by that script.",
         italic=True, size=9)

    # C-1
    h(doc, "C-1. Generate a Blurred Image", 2)
    para(doc, "g_b = h*f, implemented as a frequency-domain multiply (H = FFT of the zero-phase PSF).")
    img(doc, FC["C1"], caption="Figure C1. Original, PSF, and blurred image.")
    quant(doc, ["PSF sum = %.6f (normalized)" % mC["psf_sum"]])

    # C-2
    h(doc, "C-2. Add Zero-mean Gaussian Noise", 2)
    para(doc, "g = g_b + n. The generated noise is rescaled so its actual RMS equals the target.")
    img(doc, FC["C2"], caption="Figure C2. Blurred vs. blurred + noisy.")
    quant(doc, ["actual noise RMS = %.5f (target 0.03000)" % mC["noise_rms"]])

    # C-3
    h(doc, "C-3. Restore with a Wiener Filter", 2)
    para(doc, "F_hat(u,v) = [ H*(u,v) / (|H(u,v)|^2 + K) ] G(u,v), then restored = IFFT(F_hat). "
              "K approximates the noise-to-signal power ratio; K = 0 is the plain inverse filter.")
    img(doc, FC["C3"], caption="Figure C3. Original, degraded, and Wiener-restored (K = 1e-2).")

    # C-4
    h(doc, "C-4. Effect of K", 2)
    img(doc, FC["C4"], width=6.8, caption="Figure C4. Restored images for K = 1e-6 ... 1e-1.")
    # metrics table
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    for j, htext in enumerate(["K", "MSE", "PSNR (dB)", "SSIM"]):
        c = tbl.rows[0].cells[j]; c.text = ""
        rr = c.paragraphs[0].add_run(htext); rr.bold = True; rr.font.size = Pt(9)
    bestP = mC["best_psnr_K"]; bestS = mC["best_ssim_K"]
    for K, m_, ps, ss in mC["table"]:
        cells = tbl.add_row().cells
        note = ""
        if K == bestP: note += "  (best PSNR)"
        if K == bestS: note += "  (best SSIM)"
        vals = ["%.0e%s" % (K, note), "%.5f" % m_, "%.2f" % ps, "%.4f" % ss]
        for j, v in enumerate(vals):
            cells[j].text = ""
            rr = cells[j].paragraphs[0].add_run(v); rr.font.size = Pt(9)
    doc.add_paragraph()
    interp_head(doc, "Interpretation / Discussion")
    bullet(doc, "where |H|^2 is near zero (high frequencies of a Gaussian blur), 1/|H|^2 explodes, so the "
                "filter approaches the inverse filter and amplifies noise enormously (K = 1e-6 gives PSNR "
                "~5 dB - the result looks sharp but is buried in noise).", bold_lead="K too small: ")
    bullet(doc, "the denominator is dominated by K, so the filter approaches H*/K; deblurring is weak and "
                "the restoration is over-smoothed (less noise but lost resolution).", bold_lead="K too large: ")
    bullet(doc, "K trades off noise suppression against deblurring sharpness; the optimum is near "
                "K ~= S_n/S_f (noise-to-signal power ratio).", bold_lead="Role of K: ")
    bullet(doc, "PSNR is highest at K = %.0e (~%.1f dB), the best pixel-error balance. SSIM is highest at a "
                "larger K = %.0e, because SSIM penalizes residual noise (structural distortion) more and "
                "prefers a slightly smoother restoration. So the 'best' K depends on whether pixel error or "
                "perceptual structure matters more."
                % (bestP, max(r[2] for r in mC["table"]), bestS), bold_lead="Best K: ")

    # ---- overall summary ----
    doc.add_paragraph()
    para(doc, "Overall Summary", bold=True, size=12)
    para(doc, "Part A implemented point-processing and histogram-based enhancement from scratch and "
              "compared global (HE, stretching, gamma) vs. local (AHE, CLAHE) methods. Part B verified the "
              "convolution theorem (spatial vs. frequency filtering are identical up to round-off with "
              "linear-convolution zero-padding) and characterized blur as low-pass and sharpening as "
              "high-pass. Part C designed a Wiener filter for a Gaussian-blur-plus-noise degradation and "
              "showed quantitatively how the regularization constant K balances noise suppression against "
              "deblurring sharpness.")

    doc.save(DOCX)
    print("saved", DOCX)


if __name__ == "__main__":
    build()
