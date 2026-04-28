"""
Generate a slide-style PDF for DCE & ImOOD paper summaries.
Figures: blank placeholder boxes with citation labels (user inserts images later).
Last slide: Korean summary paragraph.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus.flowables import Flowable

W, H = A4  # 210 × 297 mm  (595 × 842 pt)

# ── Brand colours ─────────────────────────────────────────────────────────────
C_DCE   = colors.HexColor("#1A4FA0")   # deep blue  – DCE
C_IMOOD = colors.HexColor("#B33000")   # burnt red  – ImOOD
C_LITE  = colors.HexColor("#EEF3FC")
C_LITE2 = colors.HexColor("#FFF1EC")
C_GREY  = colors.HexColor("#555555")
C_DKGRY = colors.HexColor("#222222")
C_WHITE = colors.white
C_GOLD  = colors.HexColor("#E8A020")

# ── Custom Flowables ──────────────────────────────────────────────────────────
class ColorBar(Flowable):
    """Full-width coloured title bar."""
    def __init__(self, text, bg, fg=colors.white, height=22*mm, font="Helvetica-Bold", fsize=18):
        super().__init__()
        self.text   = text
        self.bg     = bg
        self.fg     = fg
        self.height = height
        self.font   = font
        self.fsize  = fsize

    def wrap(self, availW, availH):
        self.w = availW
        return (availW, self.height)

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.rect(0, 0, self.w, self.height, fill=1, stroke=0)
        c.setFillColor(self.fg)
        c.setFont(self.font, self.fsize)
        c.drawString(10*mm, self.height/2 - self.fsize/2 + 1, self.text)


class SectionLabel(Flowable):
    """Left-coloured-bar section heading."""
    def __init__(self, text, color, fsize=12):
        super().__init__()
        self.text  = text
        self.color = color
        self.fsize = fsize

    def wrap(self, availW, availH):
        self.w = availW
        return (availW, self.fsize * 1.8)

    def draw(self):
        c = self.canv
        c.setFillColor(self.color)
        c.rect(0, 0, 4, self.fsize * 1.8, fill=1, stroke=0)
        c.setFillColor(C_DKGRY)
        c.setFont("Helvetica-Bold", self.fsize)
        c.drawString(9, self.fsize * 0.35, self.text)


class FigurePlaceholder(Flowable):
    """Dashed-border box to hold a figure the user will insert."""
    def __init__(self, w, h, label, color):
        super().__init__()
        self.bw    = w
        self.bh    = h
        self.label = label
        self.color = color

    def wrap(self, availW, availH):
        return (self.bw, self.bh)

    def draw(self):
        c = self.canv
        c.setStrokeColor(self.color)
        c.setFillColor(colors.HexColor("#F8F8F8"))
        c.setDash(4, 4)
        c.rect(0, 0, self.bw, self.bh, fill=1, stroke=1)
        c.setDash()
        c.setFillColor(self.color)
        c.setFont("Helvetica-Bold", 9)
        lw = c.stringWidth(self.label, "Helvetica-Bold", 9)
        c.drawString((self.bw - lw) / 2, self.bh / 2 + 5, self.label)
        c.setFont("Helvetica", 8)
        note = "[Insert figure here]"
        nw = c.stringWidth(note, "Helvetica", 8)
        c.drawString((self.bw - nw) / 2, self.bh / 2 - 10, note)


class TwoColRow(Flowable):
    """Two equal placeholder boxes side by side."""
    def __init__(self, w, h, label_l, label_r, color):
        super().__init__()
        self.bw     = w
        self.bh     = h
        self.ll     = label_l
        self.lr     = label_r
        self.color  = color

    def wrap(self, availW, availH):
        self.aw = availW
        return (availW, self.bh)

    def draw(self):
        c   = self.canv
        gap = 6
        hw  = (self.aw - gap) / 2
        for xi, label in [(0, self.ll), (hw + gap, self.lr)]:
            c.setStrokeColor(self.color)
            c.setFillColor(colors.HexColor("#F8F8F8"))
            c.setDash(4, 4)
            c.rect(xi, 0, hw, self.bh, fill=1, stroke=1)
            c.setDash()
            c.setFillColor(self.color)
            c.setFont("Helvetica-Bold", 9)
            lw = c.stringWidth(label, "Helvetica-Bold", 9)
            c.drawString(xi + (hw - lw) / 2, self.bh / 2 + 5, label)
            c.setFont("Helvetica", 8)
            note = "[Insert figure here]"
            nw = c.stringWidth(note, "Helvetica", 8)
            c.drawString(xi + (hw - nw) / 2, self.bh / 2 - 10, note)


# ── Styles ────────────────────────────────────────────────────────────────────
SS = getSampleStyleSheet()

def style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=SS[parent], **kw)
    return s

BODY   = style("body",   fontSize=9,  leading=14, textColor=C_DKGRY, spaceAfter=4)
BODY_J = style("bodyj",  fontSize=9,  leading=14, textColor=C_DKGRY, spaceAfter=4, alignment=TA_JUSTIFY)
BULLET = style("bullet", fontSize=9,  leading=13, textColor=C_DKGRY, leftIndent=12,
               bulletIndent=2, spaceAfter=3)
SUBB   = style("subb",   fontSize=8.5,leading=12, textColor=C_GREY,  leftIndent=22,
               bulletIndent=12, spaceAfter=2)
CAPTN  = style("captn",  fontSize=8,  leading=11, textColor=C_GREY,  alignment=TA_CENTER, spaceAfter=6)
KRBODY = style("krbody", fontSize=9.5,leading=16, textColor=C_DKGRY, alignment=TA_JUSTIFY, spaceAfter=8)
KRTITLE= style("krtitle",fontSize=13, leading=18, textColor=C_DKGRY, spaceAfter=6, fontName="Helvetica-Bold")

def b(t): return f"<b>{t}</b>"
def it(t): return f"<i>{t}</i>"
def bullet(text, style=BULLET, marker="•"):
    return Paragraph(f"{marker}  {text}", style)
def subbullet(text):
    return Paragraph(f"–  {text}", SUBB)

def sp(n=4): return Spacer(1, n)
def hr(color, w=0.5): return HRFlowable(width="100%", thickness=w, color=color, spaceAfter=4, spaceBefore=4)

def tag_table(items, color, cols=2):
    """Render a list of strings as coloured badge cells."""
    rows = [items[i:i+cols] for i in range(0, len(items), cols)]
    data = []
    for row in rows:
        data.append(row + [""] * (cols - len(row)))
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 0), (-1, -1), colors.white),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUND", (0, 0), (-1, -1), color),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.white),
    ])
    t = Table(data, colWidths=[(W - 3*cm) / cols] * cols)
    t.setStyle(ts)
    return t


def info_table(rows_data, color_accent):
    """Two-column label/value table for metadata."""
    ts = TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",     (0, 0), (0, -1),  color_accent),
        ("TEXTCOLOR",     (1, 0), (1, -1),  C_DKGRY),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.3, colors.HexColor("#DDDDDD")),
    ])
    col1 = 2.8 * cm
    col2 = W - 3*cm - col1
    t = Table(rows_data, colWidths=[col1, col2])
    t.setStyle(ts)
    return t


# ── PAGE BUILDER ──────────────────────────────────────────────────────────────
def build():
    out  = "/home/user/Desktop/imbal_cic/reference/dce_imood_slides.pdf"
    doc  = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.2*cm,  bottomMargin=1.2*cm,
    )
    story = []
    AW = W - 3*cm   # available width

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 1  –  DCE Overview (title + motivation + overview figure)
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(ColorBar(
        "DCE: Dual-Balance Collaborative Experts",
        C_DCE, height=24*mm, fsize=17
    ))
    story.append(sp(3))
    story.append(Paragraph(
        "for Domain-Incremental Learning under Class Imbalance",
        style("sub1", fontSize=11, textColor=C_DCE, spaceAfter=1, fontName="Helvetica-Oblique")
    ))
    story.append(Paragraph(
        "Yan et al. · ICML 2025 · github.com/Lain810/DCE",
        style("ref1", fontSize=8, textColor=C_GREY, spaceAfter=6)
    ))
    story.append(hr(C_DCE))
    story.append(sp(2))

    # Motivation box
    mot_data = [[
        Paragraph(b("Problem"), style("ph", fontSize=9, textColor=C_DCE, fontName="Helvetica-Bold")),
        Paragraph(
            "Real-world deployments face two compounding challenges: "
            "(1) <b>domain shift</b> across incrementally arriving domains, and "
            "(2) <b>intra-domain class imbalance</b> where tail classes are severely under-represented. "
            "No prior continual learning method addresses both simultaneously — "
            "standard routers misclassify tail samples, exacerbating the imbalance.",
            BODY_J)
    ]]
    mot_t = Table(mot_data, colWidths=[2.2*cm, AW - 2.2*cm])
    mot_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LITE),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("LINEAFTER",     (0, 0), (0, -1),  1.5, C_DCE),
    ]))
    story.append(mot_t)
    story.append(sp(8))

    # Overview figure placeholder
    story.append(SectionLabel("Framework Overview", C_DCE))
    story.append(sp(4))
    story.append(FigurePlaceholder(AW, 68*mm, "Fig. 1 – DCE Framework (Yan et al., ICML 2025, Fig. 1)", C_DCE))
    story.append(Paragraph(
        it("Fig. 1: Three frequency-aware experts collaborate via an affinity matrix; "
           "the Dynamic Expert Selector (DES) routes each sample using Gaussian pseudo-samples "
           "drawn from class-conditional distributions."),
        CAPTN
    ))
    story.append(sp(4))

    # Tags
    story.append(tag_table(
        ["Domain-Incremental Learning", "Class Imbalance", "Mixture-of-Experts",
         "Continual Learning", "Gaussian Pseudo-Sampling", "Expert Collaboration"],
        C_DCE, cols=3
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 2  –  DCE Key Methods
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(ColorBar("DCE — Key Methods", C_DCE, height=16*mm, fsize=14))
    story.append(sp(6))

    story.append(SectionLabel("1 · Three Frequency-Aware Experts", C_DCE))
    story.append(sp(3))
    story.append(Paragraph(
        "The training set is partitioned into three subsets based on per-class sample counts. "
        "Each expert is a full classifier independently trained on its subset:",
        BODY
    ))
    exp_data = [
        [b("Head Expert"), "Classes whose count exceeds threshold τ_h  (abundant samples). Learns strong boundaries for majority classes."],
        [b("Balanced Expert"), "Classes in the middle range  [τ_t, τ_h]. Sees moderate diversity; acts as a global anchor."],
        [b("Tail Expert"), "Classes below threshold τ_t  (rare samples). Specialises on minority-class discrimination; trained with stronger augmentation/resampling."],
    ]
    exp_t = Table(exp_data, colWidths=[3*cm, AW - 3*cm])
    exp_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), C_LITE),
        ("BACKGROUND",    (1, 0), (1, -1), colors.white),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TEXTCOLOR",     (0, 0), (0, -1), C_DCE),
    ]))
    story.append(exp_t)
    story.append(sp(8))

    story.append(SectionLabel("2 · Dynamic Expert Selector (DES) via Gaussian Pseudo-Sampling", C_DCE))
    story.append(sp(3))
    story.append(Paragraph(
        "At inference time, the router must decide which expert(s) to trust for a given sample. "
        "DCE avoids the classic routing problem (the router itself is biased by imbalance) by learning "
        "the selector on <i>synthetic</i> samples rather than real ones:",
        BODY
    ))
    for txt in [
        b("Step 1 — Class-conditional Gaussians:") + "  For each class k, estimate μ<sub>k</sub> and Σ<sub>k</sub> from training embeddings.",
        b("Step 2 — Pseudo-sample generation:") + "  Draw synthetic samples x̃ ~ 𝒩(μ<sub>k</sub>, Σ<sub>k</sub>) for every class, including tail classes that have too few real samples for reliable routing.",
        b("Step 3 — Affinity matrix A:") + "  Compute class-level pairwise affinities between experts using pseudo-sample predictions. A<sub>ij</sub> measures how much expert i's predictions align with expert j's on class-specific pseudo-samples.",
        b("Step 4 — Soft expert combination:") + "  At inference, the final logit vector is <b>z = Σ<sub>i</sub> α<sub>i</sub>(x) · z<sub>i</sub></b> where α<sub>i</sub>(x) is the DES weight for expert i, computed via A and the sample's predicted class distribution.",
    ]:
        story.append(bullet(txt))
    story.append(sp(8))

    story.append(SectionLabel("3 · Key Design Principles", C_DCE))
    story.append(sp(3))
    kd_data = [
        ["No hard routing", "All experts contribute via soft weights → no sample is permanently lost to a wrong expert."],
        ["Imbalance-aware pseudo-samples", "Gaussian sampling equalises tail coverage for DES training regardless of real-data IR."],
        ["Plug-and-play", "Expert architecture is any standard classifier; DES is a lightweight attention-style combiner."],
    ]
    kd_t = Table(kd_data, colWidths=[3.2*cm, AW - 3.2*cm])
    kd_t.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",     (0, 0), (0, -1), C_DCE),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [C_LITE, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
    ]))
    story.append(kd_t)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 3  –  DCE Results
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(ColorBar("DCE — Experimental Results", C_DCE, height=16*mm, fsize=14))
    story.append(sp(6))

    story.append(SectionLabel("Datasets & Setup", C_DCE))
    story.append(sp(3))
    story.append(info_table([
        ["Benchmarks",  "Office-Home, DomainNet, CORe50, CDDB-Hard (4 domain-incremental datasets)"],
        ["Imbalance",   "CORe50 & CDDB-Hard: half of tasks use IR=50, half use IR=100; Office-Home & DomainNet: natural LT"],
        ["Baselines",   "EWC, DER++, FOSTER, iCaRL, BEEF, CoRe, GKEAL (SOTA continual learning methods)"],
        ["Backbone",    "ResNet-18 / ViT-B (dataset-dependent); standard domain-incremental protocol"],
    ], C_DCE))
    story.append(sp(8))

    story.append(SectionLabel("Main Result Table", C_DCE))
    story.append(sp(3))
    story.append(FigurePlaceholder(AW, 52*mm, "Table 1 – Main Results (Yan et al., ICML 2025, Table 1)", C_DCE))
    story.append(Paragraph(
        it("Table 1: Average accuracy across all domains/tasks. DCE consistently outperforms all baselines on 4 benchmarks."),
        CAPTN
    ))
    story.append(sp(6))

    story.append(SectionLabel("Key Findings", C_DCE))
    story.append(sp(3))
    for txt in [
        b("State-of-the-art on all 4 benchmarks.") + "  DCE surpasses the best competing method by a significant margin on both balanced and imbalanced domain-IL settings.",
        b("Tail-class improvement most pronounced.") + "  The tail expert + DES routing together recover minority-class accuracy that standard IL methods drop to near zero.",
        b("Gaussian pseudo-sampling is crucial.") + "  Ablation: replacing pseudo-samples with real samples for DES training degrades tail-class accuracy significantly (routing bias re-emerges).",
        b("Affinity-based soft combination beats hard routing.") + "  Hard routing (argmax expert selection) underperforms soft combination by ~3–5% on tail classes.",
    ]:
        story.append(bullet(txt))
    story.append(sp(6))

    story.append(SectionLabel("Ablation — Expert Design", C_DCE))
    story.append(sp(3))
    story.append(TwoColRow(AW, 42*mm,
        "Fig. 2a – Ablation: Expert partitioning\n(Yan et al., ICML 2025, Fig. 3a)",
        "Fig. 2b – DES weight visualisation\n(Yan et al., ICML 2025, Fig. 3b)",
        C_DCE))
    story.append(Paragraph(
        it("Left: removing the tail expert degrades few-shot class accuracy most. "
           "Right: DES assigns higher weight to the tail expert for rare-class inputs."),
        CAPTN
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 4  –  ImOOD Overview
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(ColorBar(
        "ImOOD: Balanced OOD Detection under Long-Tailed Training",
        C_IMOOD, height=24*mm, fsize=15
    ))
    story.append(sp(3))
    story.append(Paragraph(
        "Towards Balanced Out-of-Distribution Detection under Long-Tailed Training Data",
        style("sub2", fontSize=10.5, textColor=C_IMOOD, spaceAfter=1, fontName="Helvetica-Oblique")
    ))
    story.append(Paragraph(
        "Wang et al. · NeurIPS 2024 · github.com/alibaba/imood",
        style("ref2", fontSize=8, textColor=C_GREY, spaceAfter=6)
    ))
    story.append(hr(C_IMOOD))
    story.append(sp(2))

    # Motivation box
    mot_data2 = [[
        Paragraph(b("Problem"), style("ph2", fontSize=9, textColor=C_IMOOD, fontName="Helvetica-Bold")),
        Paragraph(
            "Standard OOD detectors are trained on balanced datasets. Under long-tailed training, "
            "head classes dominate the feature space, causing the model to "
            "<b>reject tail-class in-distribution (ID) samples as OOD</b> and "
            "<b>accept head-like OOD samples as ID</b>. "
            "Existing fixes (rebalancing, two-stage training) address classification but ignore OOD calibration.",
            BODY_J)
    ]]
    mot_t2 = Table(mot_data2, colWidths=[2.2*cm, AW - 2.2*cm])
    mot_t2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LITE2),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("LINEAFTER",     (0, 0), (0, -1),  1.5, C_IMOOD),
    ]))
    story.append(mot_t2)
    story.append(sp(8))

    story.append(SectionLabel("Motivation Illustration", C_IMOOD))
    story.append(sp(4))
    story.append(FigurePlaceholder(AW, 60*mm, "Fig. 3 – Head-biased OOD scoring (Wang et al., NeurIPS 2024, Fig. 1)", C_IMOOD))
    story.append(Paragraph(
        it("Fig. 3: OOD score distributions for head vs. tail ID classes under standard training (left) "
           "and ImOOD (right). Head-class ID samples get low OOD scores (correctly kept); "
           "tail-class ID samples get spuriously high scores under vanilla training."),
        CAPTN
    ))
    story.append(sp(4))
    story.append(tag_table(
        ["OOD Detection", "Long-Tailed Learning", "Training-time Regularization",
         "Class-aware Bias Correction", "ID/OOD Calibration", "No Test-time Overhead"],
        C_IMOOD, cols=3
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 5  –  ImOOD Key Methods
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(ColorBar("ImOOD — Key Methods", C_IMOOD, height=16*mm, fsize=14))
    story.append(sp(6))

    story.append(SectionLabel("Core Insight — Dual-Distribution View", C_IMOOD))
    story.append(sp(3))
    story.append(Paragraph(
        "ImOOD decomposes the standard OOD score into two components by considering "
        "<i>both</i> a balanced reference and an imbalanced trained model:",
        BODY
    ))
    for txt in [
        b("Balanced OOD score  s<sup>bal</sup>(x):") + "  The score a model would assign if trained on a perfectly balanced dataset. Provides a fair reference that does not penalise tail classes.",
        b("Imbalanced OOD score  s<sup>imb</sup>(x):") + "  The score of the actual long-tailed trained model. Head-biased; incorrectly penalises tail-class samples.",
        b("Class-aware bias term  Δ<sub>k</sub>:") + "  The per-class gap  Δ<sub>k</sub> = E[s<sup>bal</sup> | y=k] − E[s<sup>imb</sup> | y=k]  estimated from training data. Measures how much class k is systematically under-scored by the imbalanced model.",
    ]:
        story.append(bullet(txt))
    story.append(sp(8))

    story.append(SectionLabel("Corrected OOD Score", C_IMOOD))
    story.append(sp(3))
    score_data = [[
        Paragraph(
            "<b>S(x) = s<sup>imb</sup>(x) + Δ<sup>^</sup><sub>ŷ</sub></b>",
            style("eq", fontSize=13, textColor=C_IMOOD, alignment=TA_CENTER,
                  fontName="Helvetica-Bold")
        ),
        Paragraph(
            "where  ŷ = argmax<sub>k</sub> p(y=k | x)  is the predicted class.  "
            "The correction shifts the score of each sample by the estimated class-level bias, "
            "making tail-class ID samples score similarly to head-class ID samples.",
            BODY_J
        )
    ]]
    score_t = Table(score_data, colWidths=[4*cm, AW - 4*cm])
    score_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), C_LITE2),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("LINEAFTER",     (0, 0), (0, -1),  1.5, C_IMOOD),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_IMOOD),
    ]))
    story.append(score_t)
    story.append(sp(8))

    story.append(SectionLabel("Training-time Regularization", C_IMOOD))
    story.append(sp(3))
    story.append(Paragraph(
        "Rather than applying the correction only at test time, ImOOD introduces a "
        "<b>training regularizer</b> that <i>directly</i> minimises the class-level bias Δ<sub>k</sub>:",
        BODY
    ))
    for txt in [
        b("Bias estimation loss  ℒ<sub>bias</sub>:") + "  Penalises the difference between per-class mean OOD scores under balanced vs. imbalanced model. Optimised jointly with the standard classification loss.",
        b("No test-time overhead:") + "  Once trained, inference uses the standard OOD scorer (Energy / ODIN / etc.) — no extra compute at test time.",
        b("Compatible with any OOD scorer:") + "  ImOOD is a training-side plug-in; the OOD score function (Energy, ODIN, Mahalanobis, …) is interchangeable.",
    ]:
        story.append(bullet(txt))
    story.append(sp(8))

    story.append(SectionLabel("Training Objective", C_IMOOD))
    story.append(sp(3))
    obj_data = [[
        Paragraph(
            "<b>ℒ<sub>total</sub> = ℒ<sub>CE</sub> + λ · ℒ<sub>bias</sub></b>",
            style("eq2", fontSize=12, textColor=C_IMOOD, alignment=TA_CENTER,
                  fontName="Helvetica-Bold")
        ),
        Paragraph(
            "ℒ<sub>CE</sub>: standard cross-entropy for ID classification.<br/>"
            "ℒ<sub>bias</sub>: class-aware bias regularizer (Eq. 4 in paper).<br/>"
            "λ: balancing hyperparameter (typically 0.1–1.0).",
            BODY
        )
    ]]
    obj_t = Table(obj_data, colWidths=[4.2*cm, AW - 4.2*cm])
    obj_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), C_LITE2),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("LINEAFTER",     (0, 0), (0, -1),  1.5, C_IMOOD),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_IMOOD),
    ]))
    story.append(obj_t)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 6  –  ImOOD Results
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(ColorBar("ImOOD — Experimental Results", C_IMOOD, height=16*mm, fsize=14))
    story.append(sp(6))

    story.append(SectionLabel("Datasets & Setup", C_IMOOD))
    story.append(sp(3))
    story.append(info_table([
        ["ID datasets",   "CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT (IR = 10, 50, 100, 200)"],
        ["OOD datasets",  "SVHN, LSUN-C, LSUN-R, iSUN, Textures, Places365"],
        ["OOD scorers",   "Energy score, ODIN, Mahalanobis, KNN (ImOOD applied on top of each)"],
        ["Baselines",     "ODIN, Energy, VOS, LogitAdj, GDFR, GradNorm, ReAct"],
        ["Backbone",      "ResNet-32 (CIFAR) / ResNet-50 (ImageNet)"],
    ], C_IMOOD))
    story.append(sp(8))

    story.append(SectionLabel("Main Results", C_IMOOD))
    story.append(sp(3))
    story.append(FigurePlaceholder(AW, 52*mm, "Table 2 – AUROC / FPR95 Results (Wang et al., NeurIPS 2024, Table 1)", C_IMOOD))
    story.append(Paragraph(
        it("Table 2: ImOOD consistently improves AUROC and reduces FPR95 across all ID/OOD dataset pairs "
           "and all three OOD scorers, with gains largest for tail-class ID inputs."),
        CAPTN
    ))
    story.append(sp(6))

    story.append(SectionLabel("Key Findings", C_IMOOD))
    story.append(sp(3))
    for txt in [
        b("Tail-class ID detection improved most.") + "  FPR95 for the 20% least-frequent ID classes drops by up to <b>~12 pp</b> on CIFAR-100-LT, with negligible degradation on head classes.",
        b("Head-class OOD detection retained.") + "  Unlike rebalancing-only approaches, ImOOD does not trade head-class OOD separation for tail-class gains.",
        b("Compatible with all tested OOD scorers.") + "  Gains are consistent whether the underlying scorer is Energy, ODIN, or Mahalanobis — validating scorer-agnostic design.",
        b("ID classification accuracy maintained.") + "  Adding ℒ<sub>bias</sub> does not degrade standard classification accuracy (< 0.3% change on CIFAR benchmarks).",
        b("Robust across imbalance ratios.") + "  Improvements scale with IR: larger gains at IR=200 than IR=10, where class bias is more severe.",
    ]:
        story.append(bullet(txt))
    story.append(sp(6))

    story.append(SectionLabel("Ablation — Bias Term Components", C_IMOOD))
    story.append(sp(3))
    story.append(TwoColRow(AW, 38*mm,
        "Fig. 4a – AUROC vs. IR (Wang et al., NeurIPS 2024, Fig. 3)",
        "Fig. 4b – Per-class OOD score distribution (Wang et al., NeurIPS 2024, Fig. 4)",
        C_IMOOD))
    story.append(Paragraph(
        it("Left: AUROC improves with higher IR — bias correction most valuable when imbalance is severe. "
           "Right: after ImOOD training, tail-class OOD score distributions align with head-class distributions."),
        CAPTN
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 7  –  Comparative Summary (both papers side-by-side)
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(ColorBar("Comparative Summary: DCE vs. ImOOD",
                          colors.HexColor("#2D2D2D"), height=16*mm, fsize=14))
    story.append(sp(6))

    hw = (AW - 4) / 2
    comp_data = [
        [Paragraph(b("DCE"), style("ch1", fontSize=11, textColor=C_WHITE, alignment=TA_CENTER,
                                   fontName="Helvetica-Bold")),
         Paragraph(b("ImOOD"), style("ch2", fontSize=11, textColor=C_WHITE, alignment=TA_CENTER,
                                     fontName="Helvetica-Bold"))],
        [Paragraph(b("Task"), BODY), Paragraph(b("Task"), BODY)],
        [Paragraph("Domain-incremental classification", BODY),
         Paragraph("Out-of-distribution detection", BODY)],
        [Paragraph(b("Imbalance type"), BODY), Paragraph(b("Imbalance type"), BODY)],
        [Paragraph("Intra-domain class imbalance (continual)", BODY),
         Paragraph("Long-tailed ID training data", BODY)],
        [Paragraph(b("Core mechanism"), BODY), Paragraph(b("Core mechanism"), BODY)],
        [Paragraph("3 frequency-aware experts + DES router\n(Gaussian pseudo-sampling)", BODY),
         Paragraph("Class-aware bias correction Δ_k\n+ training-time regularizer ℒ_bias", BODY)],
        [Paragraph(b("Imbalance solution"), BODY), Paragraph(b("Imbalance solution"), BODY)],
        [Paragraph("Separate tail expert + soft routing avoids routing errors", BODY),
         Paragraph("Equalise OOD score distributions across head & tail classes", BODY)],
        [Paragraph(b("Venue"), BODY), Paragraph(b("Venue"), BODY)],
        [Paragraph("ICML 2025", BODY), Paragraph("NeurIPS 2024", BODY)],
        [Paragraph(b("Relevance to this project"), BODY), Paragraph(b("Relevance to this project"), BODY)],
        [Paragraph("Expert routing strategy directly applicable to MoE design for network IDS", BODY),
         Paragraph("Bias-correction idea applicable to per-class score calibration in tail-attack detection", BODY)],
    ]
    comp_t = Table(comp_data, colWidths=[hw, hw])
    comp_ts = TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), C_DCE),
        ("BACKGROUND",    (1, 0), (1, 0), C_IMOOD),
        ("TEXTCOLOR",     (0, 0), (1, 0), C_WHITE),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
    ])
    # shade label rows
    for r in [1, 3, 5, 7, 9, 11]:
        comp_ts.add("BACKGROUND", (0, r), (1, r), colors.HexColor("#EEEEEE"))
        comp_ts.add("FONTNAME",   (0, r), (1, r), "Helvetica-Bold")
        comp_ts.add("FONTSIZE",   (0, r), (1, r), 8)
    comp_t.setStyle(comp_ts)
    story.append(comp_t)
    story.append(sp(10))

    # Relevance note
    rel_data = [[
        Paragraph(b("Why these two matter for imbalanced network intrusion detection:"), BODY),
    ],[
        Paragraph(
            "Both papers tackle the same root challenge present in CIC-IDS2017/UNSW-NB15: "
            "<b>severe class imbalance causing systematic performance collapse on rare classes</b>. "
            "DCE's Gaussian pseudo-sampling DES directly addresses the routing-error failure mode "
            "identified in this project (routing misclassifies tail classes → worse than baseline). "
            "ImOOD's class-aware score calibration offers a lightweight post-hoc mechanism to "
            "improve tail-attack detection rates without retraining the classifier.",
            BODY_J),
    ]]
    rel_t = Table(rel_data, colWidths=[AW])
    rel_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#F5F5F0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (-1, -1), 0.8, C_GOLD),
        ("LINEABOVE",     (0, 0), (-1, 0),  2, C_GOLD),
    ]))
    story.append(rel_t)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 8  –  Korean Summary (줄글)
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(ColorBar("한국어 요약 / Korean Summary",
                          colors.HexColor("#2D2D2D"), height=16*mm, fsize=14))
    story.append(sp(8))

    story.append(Paragraph("■ DCE (Dual-Balance Collaborative Experts, ICML 2025)", KRTITLE))
    story.append(hr(C_DCE, 1.0))
    story.append(sp(3))
    story.append(Paragraph(
        "DCE는 도메인 점진적 학습(Domain-Incremental Learning) 환경에서 발생하는 두 가지 복합 문제—"
        "도메인 간 분포 이동(domain shift)과 도메인 내 클래스 불균형(intra-domain class imbalance)—을 "
        "동시에 해결하기 위해 제안된 Mixture-of-Experts 프레임워크다.",
        KRBODY
    ))
    story.append(Paragraph(
        "핵심 방법론은 세 가지 빈도 인식 전문가(헤드/균형/꼬리)의 협력 구조다. "
        "학습 데이터를 클래스별 샘플 수에 따라 세 구간으로 나누어 각 전문가를 독립 학습시킨 후, "
        "Dynamic Expert Selector(DES)가 추론 시 어느 전문가에게 얼마나 의존할지를 결정한다. "
        "DES의 핵심 아이디어는 실제 데이터 대신 클래스별 가우시안 분포(μ_k, Σ_k)에서 "
        "의사 샘플(pseudo-samples)을 생성하여 라우터를 학습시키는 것이다. "
        "이 방식은 꼬리 클래스의 실제 샘플이 너무 적어 라우터가 편향되는 문제를 원천적으로 회피한다. "
        "친화 행렬(affinity matrix)을 통해 전문가 간의 예측이 소프트(soft)하게 결합되므로, "
        "특정 샘플이 잘못된 전문가에게 완전히 배정되는 하드 라우팅의 실패 모드를 방지한다.",
        KRBODY
    ))
    story.append(Paragraph(
        "실험 결과, Office-Home·DomainNet·CORe50·CDDB-Hard의 네 벤치마크 전체에서 "
        "EWC·DER++·FOSTER·iCaRL 등 기존 연속 학습 SOTA를 초과 달성했다. "
        "특히 꼬리 클래스의 정확도 향상이 두드러지며, 가우시안 의사 샘플링 제거 시 "
        "꼬리 클래스 성능이 크게 하락한다는 절제 실험(ablation)이 핵심 설계의 유효성을 지지한다. "
        "이 프로젝트의 맥락에서 DCE의 라우팅 전략—불균형에 강건한 소프트 라우터, "
        "꼬리 전문가 분리 설계—은 네트워크 침입 탐지의 MoE 설계에 직접 적용 가능한 아이디어를 제공한다.",
        KRBODY
    ))
    story.append(sp(10))

    story.append(Paragraph("■ ImOOD (Imbalanced OOD Detection, NeurIPS 2024)", KRTITLE))
    story.append(hr(C_IMOOD, 1.0))
    story.append(sp(3))
    story.append(Paragraph(
        "ImOOD는 긴 꼬리 분포(long-tailed distribution)로 학습된 모델이 "
        "이상치(out-of-distribution) 탐지에서 보이는 편향 문제를 다룬다. "
        "표준 OOD 탐지기는 균형 데이터 기준으로 설계되어, 불균형 학습 환경에서는 "
        "헤드 클래스에 유리한 OOD 점수 분포를 만들어낸다. "
        "그 결과 꼬리 클래스의 실제 정상(ID) 샘플이 OOD로 잘못 탐지되는 반면, "
        "헤드 클래스와 유사한 OOD 샘플이 정상으로 통과되는 비대칭적 오류가 발생한다.",
        KRBODY
    ))
    story.append(Paragraph(
        "ImOOD의 핵심 메커니즘은 클래스별 편향 항(Δ_k)의 추정과 교정이다. "
        "Δ_k = E[s^bal | y=k] − E[s^imb | y=k]로 정의되는 이 항은 "
        "균형 참조 모델과 실제 불균형 모델 간의 클래스별 OOD 점수 차이를 나타낸다. "
        "추론 시에는 예측 클래스 ŷ에 해당하는 편향 항을 보정하여 "
        "S(x) = s^imb(x) + Δ_ŷ 로 최종 OOD 점수를 계산한다. "
        "더 나아가, 이 편향 자체를 줄이는 학습 정규화 손실 ℒ_bias를 CE 손실에 합산하여 "
        "훈련 단계에서 편향을 원천 감소시키며, 테스트 시 추가 계산 비용이 없다. "
        "또한 에너지 점수·ODIN·마할라노비스 거리 등 임의의 OOD 스코어러와 결합 가능한 "
        "플러그인 방식으로 설계되어 범용성이 높다.",
        KRBODY
    ))
    story.append(Paragraph(
        "실험에서는 CIFAR-10/100-LT, ImageNet-LT에 SVHN·LSUN·iSUN 등의 OOD 셋을 조합하여 평가했다. "
        "꼬리 클래스의 FPR95가 최대 약 12 pp 감소하고, ID 분류 정확도는 0.3% 이내로 유지된다. "
        "불균형 비율이 클수록 개선 폭이 커지는 점도 확인되었다. "
        "네트워크 침입 탐지 관점에서 ImOOD의 클래스별 점수 교정 아이디어는 "
        "희귀 공격 유형(tail attack classes)에 대한 탐지율을 높이는 "
        "후처리(post-hoc) 보정 방법으로 활용 가능성이 크다.",
        KRBODY
    ))

    doc.build(story)
    print(f"PDF saved: {out}")


if __name__ == "__main__":
    build()
