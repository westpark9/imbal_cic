# -*- coding: utf-8 -*-
"""
Homework 1 한국어 Word 보고서 생성기.
partA/partB/partC 의 run_all() 로 figure(output/)와 정량결과를 얻어
그림 + 정량결과 + 해석(한국어)을 담은 Homework1_Report_KO.docx 를 만든다.

실행:  python build_report_ko.py
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

DOCX = "Homework1_Report_KO.docx"


# ---------- docx 헬퍼 ----------
def set_base_font(doc, font="Malgun Gothic", size=10.5):
    st = doc.styles["Normal"]
    st.font.name = font; st.font.size = Pt(size)
    rpr = st.element.get_or_add_rPr(); rf = rpr.get_or_add_rFonts()
    rf.set(qn("w:eastAsia"), font); rf.set(qn("w:ascii"), font); rf.set(qn("w:hAnsi"), font)

def h(doc, text, level=1):
    return doc.add_heading(text, level=level)

def para(doc, text, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph(); run = p.add_run(text)
    run.bold = bold; run.italic = italic
    if size: run.font.size = Pt(size)
    if align: p.alignment = align
    return p

def bullet(doc, text, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        r = p.add_run(lead); r.bold = True
    p.add_run(text); return p

def img(doc, path, width=6.3, caption=None):
    if path and os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            c = doc.add_paragraph(); r = c.add_run(caption); r.italic = True; r.font.size = Pt(9)
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER

def quant(doc, lines):
    p = doc.add_paragraph(); r = p.add_run("정량 결과"); r.bold = True
    for ln in lines:
        b = doc.add_paragraph(ln); b.paragraph_format.left_indent = Inches(0.25); b.paragraph_format.space_after = Pt(0)

def interp(doc, text="해석 (왜 그런 결과가 나오는가)"):
    para(doc, text, bold=True)


def build():
    rA = A.run_all();  mA = rA["metrics"]; FA = rA["figures"]; TH = rA.get("a9_thumbs", {})

    doc = Document(); set_base_font(doc)

    # ---------- 표지/개요 ----------
    doc.add_heading("컴퓨터비전 과제 1", level=0)
    para(doc, "점처리 · 공간/주파수 필터링 · Wiener 복원", italic=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    para(doc, "개요", bold=True, size=12)
    para(doc, "본 보고서는 과제 1의 세 문제를 다룬다: (A) 점처리 및 히스토그램 기반 영상 향상, "
              "(B) 공간·주파수 영역 필터링, (C) Wiener 필터. 모든 핵심 영상처리 연산은 built-in 없이 "
              "직접(from scratch) 구현했다. 허용 built-in은 numpy(기본 배열연산), numpy.fft(FFT/IFFT, B·C에서 사용), "
              "PIL(이미지 로딩), matplotlib(시각화)이며, 모든 중간 계산은 부동소수점(float64)으로 수행한다.")
    para(doc, "각 실험마다 그림, 정량 결과, 그리고 '왜 그런 결과가 나오는가'에 대한 간단한 해석을 제시한다.", italic=True, size=9)
    para(doc, "코드는 partA_point_processing.py / partB_filtering.py / partC_wiener.py 로 제출하며, 본 보고서의 "
              "모든 그림은 해당 스크립트가 output/ 에 생성한 것이다.", italic=True, size=9)

    # =================================================================
    # PART A
    # =================================================================
    h(doc, "Part A. 점처리 및 히스토그램 기반 영상 향상", 1)
    para(doc, "입력: 컬러 RGB 영상(point_processing_input_rgb.png, %dx%d). 점처리는 각 화소를 독립적으로 "
              "변환하고, 히스토그램 기반 기법은 밝기 분포를 재배치해 대비를 개선한다. 모든 향상은 휘도 Y에만 "
              "적용하여 색상(hue/saturation)을 보존한다." % mA["shape"])
    para(doc, "사용 파라미터", bold=True)
    for s in ["RGB→YUV 행렬: 과제 제공(BT.601 계열), 역변환은 그 역행렬",
              "히스토그램: 8-bit, 레벨 L=256",
              "대비 스트레칭: 2nd / 98th 백분위를 r_min, r_max 로 사용",
              "감마: γ = 0.5, 1.0, 2.0 (c=1)",
              "AHE / CLAHE 타일: 8×8, CLAHE clip = 0.01, 0.05 (타일 화소수 대비 비율)",
              "비트평면: 상위 4개(bit 4~7)로 재구성"]:
        bullet(doc, s)

    # A1
    h(doc, "A-1. RGB → YUV 변환", 2)
    para(doc, "주어진 행렬을 직접 적용해 휘도 Y와 색차 U, V를 분리한다. R,G,B∈[0,255]이고 첫 행의 가중치 합이 "
              "1이라 Y∈[0,255](휘도)이며, U·V는 색차로 음수가 될 수 있어 클리핑/8bit 캐스팅 없이 float로 유지한다.")
    img(doc, FA["A1"], caption="그림 A1. 원본 RGB와 Y, U, V 성분")
    quant(doc, ["Y 범위 [%.2f, %.2f], U 범위 [%.2f, %.2f], V 범위 [%.2f, %.2f]"
                % (mA["Y_range"][0], mA["Y_range"][1], mA["U_range"][0], mA["U_range"][1], mA["V_range"][0], mA["V_range"][1]),
                "corr(U, B−Y) = %.3f,  corr(V, R−Y) = %.3f" % (mA["corr_U_BmY"], mA["corr_V_RmY"])])
    interp(doc, "해석 (의미)")
    bullet(doc, "사람 눈의 밝기 민감도를 반영한 가중합(G에 0.587로 최대 가중치). 구조·에지가 대부분 담긴다.", lead="Y (휘도): ")
    bullet(doc, "U는 파랑 색차(∝ B−Y), V는 빨강 색차(∝ R−Y). 상관계수가 1.000이라 정확히 비례함을 확인.", lead="U, V (색차): ")
    bullet(doc, "성조기의 파란 캔톤·우주복 파란 부분이 U에서, 빨간 줄무늬·빨간 부분이 V에서 밝게 보인다. "
                "imshow가 값의 최소~최대를 검정~흰색으로 자동 스케일하므로, 파란 화소는 B−Y가 큰 양수 → U에서 흰색.", lead="영상으로 확인: ")
    bullet(doc, "밝기(Y)와 색(U,V)을 분리하므로 이후 향상을 Y에만 적용하면 색을 보존한 채 대비만 조정할 수 있다(Part A 전략).", lead="의의: ")

    # A2
    h(doc, "A-2. 히스토그램 계산", 2)
    para(doc, "Y를 8-bit로 양자화(r_k∈{0,…,255})한 뒤 레벨별로 직접 집계한다: h(r_k)=n_k, 정규화 p(r_k)=n_k/MN.")
    img(doc, FA["A2"], caption="그림 A2. 입력 Y, 히스토그램, 정규화 히스토그램")
    quant(doc, ["sum(h) = %.0f ( = M·N ),  sum(p) = %.4f" % (mA["hist_sum"], mA["p_sum"])])
    interp(doc)
    bullet(doc, "코드의 k가 강도 r_k, np.sum(q==k)가 그 강도의 화소 수 n_k 이다.")
    bullet(doc, "정규화 전후 히스토그램 모양이 같은 이유: 모든 막대를 같은 상수 MN으로 나눌 뿐이라 상대 비율은 "
                "그대로이고 세로축 눈금만 바뀐다(확률분포 PDF로 재해석). 분포가 좁게 몰릴수록 대비가 낮다.")

    # A3
    h(doc, "A-3. 히스토그램 평활화 (HE)", 2)
    para(doc, "CDF(r_k)=Σp(r_j)를 변환함수로 사용: s_k=(L−1)·CDF(r_k). 전역적으로 분포를 균일에 가깝게 펴 대비를 "
              "높인다. 평활화된 Y를 원본 U,V와 결합해 RGB로 역변환한다.")
    img(doc, FA["A3a"], caption="그림 A3-1. 원본/평활 Y와 각 히스토그램")
    img(doc, FA["A3b"], caption="그림 A3-2. 원본 vs HE RGB, 변환곡선 s=T(r) (곡선은 참고용, 요구 항목 아님)")
    quant(doc, ["Y 평균 %.1f → %.1f,  표준편차 %.1f → %.1f" % (mA["HE_mean_before"], mA["HE_mean_after"], mA["HE_std_before"], mA["HE_std_after"])])
    interp(doc)
    bullet(doc, "변환함수 기울기 dT/dr=(L−1)·p(r). 히스토그램이 밀집한(p 큰) 구간에서 기울기가 가팔라져 그 구간을 넓게 벌림 → 대비 증가.")
    bullet(doc, "평활 히스토그램이 빗살(comb) 모양이 되는 이유: 밀집 구간은 레벨을 건너뛰어 빈 레벨(gap)이 생기고, "
                "희소 구간은 느린 CDF를 정수로 반올림해 여러 입력이 한 출력으로 합쳐진다(many-to-one).")
    bullet(doc, "평균이 %.1f→%.1f로 올라 전체적으로 밝아진다. 다만 순흑이 많아 검정 배경이 중간 회색으로 들려, "
                "중간톤 세부 대비는 커지되 암부 자체 대비는 줄 수 있다(전역 HE의 한계)." % (mA["HE_mean_before"], mA["HE_mean_after"]))

    # A4
    h(doc, "A-4. 대비 스트레칭 (Contrast Stretching)", 2)
    para(doc, "piecewise-linear 변환으로 [r_min, r_max]를 [0, L−1]로 선형 확장. r_min, r_max는 2/98 백분위(이상치에 강건).")
    img(doc, FA["A4"], caption="그림 A4. 원본/스트레칭 RGB와 전후 히스토그램")
    quant(doc, ["r_min(2%%)=%.1f, r_max(98%%)=%.1f, 이득=(L−1)/(r_max−r_min)=%.3f" % (mA["cs_rmin"], mA["cs_rmax"], mA["cs_gain"])])
    interp(doc)
    bullet(doc, "이 영상은 이미 거의 전체 범위를 써서(이득≈%.2f) 전후 차이가 미미하다. 스트레칭은 동적범위가 좁은(뿌연) 영상에서 효과가 크다." % mA["cs_gain"])
    bullet(doc, "스트레칭 vs HE: 스트레칭은 분포 모양을 유지한 채 범위(동적범위)만 선형으로 늘리고, HE는 CDF 기반 "
                "비선형으로 분포 자체를 재배치. 요약하면 '범위를 늘림' vs '분포를 폄'.")

    # A5
    h(doc, "A-5. 감마 보정 (Power-law)", 2)
    para(doc, "s = 255·(r/255)^γ. 일반형 s = c·r^γ 에서 c=1은 추가 스케일 없음. γ=0.5, 1.0, 2.0 비교.")
    img(doc, FA["A5"], caption="그림 A5. 감마별 결과 영상과 변환곡선")
    interp(doc)
    bullet(doc, "밝아짐 + 암부 확장. 기울기 γ·r^(γ−1)가 r→0에서 ∞ → 어두운 값들이 서로 벌어져(대비↑) 그림자 디테일이 "
                "살아난다(압축되는 쪽은 밝은 영역).", lead="γ<1 (0.5): ")
    bullet(doc, "항등 변환(불변).", lead="γ=1: ")
    bullet(doc, "어두워짐. r→0에서 기울기→0이라 암부가 압축(그림자 뭉개짐), 밝은 영역이 확장.", lead="γ>1 (2.0): ")

    # A6
    h(doc, "A-6. 적응형 히스토그램 평활화 (AHE)", 2)
    para(doc, "영상을 8×8 타일로 나눠 각 타일에서 독립적으로 HE를 수행(보간 없음).")
    img(doc, FA["A6"], caption="그림 A6. 원본 Y / 전역 HE / AHE")
    interp(doc)
    bullet(doc, "국소 CDF를 쓰므로 전역 HE보다 국소 대비가 크게 향상된다.")
    bullet(doc, "잡음 증폭 이유: 어두운 배경·우주복 매끄러운 부분처럼 거의 균일한 타일은 좁은 밝기 범위를 억지로 "
                "0~255로 펴면서 미세 잡음이 얼룩으로 드러난다. 또 타일 경계 불연속으로 블로킹 아티팩트 발생 → CLAHE로 해결.")

    # A7
    h(doc, "A-7. CLAHE (Contrast Limited AHE)", 2)
    para(doc, "타일 히스토그램을 clip limit로 자르고 초과분을 균등 재분배한 뒤 국소 CDF로 매핑, 타일 중심 간 "
              "쌍선형 보간으로 경계 아티팩트 제거. clip = 0.01, 0.05 비교.")
    img(doc, FA["A7"], caption="그림 A7. 원본/AHE/CLAHE와 히스토그램")
    quant(doc, ["타일 64×64=4096, clip=0.01 → clip_count=40",
                "가장 어두운 타일%s: Y=0 이 %d개(%.0f%%)" % (mA["clahe_tile"], mA["clahe_tile_zero_count"], 100 * mA["clahe_tile_zero_frac"]),
                "입력 0 → 출력:  AHE(클립없음) %d  |  CLAHE clip=0.01 %d  |  clip=0.05 %d"
                % (mA["clahe_map0_ahe"], mA["clahe_map0_clip01"], mA["clahe_map0_clip05"])])
    interp(doc, "해석 (논의)")
    bullet(doc, "AHE의 타일 내부는 밝기 범위가 좁아 CDF가 가팔라져(대비 이득 폭주) 잡음을 증폭한다. clip이 그 봉우리를 잘라 기울기에 상한을 둔다.", lead="잡음 완화 이유: ")
    bullet(doc, "clip은 '타일 입력 히스토그램'에 적용된다(지시사항과 동일). 이는 매핑(LUT)을 만드는 재료이며, "
                "결과 영상 히스토그램의 막대 높이를 직접 자르는 게 아니다.", lead="clip이 적용되는 대상: ")
    bullet(doc, "값이 0인 화소 %d개가 매핑 후에도 한 출력값으로 모이기 때문(잡음 아님). clip은 그 위치만 바꾼다"
                "(AHE 0→%d 를 clip 0→%d 로). 점 매핑이라 한 입력값을 여러 출력으로 쪼갤 수 없다."
                % (mA["clahe_tile_zero_count"], mA["clahe_map0_ahe"], mA["clahe_map0_clip01"]), lead="0 근처 스파이크가 남는 이유: ")
    bullet(doc, "너무 작으면 ≈ 원본(봉우리가 다 잘려 히스토그램이 평평→CDF 직선→항등 매핑, 향상 미미); "
                "너무 크면 ≈ AHE(봉우리 통과→과대비·배경 잡음 증폭).", lead="clip이 너무 작/크면: ")

    # A8
    h(doc, "A-8. 비트평면 분해 (Bit-Plane Slicing)", 2)
    para(doc, "Y(x,y)=Σ b_k·2^k 로 8개 비트평면 분해. 상위 4개(bit 4~7)만으로 재구성해 원본과 비교.")
    img(doc, FA["A8a"], caption="그림 A8-1. 8개 비트평면 (좌상단 MSB → 우하단 LSB)")
    img(doc, FA["A8b"], width=5.2, caption="그림 A8-2. 원본 vs 상위 4비트 재구성")
    quant(doc, ["4-MSB 재구성:  MSE=%.2f,  PSNR=%.2f dB" % (mA["bitplane_MSE"], mA["bitplane_PSNR"])])
    interp(doc)
    bullet(doc, "상위 비트(bit 7/6/5)는 큰 밝기 변화 = 시각적 구조/윤곽의 대부분. 하위 비트(bit 1/0)는 가중치가 작아 미세 질감·잡음에 가깝다.")
    bullet(doc, "상위 4비트가 중요한 근거: ① 가중치 합 16+32+64+128=240 으로 최대값 255의 약 94%%, "
                "② 재구성 PSNR≈%.1f dB·육안으로도 거의 동일. 하위 비트 손실로 매끄러운 그라데이션에 약한 윤곽선(false contour) 발생." % mA["bitplane_PSNR"])

    # A9 (landscape 표)
    sec = doc.add_section(WD_SECTION.NEW_PAGE); sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    h(doc, "A-9. 비교 및 논의", 2)
    para(doc, "각 방법의 결과 영상과 히스토그램을 표 안에 함께 제시하여 대비 향상·잡음을 직접 비교한다.")
    rows = [
        ("Contrast stretching", "stretch", "전역", "낮음 (동적범위 선형 확장, 분포 모양 유지)",
         "전 구간 균일 증폭(선형), 이득=(L−1)/(r_max−r_min) — 보통 작음", "O(MN)", "동적범위 좁은(뿌연) 영상, 전처리"),
        ("Histogram equalization", "he", "전역", "높음 (분포 균일화)",
         "밀집 밝기대에서 선택적 증폭(비선형) — 스트레칭보다 강함", "O(MN+L)", "전반적 대비가 낮은 영상"),
        ("AHE", "ahe", "국소", "매우 높음 (국소)",
         "높음 — 평탄 타일에서 좁은 밝기범위를 펴 잡음 증폭 + 블로킹", "타일 수·크기에 따라 다름", "국소 대비가 중요한 영상"),
        ("CLAHE", "clahe", "국소", "높음 (clip로 제어)",
         "낮음 — clip으로 기울기 상한 + 보간으로 블로킹 제거", "AHE + 클리핑·보간 → 구현/파라미터에 따라 다름", "의료·저조도 등 실무 표준"),
        ("Gamma correction", "gamma", "전역 (점)", "톤 곡선 조정(밝기 재배치)",
         "낮음 — 단조 점변환이라 새 잡음 안 만듦(γ 큰 암부에선 기존 잡음 부각 가능)", "O(MN)", "디스플레이 감마/노출 보정"),
    ]
    headers = ["방법", "결과 영상", "처리 범위", "대비 향상", "잡음 민감도", "히스토그램", "계산복잡도", "적합 응용"]
    table = doc.add_table(rows=1, cols=len(headers)); table.style = "Table Grid"
    for j, ht in enumerate(headers):
        cpar = table.rows[0].cells[j]; cpar.text = ""
        rr = cpar.paragraphs[0].add_run(ht); rr.bold = True; rr.font.size = Pt(9)
    for (name, key, scope, contrast, noise, cost, app) in rows:
        cells = table.add_row().cells
        cells[0].text = ""; cells[0].paragraphs[0].add_run(name).bold = True
        ip, hp = TH.get(key, (None, None))
        if ip and os.path.exists(ip): cells[1].paragraphs[0].add_run().add_picture(ip, width=Inches(1.05))
        if hp and os.path.exists(hp): cells[5].paragraphs[0].add_run().add_picture(hp, width=Inches(1.35))
        for idx, txt in [(2, scope), (3, contrast), (4, noise), (6, cost), (7, app)]:
            cells[idx].text = ""; rr = cells[idx].paragraphs[0].add_run(txt); rr.font.size = Pt(8.5)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs: p.paragraph_format.space_after = Pt(0)
    doc.add_paragraph()
    para(doc, "요약", bold=True)
    para(doc, "전역 방법(스트레칭·HE·감마)은 빠르고 단순하지만 국소 대비에 한계가 있고, 국소 방법(AHE·CLAHE)은 "
              "국소 대비에 강하나 잡음·비용이 커진다. CLAHE는 clip limit와 보간으로 두 극단을 절충한 실무 표준이다.")

    # =================================================================
    # PART B
    # =================================================================
    secp = doc.add_section(WD_SECTION.NEW_PAGE); secp.orientation = WD_ORIENT.PORTRAIT
    secp.page_width, secp.page_height = secp.page_height, secp.page_width
    rB = B.run_all(); mB = rB["metrics"]; FB = rB["figures"]

    h(doc, "Part B. 공간·주파수 영역 필터링", 1)
    para(doc, "입력: 그레이스케일 f(x,y) (spatial_frequency_filtering_input.png, %dx%d). 각 필터를 공간영역(직접 "
              "합성곱)과 주파수영역(FFT 곱)에서 모두 수행해 비교한다. 푸리에 스펙트럼은 중심화 로그 크기 "
              "log(1 + |fftshift(F)|)로 표시한다." % (mB["shape"][0], mB["shape"][1]))
    para(doc, "사용 파라미터", bold=True)
    for s in ["블러 커널 h_b = (1/9)·ones(3,3)",
              "샤프닝 커널 h_s = [[0,-1,0],[-1,5,-1],[0,-1,0]]",
              "주파수 필터링: (M+m-1, N+n-1)로 zero-pad → FFT가 '선형'(순환 아님) 합성곱을 수행; 공간과 동일한 zero 경계",
              "Unsharp masking: σ = 1.0, 3.0 ; k = 1.0, 2.0 ; 가우시안 반경 = ⌈3σ⌉"]:
        bullet(doc, s)

    # B1
    h(doc, "B-1. 블러 필터링", 2)
    para(doc, "윗줄은 요구된 4개 표시: (1) 입력 f, (2) log|F|, (3) 임펄스응답 h_b, (4) log|H_b|. 아랫줄은 공간결과 "
              "g_s=h_b*f, 주파수결과 g_f=F⁻¹{H_b F}, 출력 스펙트럼 log(1+|G_f|), 그리고 차이 |g_s − g_f|.")
    img(doc, FB["B1"], caption="그림 B1. 블러: 입력/응답(윗줄), 결과+차이(아랫줄)")
    quant(doc, ["공간 vs 주파수:  MSE = %.2e,  PSNR = %.1f dB,  SSIM = %.6f" % (mB["blur_mse"], mB["blur_psnr"], mB["blur_ssim"])])
    interp(doc)
    bullet(doc, "선형 합성곱이 되게 하는 zero-padding은 conv2d_fft 내부((M+m-1, N+n-1)로 패딩→곱→'same' crop)에서 "
                "일어난다. spectrum_of_kernel()은 H(u,v)를 '그리기 위한' 패딩일 뿐 필터링엔 안 쓴다.", lead="zero-padding 위치: ")
    bullet(doc, "과제의 번호 매긴 'display'는 1~4(입력·응답)이고, 공간결과 g_s·주파수결과 g_f는 계산해서 "
                "비교(MSE/PSNR/SSIM)하라는 항목 — 아랫줄에 표시했다.", lead="g_s의 위치: ")
    bullet(doc, "합성곱 정리로 두 방법이 같은 선형 합성곱을 계산하므로 결과가 부동소수점 반올림만 빼면 동일"
                "(MSE ~1e-26, PSNR >300 dB, SSIM=1). 차이맵 |g_s−g_f| 최댓값 ~1e-13이 확인.", lead="왜 공간=주파수: ")

    # B2
    h(doc, "B-2. 샤프닝 필터링", 2)
    para(doc, "h_s = 원본 + 라플라시안 기반 에지강조(계수합 1이라 평균밝기 보존). 레이아웃은 B-1과 동일.")
    img(doc, FB["B2"], caption="그림 B2. 샤프닝: 입력/응답(윗줄), 결과+차이(아랫줄)")
    quant(doc, ["공간 vs 주파수:  MSE = %.2e,  PSNR = %.1f dB,  SSIM = %.6f" % (mB["sharp_mse"], mB["sharp_psnr"], mB["sharp_ssim"])])
    interp(doc)
    bullet(doc, "샤프닝도 동일(반올림 제외). 출력은 에지에서 오버/언더슈트로 [0,255]를 벗어나 표시할 때만 클리핑(지표는 원본 float).")

    # B3
    h(doc, "B-3. 주파수 영역 분석", 2)
    para(doc, "각 필터에 대해 |F|, |H|, |G|=|HF|를 비교한다. 중심화 스펙트럼에서 중앙=저주파, 가장자리=고주파.")
    img(doc, FB["B3_blur"], width=6.5, caption="그림 B3-1. 블러: |F|, |H_b|, |G|")
    img(doc, FB["B3_sharpen"], width=6.5, caption="그림 B3-2. 샤프닝: |F|, |H_s|, |G|")
    img(doc, FB["B3_profile"], width=6.6, caption="그림 B3-3. |H| 중앙행 단면(선형): 블러는 감쇠(저역통과), 샤프닝은 증가(고역통과)")
    quant(doc, ["|H_b|: DC = %.3f, 고주파 끝 = %.3f  (1보다 작은 값으로 곱함 → 고주파 감쇠)" % (mB["Hb_dc"], mB["Hb_edge"]),
                "|H_s|: DC = %.3f, 고주파 끝 = %.3f  (1보다 큰 값으로 곱함 → 고주파 증폭)" % (mB["Hs_dc"], mB["Hs_edge"])])
    interp(doc)
    bullet(doc, "그림 B3-1에서 블러 |H_b|는 중앙이 밝고 가장자리로 어둡다. 단면(B3-3)이 정확히 보여줌 — |H_b|=1(DC)에서 "
                "≈0.33까지 감소(중간에 0인 null). 모든 고주파가 1보다 작은 값으로 곱해져 |G|의 바깥(고주파) 에너지가 "
                "깎임 → 흐려짐 = 저역통과.", lead="블러가 고주파를 감쇠함을 어떻게 아나: ")
    bullet(doc, "샤프닝 |H_s|=1(DC)에서 가장자리 ≈5로 증가 → 고주파 증폭 → |G|의 바깥 에너지 증가 → 에지 강조.", lead="샤프닝이 고주파 강조: ")
    bullet(doc, "에지는 급격한 밝기 변화 = 강한 고주파. 고주파를 키우면 에지가 또렷, 깎으면 뭉개짐.", lead="에지와 고주파: ")

    # B4
    h(doc, "B-4. 임펄스 응답 검증", 2)
    para(doc, "임펄스 δ에 필터를 적용하면 임펄스응답이 복원된다: h*δ = h.")
    img(doc, FB["B4_blur"], width=6.8, caption="그림 B4-1. 블러: δ, δ의 평탄한 FFT, 출력(중앙 9×9), h_b, |H_b|")
    img(doc, FB["B4_sharpen"], width=6.8, caption="그림 B4-2. 샤프닝: δ, δ의 평탄한 FFT, 출력(중앙 9×9), h_s, |H_s|")
    quant(doc, ["max |출력중앙 − h|:  블러 = %.1e,  샤프닝 = %.1e" % (mB["impulse_err_blur"], mB["impulse_err_sharpen"])])
    interp(doc)
    bullet(doc, "바로 h다. 블러는 중앙에 작은 균일 3×3 블록, 샤프닝은 중앙(+5) 밝고 상하좌우(−1) 어두운 십자.", lead="출력 h*δ는: ")
    bullet(doc, "위치 이동된 임펄스는 모든 주파수에서 크기 1 → log|FFT(δ)|가 균일(두 필터 모두 동일). "
                "(자동 스케일이면 ~1e-16 반올림이 무늬처럼 증폭돼 색 범위를 고정해 진짜 평탄하게 표시.)", lead="δ의 FFT가 평탄한 이유: ")
    bullet(doc, "평탄한 δ가 모든 주파수를 균일 입력하므로 출력이 시스템의 전체 주파수응답 H(u,v)를 드러낸다. "
                "log|H| 패널은 필터마다 다름(블러=중앙 밝음/저역통과, 샤프닝=중앙 어둡고 모서리 밝음/고역통과) — "
                "h가 다르므로 H도 다르다. 그래서 h를 임펄스 응답이라 부른다.", lead="δ-FFT는 같은데 |H|가 다른 이유: ")

    # B5
    h(doc, "B-5. 가우시안 기반 선명화 (Unsharp Masking)", 2)
    para(doc, "가우시안 저역통과로 f_L을 얻고 고주파 f_H = f − f_L을 더해 선명화: g = (1+k)f − k·f_L.")
    img(doc, FB["B5_1"], width=6.2, caption="그림 B5-1. σ별 가우시안 커널·f_L·f_H (σ에만 의존, k는 여기서 안 씀)")
    img(doc, FB["B5_2"], width=6.6, caption="그림 B5-2. 선명화 결과(원본 포함, σ→열, k→행)")
    img(doc, FB["B5_3"], width=6.8, caption="그림 B5-3. 원본/블러/고주파/선명화 스펙트럼 (대표값 σ=3.0, k=2.0)")
    img(doc, FB["B5_4"], width=6.2, caption="그림 B5-4. Unsharp masking vs 고정 샤프닝 커널 h_s")
    interp(doc)
    bullet(doc, "f_L은 저주파(저역통과)라 f_H = f − f_L은 저주파가 상쇄되고 고주파(에지·디테일)만 남음(가우시안 고역통과).", lead="왜 블러를 빼면 고주파: ")
    bullet(doc, "σ가 클수록 더 넓은 대역을 '저주파'로 제거 → f_H가 더 넓은 대역(굵은 에지 포함). 작을수록 미세 디테일만(B5-1).", lead="σ의 영향: ")
    bullet(doc, "k는 고주파를 더하는 강도. 클수록 선명하나 너무 크면 에지 오버슈트(halo/링잉)·잡음 증폭·포화(B5-2의 k=2 행).", lead="k의 영향: ")
    bullet(doc, "고정 커널 h_s는 아주 작은 블러·고정 강도의 unsharp 특수 경우. Unsharp masking은 σ(대역)·k(강도)를 "
                "독립 조절해 h_s가 못 하는 '굵은 구조 선명화'(σ 크게)도 가능(B5-4).", lead="고정 커널 h_s와 비교: ")

    # B6
    h(doc, "B-6. 논의 — 합성곱 정리", 2)
    para(doc, "g(x,y) = h(x,y) * f(x,y)   ⟺   G(u,v) = H(u,v) F(u,v).")
    bullet(doc, "복소지수는 LTI 시스템의 고유함수라, h로 합성곱하는 것은 각 주파수 성분에 H(u,v)를 곱하는 것과 같다. "
                "공간의 '미끄러뜨려 더하기'가 주파수에선 성분별 곱 — 큰 커널에선 FFT(O(N²logN))가 직접합성곱(O(N²k²))보다 유리.", lead="왜 합성곱=곱: ")
    para(doc, "실무에서 작은 차이가 생기는 이유", bold=True)
    bullet(doc, "FFT 곱은 본질적으로 순환(circular) 합성곱 — 영상이 가장자리에서 반대편으로 감겨(wrap-around) 오른쪽 "
                "끝이 왼쪽에 샌다. (M+m-1, N+n-1)로 zero-padding하면 그 감김이 패딩 영역에만 생겨 선형이 되고 'same'으로 자른다.", lead="순환 합성곱 / zero-padding: ")
    bullet(doc, "공간 합성곱은 영상 밖 화소를 뭘로 볼지(경계조건) 정해야 한다(여기선 0). 두 방법이 다른 경계(zero/replicate/wrap)를 "
                "쓰면 테두리 화소만 달라진다. 본 과제는 양쪽 0이라 테두리까지 일치.", lead="경계 처리: ")
    bullet(doc, "FFT/IFFT는 유한정밀 부동소수점이라 수학적으로 같아도 반올림이 ~1e-12~1e-13 누적. 그래서 MSE가 정확히 0이 아니라 ~1e-26.", lead="수치 정밀도: ")

    # =================================================================
    # PART C
    # =================================================================
    rC = Cm.run_all(); mC = rC["metrics"]; FC = rC["figures"]
    h(doc, "Part C. Wiener 필터 (영상 복원)", 1)
    para(doc, "열화 모델: g = h*f + n (h는 점확산함수 PSF, n은 가산 잡음). PSF는 15×15 가우시안(σ=2.5)이며 "
              "sum(h)=1로 정규화하고, 입력 f는 [0,1]로 정규화한다. Wiener 필터를 직접 설계하고 정규화 상수 K를 스윕한다.")
    para(doc, "사용 파라미터", bold=True)
    for s in ["PSF: 15×15 가우시안, σ=2.5, sum=1 정규화",
              "잡음: 평균 0 가우시안, RMS=0.03 (실제 RMS를 정확히 맞춤)",
              "Wiener: F̂ = conj(H)/(|H|² + K)·G ; K ∈ {1e-6, 1e-4, 1e-3, 1e-2, 1e-1}",
              "지표(원본 대비, 범위 [0,1]): MSE, PSNR, SSIM"]:
        bullet(doc, s)

    h(doc, "C-1. 블러 영상 생성", 2)
    para(doc, "g_b = h*f (주파수영역 곱으로 구현, H는 zero-phase PSF의 FFT).")
    img(doc, FC["C1"], caption="그림 C1. 원본, PSF, 블러 영상")
    quant(doc, ["PSF 합 = %.6f (정규화)" % mC["psf_sum"]])

    h(doc, "C-2. 평균 0 가우시안 잡음 추가", 2)
    para(doc, "g = g_b + n. 생성한 잡음을 재스케일해 실제 RMS를 목표값에 정확히 맞춘다.")
    img(doc, FC["C2"], caption="그림 C2. 블러 vs 블러+잡음")
    quant(doc, ["실제 잡음 RMS = %.5f (목표 0.03000)" % mC["noise_rms"]])

    h(doc, "C-3. Wiener 필터로 복원", 2)
    para(doc, "F̂(u,v) = [ H*(u,v) / (|H(u,v)|² + K) ]·G(u,v), 그리고 복원 = IFFT(F̂). K는 잡음/신호 전력비를 "
              "근사하며, K=0이면 역필터(inverse filter).")
    img(doc, FC["C3"], caption="그림 C3. 원본, 열화, Wiener 복원(K=1e-2)")

    h(doc, "C-4. K의 효과", 2)
    img(doc, FC["C4"], width=6.8, caption="그림 C4. K = 1e-6 … 1e-1 복원 영상")
    tbl = doc.add_table(rows=1, cols=4); tbl.style = "Table Grid"
    for j, ht in enumerate(["K", "MSE", "PSNR (dB)", "SSIM"]):
        cpar = tbl.rows[0].cells[j]; cpar.text = ""
        rr = cpar.paragraphs[0].add_run(ht); rr.bold = True; rr.font.size = Pt(9)
    bestP = mC["best_psnr_K"]; bestS = mC["best_ssim_K"]
    for K, m_, ps, ss in mC["table"]:
        cells = tbl.add_row().cells
        note = ""
        if K == bestP: note += "  (PSNR 최고)"
        if K == bestS: note += "  (SSIM 최고)"
        vals = ["%.0e%s" % (K, note), "%.5f" % m_, "%.2f" % ps, "%.4f" % ss]
        for j, v in enumerate(vals):
            cells[j].text = ""; rr = cells[j].paragraphs[0].add_run(v); rr.font.size = Pt(9)
    doc.add_paragraph()
    interp(doc, "해석 (논의)")
    bullet(doc, "|H|²이 0에 가까운 고주파에서 1/|H|²이 폭발해 역필터에 가까워져 잡음을 극단적으로 증폭한다"
                "(K=1e-6에서 PSNR ~5 dB — 선명해 보여도 잡음에 파묻힘).", lead="K가 너무 작으면: ")
    bullet(doc, "분모가 K에 지배되어 필터가 H*/K에 가까워지고, 역필터링이 약해져 흐릿한(과평활) 복원이 된다(잡음은 적지만 해상도 손실).", lead="K가 너무 크면: ")
    bullet(doc, "K는 잡음 억제 ↔ 디블러 선명도의 트레이드오프를 조절한다. K ≈ S_n/S_f(잡음대신호 전력비)일 때 최적에 가깝다.", lead="K의 역할: ")
    bullet(doc, "PSNR 최고는 K=%.0e(~%.1f dB)로 화소오차 균형이 가장 좋다. SSIM은 더 큰 K=%.0e에서 최고인데, "
                "SSIM이 잔존 잡음(구조 교란)에 더 민감해 조금 더 평활한 복원을 선호하기 때문이다. 즉 '최적 K'는 "
                "화소오차냐 지각적 구조냐에 따라 달라질 수 있다."
                % (bestP, max(r[2] for r in mC["table"]), bestS), lead="최적 K: ")

    doc.add_paragraph()
    para(doc, "전체 요약", bold=True, size=12)
    para(doc, "Part A는 점처리·히스토그램 향상을 직접 구현해 전역(HE·스트레칭·감마)과 국소(AHE·CLAHE) 방법의 "
              "대비/잡음 트레이드오프를 비교했다. Part B는 합성곱 정리를 검증(선형 합성곱용 zero-padding으로 공간·주파수 "
              "결과가 반올림 수준까지 동일)하고 블러=저역통과, 샤프닝=고역통과임을 확인했다. Part C는 가우시안 블러+잡음 "
              "열화에 Wiener 필터를 설계해 정규화 상수 K가 잡음 억제와 디블러 선명도의 균형을 어떻게 조절하는지 정량적으로 보였다.")

    doc.save(DOCX); print("saved", DOCX)


if __name__ == "__main__":
    build()
