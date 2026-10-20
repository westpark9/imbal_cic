#!/usr/bin/env python3
"""Reframe the Korean 1002 deck: claim first (user direction 2026-10-02).

Input : kor/1002_labmeeting_ko_draft.pptx as left by revise_1002_ko.py (10 slides, EXP61/63 numbers).
Output: the same file, 12 slides — two new claim slides in front, slides 9–10 rebuilt
        (feasibility results of the four follow-up approaches; temporal-arrival protocol with the
        first EXP68 numbers when available). Slides 1–8 of the previous deck are kept as is.
Numbers are read from the saved runs; no experiment is launched here.
"""
import glob
import hashlib
import json
import numpy as np
import re
import shutil
import time
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'docs/slides'
OUTPUT = HERE / 'kor/1002_labmeeting_ko_draft.pptx'
QA = HERE / 'kor/1002_draft_qa'
ARCHIVE = HERE / 'archive/20261002_before_claim_first'
NAVY, BLUE, CYAN = '00306C', '007BC6', '00B8EE'
CARD, HEAD, LINE = 'F3F6FA', 'E7F0F8', 'D9E2EC'
INK, GREY, MUTED = '21262E', '5D6773', '8393A5'
GOOD, BAD = '1B7F5A', 'B0413E'
FONT, NUMBER_FONT = 'Pretendard', 'DejaVu Sans Mono'
NUM = re.compile(r'^[+−-]?[\d,]+(?:\.\d+)?%?$')


def color(value):
    return RGBColor.from_string(value)


def set_text(frame, value, size=16, bold=False, ink=INK, align=PP_ALIGN.LEFT, middle=True, inset=.10, font=FONT):
    frame.clear(); frame.word_wrap = True; frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = Inches(inset); frame.margin_top = frame.margin_bottom = Inches(.035)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE if middle else MSO_ANCHOR.TOP
    for i, value_line in enumerate(str(value).split('\n')):
        p = frame.paragraphs[0] if not i else frame.add_paragraph()
        p.alignment = align; p.space_before = p.space_after = Pt(0); p.line_spacing = 1.10
        etree.SubElement(p._p.get_or_add_pPr(), qn('a:buNone'))
        r = p.add_run(); r.text = value_line; r.font.name = font; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color(ink)
        rp = r._r.get_or_add_rPr(); rp.set('lang', 'ko-KR')
        for name in ['a:ea', 'a:cs']:
            etree.SubElement(rp, qn(name)).set('typeface', font)


def text(s, x, y, w, h, value, **kwargs):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); set_text(sh.text_frame, value, **kwargs); return sh


def box(s, x, y, w, h, fill=CARD, border=None, rounded=True):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        sh.adjustments[0] = .035
    sh.fill.solid(); sh.fill.fore_color.rgb = color(fill); sh.shadow.inherit = False
    if border:
        sh.line.color.rgb = color(border); sh.line.width = Pt(.8)
    else:
        sh.line.fill.background()
    return sh


def table(s, x, y, widths, headers, rows, row_h=.48, font=13, first_col_bold=True):
    sh = s.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * (len(rows) + 1)))
    t = sh.table; t.first_row = t.horz_banding = False
    for c, width in enumerate(widths):
        t.columns[c].width = Inches(width)
    for r, values in enumerate([headers] + rows):
        t.rows[r].height = Inches(row_h)
        for c, value in enumerate(values):
            value = str(value); cell = t.cell(r, c); cell.fill.solid()
            cell.fill.fore_color.rgb = color(NAVY if r == 0 else ('FFFFFF' if r % 2 else CARD))
            numeric = bool(NUM.fullmatch(value)) and c > 0 and r > 0
            align = PP_ALIGN.RIGHT if numeric else (PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
            set_text(cell.text_frame, value, size=font - .35 if numeric else font, bold=(r == 0 or (c == 0 and first_col_bold)),
                     ink='FFFFFF' if r == 0 else INK, align=align, font=NUMBER_FONT if numeric else FONT, inset=.07)
            cell.margin_left = cell.margin_right = Inches(.07); cell.margin_top = cell.margin_bottom = Inches(.025)
    return sh


def chrome(s, title, section):
    text(s, .62, .30, 12.09, .65, title, size=28, bold=True, ink=NAVY, inset=0)
    box(s, 0, 6.95, 13.333333, .55, fill=NAVY, rounded=False)
    text(s, 9.05, 7.03, 3.00, .30, section, size=10, ink='D6E4F3', align=PP_ALIGN.RIGHT)
    text(s, 12.23, 7.01, .48, .34, '0', size=11, bold=True, ink='FFFFFF', align=PP_ALIGN.RIGHT)


def new_slide(prs, title, section):
    layout = next(l for l in prs.slide_layouts if l.name == 'TITLE_ONLY')
    s = prs.slides.add_slide(layout)
    for sh in list(s.shapes):
        s.shapes._spTree.remove(sh._element)
    chrome(s, title, section); return s


def rebuild(s, title, section):
    for sh in list(s.shapes):
        s.shapes._spTree.remove(sh._element)
    chrome(s, title, section); return s


def subtitle(s, value, size=13):
    return text(s, .62, 1.02, 12.09, .35, value, size=size, ink=GREY, inset=0)


def footer(s, value):
    text(s, .62, 7.035, 8.20, .30, value, size=11, bold=True, ink='FFFFFF', inset=0)


def card(s, x, y, w, h, title):
    box(s, x, y, w, h); box(s, x, y, w, .50, fill=HEAD, rounded=False)
    text(s, x + .17, y + .065, w - .34, .36, title, size=17, bold=True, ink=NAVY, inset=0)


def notes(s, value):
    s.notes_slide.notes_text_frame.text = value


def highlight(sh, row, cols, fill=HEAD, ink=BLUE):
    for c in cols:
        cell = sh.table.cell(row, c); cell.fill.fore_color.rgb = color(fill)
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True; r.font.color.rgb = color(ink)


def latest(pattern):
    paths = [p for p in sorted(glob.glob(str(ROOT / pattern))) if Path(p + '/results.json').exists()]
    return json.loads(Path(paths[-1] + '/results.json').read_text()) if paths else None


# ------------------------------------------------------------------ data from saved runs
sota = json.loads((ROOT / 'docs/research/20260930/exp61_report_data.json').read_text())
G = {ds: sota['datasets'][ds]['results'] for ds in ['cic2018', 'toniot']}
exp64 = latest('tabpfn/results/*_toniot_exp64_benign_diversity_s43')
exp66 = latest('tabpfn/results/*_toniot_exp66_supcon_s43')
exp65 = latest('tabpfn/results/*_cic2018_exp65_lito_lite_web_attacks_s43')      # the full-test run is the latest
exp67 = {}
for p in sorted(glob.glob(str(ROOT / 'tabpfn/results/*_exp67_prompt_tuned_s4*'))):
    if not Path(p + '/results.json').exists():
        continue   # interrupted run (no results)
    r = json.loads(Path(p + '/results.json').read_text())
    if r['args'].get('full_test'):
        exp67[(r['dataset'], r['args']['seed'], r['args']['query_sampling'], r['context_rows'])] = r['results']['tuned']['macro_f1']
exp68 = latest('tabpfn/results/*_toniot_exp68_temporal_arrival_s43')
assert exp64 and exp66 and exp65 and exp67, 'feasibility runs missing'
assert exp65['args']['full_test'] and 'c0+dup' in exp65['results']['arms'] and 'c0+syn' in exp65['results']['arms']
ton_tuned = {k[1]: v for k, v in exp67.items() if k[0] == 'toniot' and k[2] == 'natural' and k[3] == 3000}
cic_tuned = [v for k, v in exp67.items() if k[0] == 'cic2018' and k[2] == 'natural'][0]
ton_bal = [v for k, v in exp67.items() if k[0] == 'toniot' and k[2] == 'balanced'][0]

# ------------------------------------------------------------------ deck
ARCHIVE.mkdir(parents=True, exist_ok=True)
for name in ['1002_labmeeting_ko_draft.pptx', '1002_labmeeting_ko_draft.pdf']:
    src = HERE / 'kor' / name
    if src.exists() and not (ARCHIVE / name).exists():
        shutil.copy2(src, ARCHIVE / name)
SOURCE = ARCHIVE / '1002_labmeeting_ko_draft.pptx'   # the 10-slide deck as left by revise_1002_ko.py (archived on first run)
prs = Presentation(SOURCE)
assert len(prs.slides) == 10, 'source deck must be the 10-slide revise_1002_ko.py output'
old = list(prs.slides)

# A. claim
s = new_slide(prs, '왜 TabPFN인가 · 재학습 없이 context로 적응하는 탐지기', '연구 의도')
subtitle(s, f'같은 100,000행에서 정확도는 비슷하다 · CIC2018 XGBoost {G["cic2018"]["xgb"]["macro_f1"]:.4f} vs TabPFN {G["cic2018"]["global_raw"]["macro_f1"]:.4f} · ToN {G["toniot"]["xgb"]["macro_f1"]:.4f} vs {G["toniot"]["global_raw"]["macro_f1"]:.4f} · 차이는 새 공격이 나타났을 때의 갱신 방식')
card(s, .62, 1.55, 5.90, 3.05, '학습 모델 (XGBoost 등)')
text(s, .80, 2.15, 5.55, 1.05, '새 공격 출현 → 라벨 수집 → 전체 재학습 → 재배포\n모델 자체가 바뀌므로 회귀 검증을 다시 한다', size=13.5, ink=INK, inset=0)
text(s, .80, 3.25, 5.55, 1.20, f'전체 train 재학습 {G["cic2018"]["xgb_full"]["fit_seconds"]:.0f} s / {G["toniot"]["xgb_full"]["fit_seconds"]:.0f} s (CIC / ToN, EXP61)\n추론 1,000행당 {G["cic2018"]["xgb"]["seconds_per_1000_rows"]:.4f} s · 매우 빠름', size=12.5, ink=GREY, inset=0)
card(s, 6.81, 1.55, 5.90, 3.05, 'TabPFN (backbone 고정)')
text(s, 6.99, 2.15, 5.55, 1.05, '새 공격 출현 → 소수 라벨을 context에 추가 → 즉시 추론\n모델은 그대로, 바뀌는 것은 context 행뿐', size=13.5, ink=INK, inset=0)
text(s, 6.99, 3.25, 5.55, 1.20, f'학습 0 s · context 적재 {G["toniot"]["global_raw"]["fit_seconds"]:.0f} s\n추론 1,000행당 {G["toniot"]["global_raw"]["seconds_per_1000_rows"]:.2f} s · XGBoost보다 약 1,000배 느림', size=12.5, ink=GREY, inset=0)
box(s, .62, 4.85, 12.09, 1.55, fill=HEAD)
text(s, .80, 4.93, 11.75, .36, '연구 질문', size=14, bold=True, ink=NAVY, inset=0)
text(s, .80, 5.33, 11.75, .95, '제한된 context 예산(100,000행 ≪ 보유 train 수백만 행)에 무엇을 담아야 tail 공격이 회수되는가\n평가 기준: 전체 macro-F1 경쟁이 아니라 tail 클래스 회수 · benign 오탐 · 갱신에 드는 라벨과 시간', size=14, ink=INK, inset=0)
footer(s, '주장: 성능 경쟁이 아니라 적응 비용과 tail 회수로 TabPFN의 자리를 보인다')
notes(s, '사용자 방향(10-02): 수치 경쟁이 아니라 모델의 의도를 먼저 세운다\n'
      'XGB와 TabPFN의 100k 비교 수치는 EXP61 전체 test macro-F1\n'
      '재학습 시간은 EXP61 전체 train XGB fit(CIC 8,666,430행 / ToN 4,669,119행), 검증·배포 시간은 포함하지 않음\n'
      'TabPFN 추론 비용이 큰 것은 숨기지 않는다. 주장은 갱신 방식(라벨 효율·무학습)에 있다\n'
      '적응 효율은 다음 실험(시간순 출현 시나리오)에서 측정한다')

# B. target and structural confusion
s = new_slide(prs, '목표는 tail 회수 · benign 혼동의 원인은 클래스마다 다르다', '연구 의도')
subtitle(s, 'benign과 공격이 시간순으로 공존하는 window 라벨 → 혼동은 피할 수 없다 · 10-01 진단: Global이 틀린 행과 train의 최근접 거리 (표준화 46차원 L2, 참조 benign은 0.01)', size=12.2)
rows = [['ToN scanning', '0.039', '0.60', '2.92', 'context 안 benign:scanning 27:1 (prior)', '가능 · expert+V 0.90'],
        ['CIC infiltration', '0.288', '0.11', '0.17', 'benign과 겹치는 영역', '불가 · feature 한계'],
        ['ToN ransomware', '0.121', '0.12', '2.84', 'benign 라벨 행이 ransomware 근사 복제', '불가 · 라벨 모순'],
        ['ToN mitm', '0.132', '9.76', '11.40', 'train에 없는 새 benign 블록 (drift)', '시간순 갱신 · novelty 거부'],
        ['CIC web attack', '0.210', '2.23', '1.79', '양성 127행 · dos와 오인', '부분 · 양성 비율 +0.11']]
sh = table(s, .62, 1.62, [1.75, 1.05, 1.42, 1.42, 3.60, 2.85], ['클래스', 'Global F1', '→ train 해당 클래스', '→ train benign', '원인', 'context로 고칠 수 있나'], rows, row_h=.56, font=12.5)
highlight(sh, 1, [5], fill='EEF8F2', ink=GOOD); highlight(sh, 5, [5], fill='EEF8F2', ink=GOOD)
highlight(sh, 2, [5], fill='FBEEEE', ink=BAD); highlight(sh, 3, [5], fill='FBEEEE', ink=BAD)
text(s, .62, 5.10, 12.09, .40, '거리 열: scanning은 Global이 놓친(FN) 행, 나머지는 benign을 해당 클래스로 부른(FP) 행 기준', size=11.5, ink=GREY, inset=0)
box(s, .62, 5.58, 12.09, .95, fill=HEAD)
text(s, .80, 5.66, 11.75, .80, 'context 설계의 대상 = prior형(scanning)과 희소 양성형(web attack) · 겹침·라벨 모순은 데이터 상한으로 보고 · drift는 시간순 라벨 갱신으로 다룬다', size=13.5, bold=True, ink=NAVY, inset=0)
footer(s, 'tail별로 원인이 다르므로 목표도 하나가 아니다 · 회수 가능한 tail에 context를 쓴다')
notes(s, '10-01 분석(lablog/report/0930.md): EXP63 K=6 cache의 test feature와 route 풀(train 대리)의 표준화 L2 최근접 거리, 집합당 4,000행 표본\n'
      'scanning: FN 5,886행이 train scanning에 0.60, benign에 2.92 → 유사성 아님, C0 prior 문제. expert e1 + verifier로 0.90(EXP62/63)\n'
      'infiltration: FP benign 7,935행이 train infiltration 0.11·benign 0.17에 모두 가까움 → 겹침. 모든 SOTA 0.31 이하\n'
      'ransomware: FP benign 9,368행이 train ransomware 0.12, benign 2.84 → benign 라벨 행이 ransomware와 근사 동일\n'
      'mitm: FP benign 13,125행이 train 어느 클래스에서도 10 안팎 → test에만 있는 블록. 정답 배정 oracle에서도 FP 55,278로 악화\n'
      'web attack: EXP65 전체 test, 실제 web 행 복제 1,479행으로 F1 0.210 → 0.320\n'
      '참조 benign(Global 정답)은 train benign까지 0.01')

# 9. feasibility of the four follow-up approaches (rebuilt)
s = rebuild(old[8], '후속 네 접근의 feasibility · context 최적화만 새 정보를 만든다', '후속 실험')
subtitle(s, '10-01 저녁 · seed 43 단일(4번은 seed 44 추가) · 전체 test macro-F1 · feasibility 규모, 기록급 실험 아님', size=12.5)
e64 = exp64['results']; e66 = exp66['results']; e65 = exp65['results']['arms']
rows = [['1  Benign 다양화', 'C0의 benign 75,000행만 교체 (time · k-center · 역밀도 · 공격 근접)',
         f'ToN {e64["random"]["macro_f1"]:.3f} → {min(v["macro_f1"] for v in e64.values()):.3f} … {max(v["macro_f1"] for v in e64.values()):.3f}, scanning ≤ {max(v["classes"][8]["f1"] for v in e64.values()):.2f}', '효과 없음 · scanning은 prior 문제'],
        ['2  LITO식 tail 합성', 'web 224행 → 합성 2,000행, self-authentication 통과 1,479행',
         f'CIC {G["cic2018"]["global_raw"]["macro_f1"]:.4f} → 합성 {e65["c0+syn"]["macro_f1"]:.4f} · 실제 복제 {e65["c0+dup"]["macro_f1"]:.4f} (web F1 {G["cic2018"]["global_raw"]["classes"][6]["f1"]:.2f} → {e65["c0+dup"]["classes"][6]["f1"]:.2f})', '합성 기각 · 양성 비율은 유효한 knob'],
        ['3  SupCon 임베딩', 'route 풀로 encoder 학습, 16차원 임베딩을 feature에 추가',
         f'ToN {e66["raw"]["macro_f1"]:.3f} → {e66["raw+emb"]["macro_f1"]:.3f} (raw+emb) · {e66["emb"]["macro_f1"]:.3f} (emb)', '효과 없음'],
        ['4  Context 최적화', '3,000행 context를 train 풀 query의 NLL로 300 step 최적화 (backbone 고정)',
         f'ToN 0.645 → {ton_tuned[43]:.3f} / {ton_tuned[44]:.3f} (seed 43 / 44) · CIC 2,100행 {cic_tuned:.3f}', f'100k C0 {G["toniot"]["global_raw"]["macro_f1"]:.3f} / {G["cic2018"]["global_raw"]["macro_f1"]:.3f} 상회·동급']]
sh = table(s, .62, 1.55, [1.85, 4.05, 3.85, 2.34], ['접근', '무엇을 바꿨나', '결과 (전체 test)', '판정'], rows, row_h=.78, font=11.8)
highlight(sh, 4, [0, 3], fill='EEF8F2', ink=GOOD)
box(s, .62, 5.58, 12.09, .95, fill=HEAD)
text(s, .80, 5.64, 11.75, .84, f'ToN: 3,000행 최적화 context {ton_tuned[43]:.3f} / {ton_tuned[44]:.3f} > 100,000행 무작위 C0 {G["toniot"]["global_raw"]["macro_f1"]:.3f} > XGBoost-100k {G["toniot"]["xgb"]["macro_f1"]:.3f} · 최적화 query는 자연 비율이어야 한다 (균형 query {ton_bal:.3f})\n제한: 미분 경로는 24 GB에서 5,000행까지 · estimator 1개 · seed 민감(mitm 0.32 / 0.65)', size=12.5, ink=NAVY, inset=0)
footer(s, '정보는 행 수가 아니라 구성에서 온다 · 다음은 최적화 context를 Global과 expert에 쓰는 설계')
notes(s, '근거: tabpfn/results/20261001_*_exp64_benign_diversity_s43 (부분집합 screening), *_exp66_supcon_s43, *_exp65_lito_lite_web_attacks_s43 (전체 test), *_exp67_prompt_tuned_s43/s44 (전체 test)\n'
      '1·3은 층화 부분집합(클래스별 20k cap)에서 screening한 값이라 절대값은 전체 test와 다름, 상대 비교만 사용\n'
      '2: 합성은 v3 분류기로 분위 구간을 예측하는 imputation 3회 + Global self-authentication p≥0.5. 같은 수의 실제 행 복제가 이기므로 이득은 prior 효과\n'
      '4: TabPFN differentiable_input, 라벨 고정, context 행만 Adam으로 갱신, query는 D_global 풀(test 미사용). 6개 feature는 grad 비유한으로 고정\n'
      '4의 첫 run은 NaN grad가 결측으로 숨어 무효였고, 마스킹 후 재실행한 값만 사용\n'
      '모두 단일 seed(4번만 2 seed)·관찰한 holdout의 예비 결과')

# 10. temporal arrival protocol (rebuilt), with first EXP68 numbers when present
s = rebuild(old[9], '다음 실험 · 공격이 시간순으로 나타나는 시나리오', '다음 실험')
subtitle(s, 'test를 시간순 5구간으로 분할 · 구간마다 각 family의 첫 10% 중 최대 50행만 라벨 → 나머지 평가 · 라벨은 누적 · ToN seed 43', size=12.4)
text(s, .62, 1.55, 6.00, .36, 'ToN test에서 family가 나타나는 순서 (구간별 출현 클래스)', size=14, bold=True, ink=NAVY, inset=0)
periods = [('1', 'ddos · dos · injection · scanning'), ('2', 'ddos'), ('3', 'password'), ('4', 'xss'), ('5', 'backdoor · mitm · ransomware')]
for i, (k, fam) in enumerate(periods):
    x = .62 + i * 1.22
    box(s, x, 1.98, 1.12, 1.18, fill=HEAD if i in (0, 4) else CARD, border=LINE)
    text(s, x + .04, 2.03, 1.04, .30, f'구간 {k}', size=12.5, bold=True, ink=NAVY, align=PP_ALIGN.CENTER, inset=0)
    text(s, x + .04, 2.34, 1.04, .80, fam.replace(' · ', '\n'), size=10.2, ink=INK, align=PP_ALIGN.CENTER, inset=0)
text(s, .62, 3.20, 6.00, .32, 'benign은 전 구간에 공존 · 구간 5의 benign은 train에 없는 블록을 포함', size=11.2, ink=GREY, inset=0)
text(s, 6.95, 1.55, 5.76, .36, '비교 조건 (같은 평가 행)', size=14, bold=True, ink=NAVY, inset=0)
_have68 = bool(exp68 and len(exp68['results'].get('tabpfn_context', [])) == 5)
if _have68:  # measured in EXP68 itself (per period, mean of 5), so the slide quotes one source only
    _R = exp68['results']
    _ctx = f'적재 {np.mean([r["fit_seconds"] for r in _R["tabpfn_context"]]):.0f} s / 구간'; _xgb = f'재학습 {np.mean([r["fit_seconds"] for r in _R["xgb_retrain"]]):.0f} s / 구간'
else:
    _ctx = f'적재 {G["toniot"]["global_raw"]["fit_seconds"]:.0f} s'; _xgb = f'{G["toniot"]["xgb"]["fit_seconds"]:.0f} s / 구간'
arms = [['TabPFN 고정', '갱신 없음 (현재 Global)', '0 s'], ['TabPFN + context', '라벨 행을 context에 추가, 학습 없음', _ctx],
        ['XGBoost 고정', '갱신 없음', '0 s'], ['XGBoost 재학습', 'C0 + 누적 라벨로 재학습', _xgb]]
table(s, 6.95, 1.98, [1.80, 2.56, 1.40], ['조건', '갱신 방식', '구간당 비용'], arms, row_h=.30 if _have68 else .42, font=10.5 if _have68 else 11.5)
if exp68 and len(exp68['results'].get('tabpfn_context', [])) == 5:
    R = exp68['results']; names = exp68['class_names']
    def f1(arm, w, cls):
        c = next(r for r in R[arm][w]['classes'] if r['cls'] == cls); return f'{c["f1"]:.3f}' if c['support'] else '–'
    hdr = ['구간', 'TabPFN 고정', '+context', 'XGB 고정', 'XGB 재학습']
    rows = [[f'{w+1}', f'{R["tabpfn_frozen"][w]["macro_f1_present"]:.3f}', f'{R["tabpfn_context"][w]["macro_f1_present"]:.3f}', f'{R["xgb_frozen"][w]["macro_f1_present"]:.3f}', f'{R["xgb_retrain"][w]["macro_f1_present"]:.3f}'] for w in range(5)]
    rows.append(['scanning (1)', f1('tabpfn_frozen', 0, 'scanning'), f1('tabpfn_context', 0, 'scanning'), f1('xgb_frozen', 0, 'scanning'), f1('xgb_retrain', 0, 'scanning')])
    rows.append(['mitm (5)', f1('tabpfn_frozen', 4, 'mitm'), f1('tabpfn_context', 4, 'mitm'), f1('xgb_frozen', 4, 'mitm'), f1('xgb_retrain', 4, 'mitm')])
    rows.append(['ransomware (5)', f1('tabpfn_frozen', 4, 'ransomware'), f1('tabpfn_context', 4, 'ransomware'), f1('xgb_frozen', 4, 'ransomware'), f1('xgb_retrain', 4, 'ransomware')])
    text(s, .62, 3.60, 12.09, .34, '첫 결과 (EXP68, ToN, seed 43) · 구간별 macro-F1은 그 구간에 출현한 클래스 기준 · 아래 세 행은 tail F1', size=12.5, bold=True, ink=NAVY, inset=0)
    sh = table(s, .62, 3.98, [2.05, 2.51, 2.51, 2.51, 2.51], hdr, rows, row_h=.255, font=10.2)
    sc = next(r for r in R['tabpfn_context'][0]['classes'] if r['cls'] == 'scanning')['f1']; sx = next(r for r in R['xgb_retrain'][0]['classes'] if r['cls'] == 'scanning')['f1']
    text(s, .62, 6.33, 12.09, .50, f'라벨 50행을 100,000행 context에 더하는 것만으로는 scanning이 열리지 않는다 (0.038 → {sc:.3f}; XGB 재학습 {sx:.3f}) · 효과를 낸 것은 구성이다: scanning block expert + V 0.86, 최적화 context 0.71 · 적응의 단위는 라벨 추가가 아니라 context 구성', size=11.2, ink=NAVY, bold=True, inset=0)
    footer(s, f'비용: TabPFN 학습 0 s, context 적재 {sum(r["fit_seconds"] for r in R["tabpfn_context"]):.0f} s · 추론 {sum(r["predict_seconds"] for r in R["tabpfn_context"]):.0f} s / XGB 재학습 {sum(r["fit_seconds"] for r in R["xgb_retrain"]):.0f} s · 추론 {sum(r["predict_seconds"] for r in R["xgb_retrain"]):.0f} s · 라벨 {R["tabpfn_context"][-1]["labels_used"]}행')
else:
    text(s, .62, 4.10, 12.09, .36, '측정 항목', size=14, bold=True, ink=NAVY, inset=0)
    text(s, .62, 4.50, 12.09, 1.30, '구간별 macro-F1(출현 클래스 기준) · tail F1 · benign 오탐률 · 사용한 라벨 수 · 학습·추론 시간\n확장: 한 family를 train에서 제외하고 test 구간에서 처음 나타나게 하는 unseen 출현(OOD) 시나리오', size=13, ink=INK, inset=0)
    footer(s, '실행 중 · 결과는 랩로그 0930.md와 HTML 보고서에 추가')
notes(s, 'EXP68 (tabpfn/scripts/exp68_temporal_arrival.py): test를 시간순 정렬 후 균등 5구간, 각 구간에서 클래스별 첫 10% 행 중 최대 50행을 라벨로 사용(분석가가 새 family의 첫 flow를 라벨링하는 상황), 나머지 행 평가\n'
      'TabPFN + context: C0 100k에 누적 라벨 행을 추가하고 fit(적재)만 다시 함, backbone·전처리 동일\n'
      'XGB 재학습: C0 + 누적 라벨로 EXP61 설정(300 trees, depth 8) 재학습\n'
      '구간별 macro-F1은 그 구간 평가 행에 존재하는 클래스만 평균하므로 전체 test macro-F1과 직접 비교하지 않음\n'
      'ToN family 출현 순서는 test 시간 5분위 집계(1: ddos·dos·injection·scanning, 2: ddos, 3: password, 4: xss, 5: backdoor·mitm·ransomware)\n'
      '확장: family holdout(train 제외) → 처음 보는 공격의 출현. 현재는 모든 family가 train에 있는 prequential 설정')

# ------------------------------------------------------------------ order: A, B, 1..10
lst = prs.slides._sldIdLst; ids = list(lst)
for el in ids:
    lst.remove(el)
for i in [10, 11] + list(range(10)):
    lst.append(ids[i])
for n, s in enumerate(prs.slides, 1):
    for sh in s.shapes:
        if sh.has_text_frame and abs(sh.left - Inches(12.23)) < Inches(.05) and abs(sh.top - Inches(7.01)) < Inches(.05):
            set_text(sh.text_frame, str(n), size=11, bold=True, ink='FFFFFF', align=PP_ALIGN.RIGHT)
prs.core_properties.title = '10월 2일 랩미팅 · 한국어 (주장 우선 구성)'
prs.core_properties.comments = '2026-10-02 주장 우선 재구성: 의도·목표 2장 추가, 후속 접근 feasibility 결과와 시간순 시나리오로 9·10장 교체'
prs.save(OUTPUT)

# ------------------------------------------------------------------ checks
check = Presentation(OUTPUT); assert len(check.slides) == 12
issues, dump, numeric_cells = [], [], 0
for i, s in enumerate(check.slides, 1):
    entries, text_shapes = [], []
    for sh in s.shapes:
        if sh.left < 0 or sh.top < 0 or sh.left + sh.width > check.slide_width + 15 or sh.top + sh.height > check.slide_height + 15:
            issues.append([i, 'outside slide', sh.name])
        if sh.top < Inches(6.90) and sh.top + sh.height > Inches(6.90):
            issues.append([i, 'footer collision', sh.name])
        if sh.has_text_frame and sh.text.strip():
            entries.append(sh.text); text_shapes.append(sh)
        if sh.has_table:
            entries.extend(' | '.join(c.text for c in r.cells) for r in sh.table.rows)
            for r, row in enumerate(sh.table.rows):
                for c, cell in enumerate(row.cells):
                    if r > 0 and c > 0 and NUM.fullmatch(cell.text):
                        numeric_cells += 1; assert all(p.alignment == PP_ALIGN.RIGHT for p in cell.text_frame.paragraphs), (i, r, c, cell.text)
    for j, a0 in enumerate(text_shapes):
        for b0 in text_shapes[j + 1:]:
            dx = min(a0.left + a0.width, b0.left + b0.width) - max(a0.left, b0.left); dy = min(a0.top + a0.height, b0.top + b0.height) - max(a0.top, b0.top)
            if dx > Inches(.005) and dy > Inches(.005):
                issues.append([i, 'text box overlap', a0.text[:40], b0.text[:40]])
    joined = '\n'.join(entries)
    for forbidden in ['TBD', '상대사례', '관점 A', '관점 B']:
        assert forbidden not in joined, (i, forbidden)
    assert not re.search(r'합니다|입니다|습니다|[—–]', joined), (i, 'style')
    dump.append(f'## {i}\n{joined}\n')
assert not issues, issues
page_numbers = [next(sh.text for sh in s.shapes if sh.has_text_frame and abs(sh.left - Inches(12.23)) < Inches(.05)) for s in check.slides]
assert page_numbers == [str(n) for n in range(1, 13)], page_numbers
QA.mkdir(exist_ok=True)
(QA / 'slide_text.md').write_text('\n'.join(dump))
(QA / 'validation.json').write_text(json.dumps(dict(slides=12, output=str(OUTPUT.relative_to(ROOT)), updated='2026-10-02', generator='docs/slides/revise_1002_ko_v2.py',
    structure=['A 의도', 'B 목표·구조적 혼동', '1–8 이전 덱 유지', '9 후속 접근 feasibility', '10 시간순 시나리오'], exp68_included=bool(exp68 and len(exp68['results'].get('tabpfn_context', [])) == 5),
    numeric_cells_right_aligned=numeric_cells, geometry_and_style_issues=issues, archive=str(ARCHIVE.relative_to(ROOT)),
    sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest()), ensure_ascii=False, indent=2) + '\n')
print(OUTPUT, '12 slides, exp68 included:', bool(exp68 and len(exp68['results'].get('tabpfn_context', [])) == 5))
