#!/usr/bin/env python3
"""Refresh the existing Korean deck with EXP61/63 and four future directions.

No training, HTML editing, or English-deck mutation. Numeric data are read from
the saved experiments. Run with the project Python, then render with LibreOffice.
"""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

from pathlib import Path
import hashlib
import json
import re
import types

import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from intro_1002 import build as build_intro, footer, line, LABELS

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'docs/slides'
OUTPUT = HERE / 'kor/1002_labmeeting_ko_draft.pptx'
QA = HERE / 'kor/1002_draft_qa'
DATA = ROOT / 'docs/research/20260930'
SWEEP = ROOT / 'tabpfn/results/v4/exp63/20260930_exp63_k_sweep_s43'
NAVY, BLUE, CYAN = '00306C', '007BC6', '00B8EE'
CARD, HEAD, LINE = 'F3F6FA', 'E7F0F8', 'D9E2EC'
INK, GREY, MUTED = '21262E', '5D6773', '8393A5'
FONT, NUMBER_FONT = 'Pretendard', 'DejaVu Sans Mono'
NUM = re.compile(r'^[+−-]?[\d,]+(?:\.\d+)?%?$')
prs = Presentation(OUTPUT)
assert len(prs.slides) == 10
cursor = 0
checks = []


def color(value):
    return RGBColor.from_string(value)


def set_text(frame, value, size=16, bold=False, ink=INK, align=PP_ALIGN.LEFT,
             middle=True, inset=.10, font=FONT):
    frame.clear()
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = Inches(inset)
    frame.margin_top = frame.margin_bottom = Inches(.035)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE if middle else MSO_ANCHOR.TOP
    for i, value_line in enumerate(str(value).split('\n')):
        p = frame.paragraphs[0] if not i else frame.add_paragraph()
        p.alignment = align
        p.space_before = p.space_after = Pt(0)
        p.line_spacing = 1.10
        etree.SubElement(p._p.get_or_add_pPr(), qn('a:buNone'))
        r = p.add_run()
        r.text = value_line
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color(ink)
        rp = r._r.get_or_add_rPr()
        rp.set('lang', 'ko-KR')
        for name in ['a:ea', 'a:cs']:
            etree.SubElement(rp, qn(name)).set('typeface', font)


def text(s, x, y, w, h, value, **kwargs):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    set_text(sh.text_frame, value, **kwargs)
    return sh


def box(s, x, y, w, h, fill=CARD, border=None, rounded=True):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                            Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        sh.adjustments[0] = .035
    sh.fill.solid()
    sh.fill.fore_color.rgb = color(fill)
    sh.shadow.inherit = False
    if border:
        sh.line.color.rgb = color(border)
        sh.line.width = Pt(.8)
    else:
        sh.line.fill.background()
    return sh


def table(s, x, y, widths, headers, rows, row_h=.48, font=13):
    sh = s.shapes.add_table(len(rows)+1, len(headers), Inches(x), Inches(y),
                             Inches(sum(widths)), Inches(row_h*(len(rows)+1)))
    t = sh.table
    t.first_row = t.horz_banding = False
    for c, width in enumerate(widths):
        t.columns[c].width = Inches(width)
    for r, values in enumerate([headers]+rows):
        t.rows[r].height = Inches(row_h)
        for c, value in enumerate(values):
            value = str(value)
            cell = t.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = color(NAVY if r == 0 else ('FFFFFF' if r%2 else CARD))
            numeric = bool(NUM.fullmatch(value)) and c > 0 and r > 0
            align = PP_ALIGN.RIGHT if numeric else (PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
            set_text(cell.text_frame, value, size=font-.35 if numeric else font,
                     bold=(r == 0 or c == 0), ink='FFFFFF' if r == 0 else INK,
                     align=align, font=NUMBER_FONT if numeric else FONT, inset=.07)
            cell.margin_left = cell.margin_right = Inches(.07)
            cell.margin_top = cell.margin_bottom = Inches(.025)
    return sh


def slide(title, section):
    global cursor
    s = prs.slides[cursor]
    cursor += 1
    for sh in list(s.shapes):
        s.shapes._spTree.remove(sh._element)
    text(s, .62, .30, 12.09, .65, title, size=28, bold=True, ink=NAVY, inset=0)
    box(s, 0, 6.95, 13.333333, .55, fill=NAVY, rounded=False)
    text(s, 9.05, 7.03, 3.00, .30, section, size=10, ink='D6E4F3', align=PP_ALIGN.RIGHT)
    text(s, 12.23, 7.01, .48, .34, str(cursor), size=11, bold=True,
         ink='FFFFFF', align=PP_ALIGN.RIGHT)
    return s


def subtitle(s, value, size=13):
    return text(s, .62, 1.02, 12.09, .35, value, size=size, ink=GREY, inset=0)


def highlight(sh, row, cols, fill=HEAD, ink=BLUE):
    for c in cols:
        cell = sh.table.cell(row, c)
        cell.fill.fore_color.rgb = color(fill)
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = color(ink)


def notes(s, value):
    s.notes_slide.notes_text_frame.text = value


def card(s, x, y, w, h, title):
    box(s, x, y, w, h)
    box(s, x, y, w, .50, fill=HEAD, rounded=False)
    text(s, x+.17, y+.065, w-.34, .36, title, size=17, bold=True, ink=NAVY, inset=0)


def reference(s, x, y, w, h, value, url):
    sh = text(s, x, y, w, h, value, size=11, ink=GREY, inset=0)
    for p in sh.text_frame.paragraphs:
        for r in p.runs:
            r.hyperlink.address = url
    return sh


cap = read_record(DATA/'exp63_capability_report_data.json')
sota = read_record(DATA/'exp61_report_data.json')
assert cap['seed'] == sota['seed'] == 43 and cap['ks'] == [2, 4, 6, 8]
systems = {}
for ds, d in cap['datasets'].items():
    systems[ds] = pd.read_csv(SWEEP/f'{ds}_k4/class_metrics.csv').query("arm=='s1v0' and split=='full_test'").set_index('class')
    for k, b in d['banks'].items():
        raw = pd.read_csv(SWEEP/f'{ds}_k{k}/summary.csv').query("arm=='s1v0' and split=='full_test'").iloc[0]
        for metric in ['macro_f1', 'rows', 'logical_expert_calls', 'accepted', 'helpful', 'harmful']:
            assert abs(float(raw[metric])-float(b['policy'][metric])) < 1e-10, (ds, k, metric)
        direct = pd.read_csv(SWEEP/f'{ds}_k{k}/diagnostics/expert_class_metrics.csv')
        for split in ['full_test', 'cal_confirm']:
            for model, result in b['models'][split].items():
                src = direct.query('split==@split and model==@model').set_index('class')
                for c in result['classes']:
                    for metric in ['support', 'f1', 'precision', 'recall']:
                        assert abs(float(src.loc[c['class'], metric])-float(c[metric])) < 1e-10
    assert round(d['banks']['4']['models']['full_test']['global']['summary']['macro_f1'],4) == round(sota['datasets'][ds]['results']['global_raw']['macro_f1'],4)
checks.append('EXP63 summary and all class metrics match the saved source CSVs')

h = types.SimpleNamespace(**globals())
build_intro(h, through=3)
assert cursor == 3

# 4. Context construction as the first tried design, not a final method.
s = slide('첫 context 구성 실험 · 공통 anchor와 residual block', 'Context 구성')
subtitle(s, '정제 CIC2018·ToN · seed 43 · Global 고정 · K = 2, 4, 6, 8')
card(s, .62, 1.62, 5.32, 4.67, '각 expert에 제공하는 context')
box(s, .91, 2.37, 4.74, 1.12, fill='FFFFFF', border=LINE)
text(s, 1.09, 2.49, 4.38, .33, '공통 anchor', size=19, bold=True, ink=NAVY, inset=0)
text(s, 1.09, 2.94, 4.38, .30, '모든 클래스의 기본 사례 · 클래스당 최대 2,000행', size=12.2, ink=GREY, inset=0)
text(s, 2.78, 3.62, 1.00, .36, '+', size=23, bold=True, ink=BLUE, align=PP_ALIGN.CENTER)
box(s, .91, 4.12, 4.74, 1.12, fill=HEAD)
text(s, 1.09, 4.24, 4.38, .33, 'Residual 군집별 특화 block', size=18, bold=True, ink=NAVY, inset=0)
text(s, 1.09, 4.68, 4.38, .31, 'Global 오류 양상과 입력 특성을 반영한 표본 선택', size=12.1, ink=GREY, inset=0)
text(s, .96, 5.53, 4.64, .43, '같은 TabPFN backbone · 서로 다른 context', size=14, bold=True, ink=NAVY, align=PP_ALIGN.CENTER, inset=0)
text(s, 6.37, 1.65, 6.34, .40, 'K에 따른 expert 구성과 context 행 수 합계', size=18, bold=True, ink=NAVY, inset=0)
rows=[]
for k in cap['ks']:
    rows.append([f'K = {k}', f"{cap['datasets']['cic2018']['banks'][str(k)]['total_context_rows']:,}", f"{cap['datasets']['toniot']['banks'][str(k)]['total_context_rows']:,}"])
table(s, 6.37, 2.24, [1.28,2.53,2.53], ['Expert 수','CIC2018','ToN'], rows, row_h=.53, font=14)
table(s, 6.37, 5.21, [2.12,2.11,2.11], ['공통 anchor','CIC2018','ToN'],
      [['행 수','12,224','18,717']], row_h=.43, font=13)
text(s, 6.37, 6.19, 6.34, .42, '합계에 expert별 anchor 반복 포함 · K별 총예산 상이', size=11.5, ink=GREY, inset=0)
footer(h, s, 'Residual block + anchor를 context 구성의 첫 비교 기준으로 사용')
notes(s, '현재까지 시도한 context 구성 방법 중 하나로 제시\n'
      'Global 100,000행 context 및 anchor ID 고정, residual KMeans 군집 수만 2/4/6/8로 변경\n'
      'Residual에는 입력 표현, Global 예측 확률, 정답과 확률의 차이, 오류 크기가 포함\n'
      'Expert context에는 선택된 원래 feature와 label 쌍을 제공, residual을 추가 feature로 제공하지 않음\n'
      'Block은 Global이 틀린 행만으로 구성되지 않으며 다양성 선택을 포함\n'
      '각 block 상한 186,000행, 군집이 작으면 가능한 표본 사용\n'
      'K별 합계는 중복 anchor를 포함한 context 행 수의 합이며 고유 학습 표본 수와 다름\n'
      'K가 증가하면 총 context 크기도 달라지므로 고정 총예산 실험으로 해석하지 않음\n'
      '근거: docs/research/20260930/exp63_capability_report_data.json')

# 5. Same-population direct competence, no A/B routing comparison.
s = slide('동일 test에서 비교한 Global과 expert의 분류 성능', 'Expert 평가')
subtitle(s, 'K = 4 · 클래스별 F1 · test 표본 수 내림차순 · 파란 숫자: 표시값 기준 Global 대비 개선', size=12.4)
for ds, title, x in [('cic2018','CIC2018',.62), ('toniot','ToN',6.81)]:
    bank = cap['datasets'][ds]['banks']['4']
    models = bank['models']['full_test']
    names = sorted(models['global']['classes'],key=lambda c:(-c['support'],c['class']))
    lookup = {m:{c['class']:c for c in r['classes']} for m,r in models.items()}
    text(s, x, 1.52, 5.90, .37, title, size=18, bold=True, ink=NAVY, inset=0)
    rows = [[LABELS[c['class']],f"{c['support']:,}"]+[f"{lookup[m][c['class']]['f1']:.3f}" for m in models] for c in names]
    rows.append(['Macro-F1','']+[f"{r['summary']['macro_f1']:.3f}" for r in models.values()])
    sh = table(s, x, 2.00, [1.40,1.10,.68,.68,.68,.68,.68], ['클래스','표본 수','Global','e1','e2','e3','e4'],rows,row_h=.345,font=10.5)
    for r, c in enumerate(names,1):
        for j,m in enumerate(list(models)[1:],3):
            if round(lookup[m][c['class']]['f1'],3)>round(c['f1'],3):
                highlight(sh,r,[j],fill='EEF5FB')
    highlight(sh,len(rows),list(range(7)),fill=HEAD,ink=NAVY)
    if ds=='cic2018':
        text(s, x+.02, 5.31, 5.86, .72, 'Infiltration: 모든 expert에서 Global보다 낮은 F1\nWeb attack: e1의 test 개선, validation 확인에서는 미개선',size=12,ink=GREY,inset=0)
text(s,.62,6.37,12.09,.33,'각 expert가 같은 전체 test를 분류 · 다른 클래스에서 발생한 FP까지 포함한 F1',size=12.5,ink=GREY,inset=0)
footer(h,s,'ToN Scanning은 e1~e4 모두 test와 validation 확인에서 Global 대비 개선')
notes(s,'EXP63 K=4 saved fitted bank 기준\n'
      '각 expert가 같은 전체 test를 분류한 raw multiclass F1, S/V 적용 전\n'
      '이 표는 정답 residual로 표본을 expert에 배정한 oracle 결과가 아님\n'
      'A/B는 expert별 평가 표본군이 달라 자체 역량의 두 관점으로 사용하지 않음\n'
      '파란 숫자는 소수 셋째 자리 표시값이 Global보다 클 때만 강조\n'
      'ToN Scanning은 K=2/4/6/8의 모든 expert에서 test·validation confirm F1 개선\n'
      'CIC2018 Infiltration은 모든 K에서 test F1 개선 expert 없음\n'
      'CIC2018 Web attack은 일부 test 개선이 있으나 동일 expert의 confirm 개선 없음\n'
      '같은 클래스에서 test와 confirm의 개선 방향을 확인하며 두 split의 F1 절대값 차이를 직접 일반화 격차로 해석하지 않음\n'
      '단일 클래스의 개선으로 expert 전체가 Global보다 우수하다고 해석하지 않음\n'
      '근거: 각 dataset_k4/diagnostics/expert_class_metrics.csv 및 EXP63 capability report')

# 6. Actual S+V results across every K, no component ablation claims.
s = slide('S/V 적용 결과 · ToN 호출 활성화, CIC2018 호출 0건', 'S/V 결과')
subtitle(s,'Seed 43 · validation에서 호출·채택 기준 선택 · 전체 test 평가')
for ds,title,x in [('cic2018','CIC2018',.62),('toniot','ToN',6.81)]:
    d=cap['datasets'][ds]
    text(s,x,1.59,5.90,.40,title,size=19,bold=True,ink=NAVY,inset=0)
    g=d['banks']['4']['models']['full_test']['global']['summary']['macro_f1']
    rows=[['Global',f'{g:.4f}','0.00','0.00']]
    for k in cap['ks']:
        p=d['banks'][str(k)]['policy']
        rows.append([f'K = {k}',f"{p['macro_f1']:.4f}",f"{100*p['logical_expert_calls']/p['rows']:.2f}",f"{100*p['accepted']/p['rows']:.2f}"])
    sh=table(s,x,2.17,[1.20,1.66,1.52,1.52],['구성','Macro-F1','호출률 (%)','채택률 (%)'],rows,row_h=.52,font=13.5)
    highlight(sh,3,[0,1,2,3],ink=NAVY)
    box(s,x,5.53,5.90,.79,fill=CARD)
    if ds=='cic2018':
        msg='K = 2, 4, 6, 8 모두 Global 예측 유지'
    else:
        scan=systems[ds].loc['scanning']
        msg=f"K = 4 Scanning F1   {scan.global_f1:.4f} → {scan.system_f1:.4f}"
    text(s,x+.16,5.67,5.58,.47,msg,size=14,bold=True,ink=NAVY,inset=0)
text(s,.62,6.46,12.09,.27,'호출률: 정책상 expert 호출 표본 비율 · 채택률: expert 출력 수락 표본 비율 · 분모는 전체 test',size=11.1,ink=GREY,inset=0)
footer(h,s,'현재 context 구성에서 데이터셋별 S/V 활성화 차이 확인')
notes(s,'EXP63의 S1V0 설정 이름은 S와 V를 모두 학습한 현재 완전판이며 V 제거 ablation을 뜻하지 않음\n'
      'K=4는 기존 대표 설정이므로 다음 SOTA 표에도 사용, test 최고 K를 선택한 것이 아님\n'
      'CIC2018 모든 K: 선택된 정책에서 호출과 채택 모두 0\n'
      'ToN 모든 K: 정책상 모든 표본에 선택 expert 한 개 호출, 모든 K개 expert를 호출한다는 뜻은 아님\n'
      '호출과 verifier 채택은 다르며 채택된 출력이 Global과 같은 클래스일 수 있음\n'
      '실험은 모든 expert 예측을 미리 저장해 정책을 평가했으므로 호출 수는 논리적 호출 수\n'
      '현재 값으로 sparse online latency를 측정했다고 주장하지 않음\n'
      'CIC의 호출 부재를 V 하나의 원인으로 단정하지 않으며 이번 발표에서는 원인 분석을 확장하지 않음\n'
      '별도의 S 제거/V 제거 기여 분리 실험은 이번 표에 포함하지 않음\n'
      '근거: tabpfn/results/20260930_exp63_k_sweep_s43/*/summary.csv')

# 7. SOTA comparison with a clearly bounded representative current model.
methods=['global_raw','xgb','boostpfn','localpfn','distpfn','xgb_full']
labels=dict(global_raw='Global TabPFN v3',xgb='XGBoost',boostpfn='BoostPFN',localpfn='LoCalPFN · FT',distpfn='DistPFN · v3',xgb_full='XGBoost · 전체 train')
a=sota['datasets']['cic2018']['results'];b=sota['datasets']['toniot']['results']
s=slide('정제 데이터의 SOTA와 현재 모델 성능','SOTA 비교')
subtitle(s,'Seed 43 · 동일 정제 test · 대표 설정 K = 4 · Macro-F1')
rows=[]
for m in methods:
    pool='공통 100,000행' if m!='xgb_full' else '전체 train'
    rows.append([labels[m],f"{a[m]['macro_f1']:.4f}",f"{b[m]['macro_f1']:.4f}",pool])
rows.append(['현재 모델 · K = 4 · S+V',f"{cap['datasets']['cic2018']['banks']['4']['policy']['macro_f1']:.4f}",f"{cap['datasets']['toniot']['banks']['4']['policy']['macro_f1']:.4f}",'Global + Expert·Route 풀'])
sh=table(s,.62,1.71,[3.57,2.11,2.11,4.30],['방법','CIC2018','ToN','학습 정보'],rows,row_h=.52,font=15)
highlight(sh,5,[1]);highlight(sh,2,[2]);highlight(sh,7,[0,1,2,3],fill=HEAD,ink=NAVY)
text(s,.62,6.04,12.09,.31,'첫 5개 방법은 같은 100,000개 train ID 사용 · 전체 train XGB와 현재 모델은 추가 학습 데이터 사용',size=12.2,ink=GREY,inset=0)
text(s,.62,6.47,12.09,.29,'Global·DistPFN: v3 · BoostPFN·LoCalPFN: v1 · LoCalPFN은 미세조정 포함',size=11.8,ink=GREY,inset=0)
footer(h,s,'현재 모델은 anchor + residual block의 예비 결과 · 동일 학습 예산의 비교와 구분')
notes(s,'SOTA는 EXP61, 현재 모델은 EXP63 K=4 결과\n'
      '현재 모델 수치는 residual oracle 또는 expert별 test 최고값이 아니라 S/V 포함 최종 예측\n'
      '대표 설정 K=4를 사전에 사용했던 구성으로 유지, ToN test 최고 K=6을 골라 비교하지 않음\n'
      '첫 5개 방법만 Global C0의 정확히 같은 100,000 train ID 사용\n'
      '전체 train XGB는 CIC2018 8,666,430행, ToN 4,669,119행 사용\n'
      '현재 모델은 expert 구성과 S/V 학습을 위한 추가 라벨 데이터를 사용하므로 같은 100k 학습 예산의 SOTA 우위로 주장하지 않음\n'
      'Global/DistPFN은 v3, BoostPFN/LoCalPFN은 v1, backbone과 validation 사용도 동일하지 않음\n'
      '단일 seed이며 이미 관찰한 development holdout 결과\n'
      '근거: docs/research/20260930/exp61_report_data.json 및 exp63_capability_report_data.json')

# 8. Measured baseline cost, separated from logical current-policy calls.
s=slide('SOTA 측정 비용과 현재 모델의 호출','비용 비교')
subtitle(s,'RTX 4090 24GB · 최대 2개 작업 병렬 · 자원 경합을 포함한 경과 시간')
rows=[]
for m in methods:
    rows.append([labels[m],f"{a[m]['fit_seconds']:,.1f}",f"{a[m]['predict_seconds']:,.1f}",f"{b[m]['fit_seconds']:,.1f}",f"{b[m]['predict_seconds']:,.1f}",f"{a[m]['peak_gpu_sampled_gib']:.2f}",f"{b[m]['peak_gpu_sampled_gib']:.2f}"])
table(s,.62,1.71,[3.03,1.54,1.54,1.54,1.54,1.45,1.45],
      ['방법','CIC 학습\n초','CIC 추론\n초','ToN 학습\n초','ToN 추론\n초','CIC GPU\nGiB','ToN GPU\nGiB'],rows,row_h=.53,font=12.1)
text(s,.62,5.58,12.09,.33,'LoCalPFN: 학습에 validation, 추론에 검색 포함 · DistPFN: Global 계산과 prior 보정 포함',size=12.0,ink=GREY,inset=0)
box(s,.62,6.10,12.09,.57,fill=HEAD)
text(s,.80,6.19,11.73,.37,'현재 모델 K = 4 · 정책상 호출률 CIC2018 0.00%, ToN 100.00% · 조건부 추론 시간 별도 측정 예정',size=12.5,bold=True,ink=NAVY,inset=0)
notes(s,'EXP61 측정 비용은 기존 SOTA 표의 실제 fit/predict/GPU 값을 그대로 사용\n'
      '단독 실행 latency 비교나 고유 속도 배수로 해석하지 않음\n'
      'GPU는 1초 간격 프로세스별 최대 사용량\n'
      '현재 모델은 offline으로 모든 expert 예측을 준비한 뒤 정책을 평가했으므로 실제 조건부 호출 비용은 미측정\n'
      '재사용 집계 작업의 3초 등은 모델의 학습·추론 시간으로 사용하지 않음\n'
      'ToN 100% 호출은 선택 expert 한 개를 호출하는 정책이며 4개 전부 호출이 아님\n'
      '근거: docs/research/20260930/exp61_report_data.json')

# 9. Future work 1/2, no new experiment launch.
s=slide('후속 접근 · Benign 다양성과 tail 합성','후속 실험')
subtitle(s,'현재 기준: anchor + residual block · 제한된 context에 담을 정보의 확장')
card(s,.62,1.64,5.90,4.99,'1  Benign을 다양하게 구성')
card(s,6.81,1.64,5.90,4.99,'2  LITO 방식의 tail 합성')
text(s,.87,2.39,5.40,.73,'프로토콜·통신량·시간 특성이 다른\n정상 트래픽 패턴을 context에 포함',size=18,bold=True,ink=NAVY,inset=0)
for j,(label,sub) in enumerate([('프로토콜','TCP · UDP 등'),('통신량','패킷 수 · 크기'),('시간 특성','지속시간 · 간격')]):
    x=.90+j*1.80
    box(s,x,3.43,1.59,.88,fill='FFFFFF',border=LINE)
    text(s,x+.07,3.54,1.45,.25,label,size=13,bold=True,ink=NAVY,align=PP_ALIGN.CENTER,inset=0)
    text(s,x+.04,3.92,1.51,.23,sub,size=10.7,ink=GREY,align=PP_ALIGN.CENTER,inset=0)
text(s,.87,4.66,5.40,.91,'클래스별 할당량과 context 크기 고정\n유사한 정상 사례의 반복 비중 축소\n다양한 실측 사례를 선택하는 효과 확인',size=15.4,ink=INK,inset=0)
text(s,.87,6.05,5.40,.30,'훈련 데이터에서 구성 · 후속 실험 예정',size=12.5,ink=GREY,inset=0)
text(s,7.06,2.39,5.40,.73,'Feature 일부를 가린 뒤 tail 조건으로 생성\n생성 결과의 클래스 일관성과 유효성 확인',size=17,bold=True,ink=NAVY,inset=0)
for j,label in enumerate(['Feature 마스킹','조건부 생성','생성 결과 검증']):
    x=7.08+j*1.80
    box(s,x,3.43,1.57,.88,fill='FFFFFF',border=LINE)
    text(s,x+.06,3.62,1.45,.40,label,size=12.5,bold=True,ink=NAVY,align=PP_ALIGN.CENTER,inset=0)
    if j<2:line(h,s,x+1.57,3.87,x+1.80,3.87,head=True)
text(s,7.06,4.66,5.40,.89,'네트워크 feature 간 관계를 반영한 합성\n같은 tail 할당량에서 실측 반복과 비교\n실측 validation에서 분류 성능 확인',size=15.4,ink=INK,inset=0)
reference(s,7.06,5.82,5.40,.63,'Yang 외 · Language-Interfaced Tabular Oversampling\nvia Progressive Imputation and Self Authentication · ICLR 2024',
          'https://proceedings.iclr.cc/paper_files/paper/2024/file/5d54d2df6ec8f7b920aa0fec9a6d1b2e-Paper-Conference.pdf')
notes(s,'이 장의 두 접근은 계획이며 이번 수정에서 실행하지 않음\n'
      '현재 anchor+residual은 context 설계의 한 가지 시도이며 최종 구조로 확정하지 않음\n'
      'Benign 다양성: 같은 크기/클래스 수 조건에서 선택 방식의 효과 확인\n'
      'LITO는 majority feature를 중요도에 따라 masking하고 minority 조건으로 imputation, self-authentication 수행\n'
      '우리 IDS 적용은 네트워크 feature 제약을 반영하는 응용이며 아직 검증되지 않음\n'
      '생성 모델의 자기 검증은 실제 공격 라벨의 정당성을 보장하지 않음\n'
      '동일 tail 할당량에서 실측 반복 대비 합성 효과를 구분\n'
      'Reference: June Yong Yang, Geondo Park, Joowon Kim, Hyeongwon Jang, Eunho Yang. Language-Interfaced Tabular Oversampling via Progressive Imputation and Self Authentication. ICLR 2024.\n'
      'https://proceedings.iclr.cc/paper_files/paper/2024/file/5d54d2df6ec8f7b920aa0fec9a6d1b2e-Paper-Conference.pdf')

# 10. Future work 3/4, with the user-specified references retained.
s=slide('후속 접근 · Feature 표현 학습과 context 최적화','후속 실험')
subtitle(s,'훈련 데이터로 사전 학습 · 새 입력의 정답 없이 적용 가능한 구성')
card(s,.62,1.64,5.90,4.99,'3  Feature 학습으로 embedding 추가')
card(s,6.81,1.64,5.90,4.99,'4  Context 자체의 최적화')
text(s,.87,2.39,5.40,.73,'정상·공격 구분에 유용한 표현을 학습\n각 행의 embedding을 기존 feature에 추가',size=17,bold=True,ink=NAVY,inset=0)
for j,label in enumerate(['Flow 한 행','학습한 encoder','행별 embedding']):
    x=.90+j*1.80
    box(s,x,3.43,1.57,.88,fill='FFFFFF',border=LINE)
    text(s,x+.06,3.62,1.45,.40,label,size=12.5,bold=True,ink=NAVY,align=PP_ALIGN.CENTER,inset=0)
    if j<2:line(h,s,x+1.57,3.87,x+1.80,3.87,head=True)
text(s,.87,4.64,5.40,.81,'학습 시 샘플·클래스 관계 활용\nContext와 query에 동일한 변환 적용\n원래 feature만 사용한 경우와 비교',size=15.4,ink=INK,inset=0)
reference(s,.87,5.70,5.40,.76,'Lopez-Martin 외 · Supervised contrastive learning over\nprototype-label embeddings for network intrusion detection\nInformation Fusion 2022 · 공개 코드',
          'https://github.com/mlopezm/Supervised-contrastive-learning-over-prototype-label-embeddings')
text(s,7.06,2.39,5.40,.73,'TabPFN을 고정하고 context 값을 학습\n작은 context로 훈련 풀의 정보를 압축',size=17,bold=True,ink=NAVY,inset=0)
for j,label in enumerate(['학습할 context','고정 TabPFN','훈련 query 손실']):
    x=7.08+j*1.80
    box(s,x,3.43,1.57,.88,fill='FFFFFF',border=LINE)
    text(s,x+.06,3.62,1.45,.40,label,size=12.5,bold=True,ink=NAVY,align=PP_ALIGN.CENTER,inset=0)
    if j<2:line(h,s,x+1.57,3.87,x+1.80,3.87,head=True)
line(h,s,11.99,4.31,11.99,4.49,ink=MUTED)
line(h,s,11.99,4.49,7.87,4.49,ink=MUTED)
line(h,s,7.87,4.49,7.87,4.31,ink=MUTED,head=True)
text(s,7.06,4.73,5.40,.72,'고정된 context 행 수에서 최적화\n원본 표본 선택과 학습된 context 비교\n현재 v3의 gradient·메모리 조건 확인',size=15.4,ink=INK,inset=0)
reference(s,7.06,5.70,5.40,.55,'Ma 외 · In-Context Data Distillation with TabPFN\nICLR 2024 ME-FoMo Workshop', 'https://arxiv.org/abs/2402.06971')
reference(s,7.06,6.28,5.40,.24,'Valentin Thomas 연구 페이지', 'https://valthom.github.io/')
notes(s,'두 접근 모두 후속 실험 예정이며 현재 성능 표에 포함하지 않음\n'
      '3번은 여러 행을 하나의 embedding으로 묶는 방법이 아님, 훈련 관계를 이용해 encoder를 학습하고 적용 시 각 행에서 embedding 산출\n'
      '인용 논문은 feature와 class label prototype을 같은 공간에 표현, TabPFN에 concatenation하는 부분은 우리 적용 제안\n'
      'Manuel Lopez-Martin, Antonio Sanchez-Esguevillas, Juan Ignacio Arribas, Belen Carro. Supervised contrastive learning over prototype-label embeddings for network intrusion detection. Information Fusion 79 (2022), 200–228. DOI: 10.1016/j.inffus.2021.09.014\n'
      'https://github.com/mlopezm/Supervised-contrastive-learning-over-prototype-label-embeddings\n'
      '4번은 고정 PFN에 대해 작은 context를 최적화하는 ICD이며 기존 posterior 보정 DistPFN과 다른 방법\n'
      'Junwei Ma, Valentin Thomas, Guangwei Yu, Anthony Caterini. In-Context Data Distillation with TabPFN. arXiv:2402.06971, 2024. ICLR 2024 ME-FoMo Workshop 발표는 저자 페이지에서 확인\n'
      'https://arxiv.org/abs/2402.06971\nhttps://valthom.github.io/\n'
      '현재 v3에서의 구현 호환성과 비용을 확인한 뒤 검토, 배포에서 새 입력의 정답은 불필요\n'
      '자료 정리: docs/research/20261001/context_followup_approaches.md')

assert cursor == 10
prs.core_properties.title='10월 2일 랩미팅 · 한국어'
prs.core_properties.subject='데이터 검토, anchor와 residual context, expert 및 S/V 결과, SOTA, 후속 context 접근'
prs.core_properties.comments='2026-10-01 EXP61/63 반영 · 한국어만 갱신 · 후속 네 접근은 계획'
prs.save(OUTPUT)

# Inspect the saved result, including right alignment and geometry.
check=Presentation(OUTPUT)
issues=[];dump=[];numeric_cells=0
for i,s in enumerate(check.slides,1):
    entries=[]
    text_shapes=[]
    for sh in s.shapes:
        if sh.left<0 or sh.top<0 or sh.left+sh.width>check.slide_width+15 or sh.top+sh.height>check.slide_height+15:
            issues.append([i,'outside slide',sh.name])
        if sh.top<Inches(6.90) and sh.top+sh.height>Inches(6.90):
            issues.append([i,'footer collision',sh.name])
        if sh.has_text_frame and sh.text.strip():
            entries.append(sh.text)
            text_shapes.append(sh)
        if sh.has_table:
            entries.extend(' | '.join(c.text for c in r.cells) for r in sh.table.rows)
            for r,row in enumerate(sh.table.rows):
                for c,cell in enumerate(row.cells):
                    if r>0 and c>0 and NUM.fullmatch(cell.text):
                        numeric_cells+=1
                        assert all(p.alignment==PP_ALIGN.RIGHT for p in cell.text_frame.paragraphs),(i,r,c,cell.text)
            for c in range(1,len(sh.table.columns)):
                vals=[sh.table.cell(r,c).text for r in range(1,len(sh.table.rows))]
                decimals={len(v.rstrip('%').split('.')[1]) for v in vals if NUM.fullmatch(v) and '.' in v}
                assert len(decimals)<=1,(i,c,decimals)
        if sh.has_text_frame and sh.text.strip():
            assert sh.text_frame.auto_size==MSO_AUTO_SIZE.NONE
    # Text on card backgrounds and connectors is intentional. Text-text overlaps are not.
    for j,a0 in enumerate(text_shapes):
        for b0 in text_shapes[j+1:]:
            dx=min(a0.left+a0.width,b0.left+b0.width)-max(a0.left,b0.left)
            dy=min(a0.top+a0.height,b0.top+b0.height)-max(a0.top,b0.top)
            if dx>Inches(.005) and dy>Inches(.005):
                issues.append([i,'text box overlap',a0.text[:45],b0.text[:45]])
    joined='\n'.join(entries)
    for forbidden in ['TBD','상대사례','입력유사군','관점 A','관점 B']:
        assert forbidden not in joined,(i,forbidden)
    assert not re.search(r'합니다|입니다|습니다|[—–]',joined),(i,'style')
    dump.append(f'## {i}\n{joined}\n')
assert not issues,issues
qa=QA;qa.mkdir(exist_ok=True)
(qa/'slide_text.md').write_text('\n'.join(dump))
english=read_record(HERE/'archive/20261001_before_current_results/english_untouched_sha256.json')
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in english.items())
validation=dict(slides=10,output=str(OUTPUT.relative_to(ROOT)),updated='2026-10-01',
                source='existing edited Korean PPTX; source experiment CSVs and JSON',
                numeric_cells_right_aligned=numeric_cells,consistent_decimal_places=True,
                geometry_and_style_issues=issues,english_unchanged=True,
                source_checks=checks,render='pending PDF render and visual inspection',
                input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'exp61_report_data.json',DATA/'exp63_capability_report_data.json']})
(qa/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print(OUTPUT)
print(f'{numeric_cells} numeric cells verified; 10 slides; English unchanged')
