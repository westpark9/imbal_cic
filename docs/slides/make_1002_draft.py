#!/usr/bin/env python3
"""Korean lab-meeting deck. --fill-intro updates slides 1–4 in the edited PPTX."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

from pathlib import Path
import json
import re
import sys
import types

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'kor/0904_labmeeting_tabpfn5.pptx'
OUTPUT = HERE / 'kor/1002_labmeeting_ko_draft.pptx'
NAVY, BLUE, CYAN = '00306C', '007BC6', '00B8EE'
CARD, HEAD, LINE = 'F3F6FA', 'E7F0F8', 'D9E2EC'
INK, GREY, MUTED = '21262E', '5D6773', '8393A5'
FONT = 'Pretendard'
FILL_INTRO = '--fill-intro' in sys.argv
FILL_SOTA = '--fill-sota' in sys.argv
assert not (FILL_INTRO and FILL_SOTA)
EDITING = FILL_INTRO or FILL_SOTA
prs = Presentation(OUTPUT if EDITING else SOURCE)
UNCHANGED_SLIDES = [etree.tostring(s._element) for s in prs.slides][4:] if FILL_INTRO else []
PRESERVED_INTRO = [etree.tostring(s._element) for s in prs.slides][:4] if FILL_SOTA else []
_intro_cursor = 0
if not EDITING:
    for item in list(prs.slides._sldIdLst):
        prs.part.drop_rel(item.rId)
        prs.slides._sldIdLst.remove(item)
layout = next(l for l in prs.slide_layouts if l.name == 'TITLE_ONLY')


def color(hexval):
    return RGBColor.from_string(hexval)


def set_text(frame, value, size=16, bold=False, ink=INK, align=PP_ALIGN.LEFT,
             middle=True, inset=0.12):
    frame.clear()
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = Inches(inset)
    frame.margin_top = frame.margin_bottom = Inches(0.04)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE if middle else MSO_ANCHOR.TOP
    for i, line in enumerate(value.split('\n')):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        p.space_before = p.space_after = Pt(0)
        p.line_spacing = 1.12
        pp = p._p.get_or_add_pPr()
        etree.SubElement(pp, qn('a:buNone'))
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color(ink)
        rp = r._r.get_or_add_rPr()
        rp.set('lang', 'ko-KR')
        for name in ['a:ea', 'a:cs']:
            etree.SubElement(rp, qn(name)).set('typeface', FONT)


def text(slide, x, y, w, h, value, **kwargs):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    set_text(s.text_frame, value, **kwargs)
    return s


def box(slide, x, y, w, h, fill=CARD, border=None, rounded=True):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                              Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        s.adjustments[0] = 0.035
    s.fill.solid()
    s.fill.fore_color.rgb = color(fill)
    s.shadow.inherit = False
    if border:
        s.line.color.rgb = color(border)
        s.line.width = Pt(0.8)
    else:
        s.line.fill.background()
    return s


def panel(slide, x, y, w, h, label, tbd=True):
    box(slide, x, y, w, h)
    box(slide, x, y, w, 0.48, fill=HEAD, rounded=False)
    text(slide, x+0.10, y+0.03, w-0.20, 0.41, label, size=15, bold=True, ink=NAVY)
    if tbd:
        text(slide, x+0.18, y+0.61, w-0.36, h-0.77, 'TBD', size=27,
             ink=MUTED, align=PP_ALIGN.CENTER)


def takeaway(slide, label='해석'):
    box(slide, 0.62, 6.02, 12.09, 0.62, fill='FFFFFF', border=LINE)
    text(slide, 0.78, 6.10, 1.5, 0.40, label, size=14, bold=True, ink=NAVY)
    text(slide, 2.30, 6.10, 10.14, 0.40, 'TBD', size=16, ink=MUTED)


def table(slide, x, y, widths, headers, rows, row_h=0.55, font=14):
    sh = slide.shapes.add_table(len(rows)+1, len(headers), Inches(x), Inches(y),
                               Inches(sum(widths)), Inches(row_h*(len(rows)+1)))
    t = sh.table
    t.first_row = False
    t.horz_banding = False
    for i,w in enumerate(widths):
        t.columns[i].width = Inches(w)
    for r,values in enumerate([headers]+rows):
        t.rows[r].height = Inches(row_h)
        for c,value in enumerate(values):
            cell = t.cell(r,c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = color(NAVY if r == 0 else ('FFFFFF' if r%2 else CARD))
            set_text(cell.text_frame, value, size=font, bold=r==0 or c==0,
                     ink='FFFFFF' if r==0 else (MUTED if value=='TBD' else INK),
                     align=PP_ALIGN.LEFT if c==0 else PP_ALIGN.CENTER)
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.06)
    return sh


def slide(title, section):
    global _intro_cursor
    if EDITING:
        index = [4,5,8][_intro_cursor] if FILL_SOTA else _intro_cursor
        s = prs.slides[index]
        _intro_cursor += 1
        page = index+1
    else:
        s = prs.slides.add_slide(layout)
        page = len(prs.slides)
    for sh in list(s.shapes):
        s.shapes._spTree.remove(sh._element)
    text(s, 0.62, 0.30, 10.85, 0.65, title, size=28, bold=True, ink=NAVY, inset=0)
    if not EDITING:
        tag = box(s, 11.68, 0.44, 1.03, 0.31, fill=HEAD)
        set_text(tag.text_frame, '초안 · TBD', size=10, bold=True, ink=BLUE,
                 align=PP_ALIGN.CENTER, inset=0.02)
    box(s, 0, 6.95, 13.333333, 0.55, fill=NAVY, rounded=False)
    text(s, 9.05, 7.03, 3.00, 0.30, section, size=10, ink='D6E4F3', align=PP_ALIGN.RIGHT)
    text(s, 12.23, 7.01, 0.48, 0.34, str(page), size=11, bold=True,
         ink='FFFFFF', align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = '본문·수치·그림·해석 TBD\n'+title
    return s


if FILL_SOTA:
    from sota_1002 import build
    build(types.SimpleNamespace(**globals()))
    assert _intro_cursor == 3
    assert [etree.tostring(s._element) for s in prs.slides][:4] == PRESERVED_INTRO
    notes=prs.slides[3].notes_slide.notes_text_frame
    notes.text=notes.text.replace('EXP59/60에서 공통으로 고정된','EXP59와 EXP61에서 확인한').replace('상대사례 추가 전 EXP59 context','EXP59의 기존 residual context')
elif FILL_INTRO:
    from intro_1002 import build
    build(types.SimpleNamespace(**globals()))
    assert _intro_cursor == 4
    assert [etree.tostring(s._element) for s in prs.slides][4:] == UNCHANGED_SLIDES
else:
    s = slide('동일 입력의 상충 라벨', '데이터 검토')
    panel(s, 0.62, 1.40, 5.00, 4.28, '동일 feature · 서로 다른 라벨')
    text(s, 5.95, 1.42, 6.76, 0.45, '데이터셋별 상충 라벨 현황', size=17, bold=True, ink=NAVY)
    table(s, 5.95, 2.06, [2.06,1.67,1.73,1.30], ['데이터셋','전체 행','상충 행','비율'],
          [[d,'TBD','TBD','TBD'] for d in ['CIC2018','ToN-IoT','BoT-IoT','UNSW-NB15']], row_h=0.66, font=13)
    takeaway(s)

    s = slide('정제 이후 클래스 분포', '데이터 검토')
    panel(s, 0.62, 1.40, 5.90, 4.28, 'CIC2018 · 제거 전후 클래스 분포')
    panel(s, 6.81, 1.40, 5.90, 4.28, 'ToN-IoT · 제거 전후 클래스 분포')
    takeaway(s, '제거 기준')

    s = slide('Global과 residual expert 구조', '모델 소개')
    text(s, 0.62, 1.25, 12.09, 0.35, '학습 구성', size=15, bold=True, ink=GREY)
    for x,label in [(0.62,'Global 학습 풀'),(4.73,'Expert 학습 풀'),(8.84,'Route 학습 풀')]:
        panel(s, x, 1.82, 3.87, 1.58, label)
    text(s, 0.62, 3.75, 12.09, 0.35, '추론 구조', size=15, bold=True, ink=GREY)
    panel(s, 0.62, 4.27, 12.09, 1.55, 'Global · Scorer · Expert · Verifier')
    takeaway(s, '모델 설명')

    s = slide('Global의 클래스별 성능과 expert의 병목', '모델 소개')
    panel(s, 0.62, 1.40, 5.90, 3.47, 'CIC2018 · Global 클래스별 F1')
    panel(s, 6.81, 1.40, 5.90, 3.47, 'ToN-IoT · Global 클래스별 F1')
    box(s, 0.62, 5.13, 12.09, 0.56, fill=HEAD)
    text(s, 0.78, 5.20, 2.60, 0.38, 'Expert · S/V 관찰', size=14, bold=True, ink=NAVY)
    text(s, 3.50, 5.20, 8.93, 0.38, 'TBD', size=16, ink=MUTED)
    takeaway(s, '개선 방향')

    s = slide('정제 데이터의 SOTA 분류 성능', 'SOTA 비교')
    table(s,.62,1.45,[4.29,3.90,3.90],['방법','CIC2018 Macro-F1','ToN Macro-F1'],
          [[name,'TBD','TBD'] for name in ['Global TabPFN','XGBoost','BoostPFN','LoCalPFN','DistPFN','XGBoost 전체 train']],row_h=.55)
    takeaway(s)

    s = slide('SOTA 학습·추론 비용', 'SOTA 비교')
    table(s, 0.62, 1.45, [4.29,3.90,3.90], ['조건','CIC2018 · Macro-F1','ToN-IoT · Macro-F1'],
          [[name,'TBD','TBD'] for name in ['학습 시간','전체 추론 시간','GPU 메모리','RAM']], row_h=0.57)
    panel(s, 0.62, 4.60, 12.09, 1.15, '측정 조건')
    takeaway(s)

    s = slide('담당 영역과 입력 유사군의 분류 능력', 'Expert 평가')
    panel(s, 0.62, 1.40, 5.90, 4.28, '관점 A · 담당 residual 영역', tbd=False)
    panel(s, 6.81, 1.40, 5.90, 4.28, '관점 B · 입력 유사군', tbd=False)
    for x in [0.62,6.81]:
        text(s, x+0.18, 2.04, 5.54, 0.40, '대표 expert · 클래스: TBD', size=14, ink=GREY)
        table(s, x+0.18, 2.65, [1.94,1.80,1.80], ['지표','Global','Expert'],
              [[m,'TBD','TBD'] for m in ['Precision','Recall','F1']], row_h=0.50, font=13)
        text(s, x+0.18, 4.92, 2.65, 0.41, '교정: TBD', size=15, ink=GREY)
        text(s, x+3.03, 4.92, 2.69, 0.41, '훼손: TBD', size=15, ink=GREY)
    takeaway(s)

    s = slide('Scorer와 verifier의 기여', '구성요소 비교')
    table(s, 0.62, 1.45, [3.49,2.00,2.00,2.30,2.30], ['구성','CIC2018','ToN-IoT','호출 · 채택','교정 · 훼손'],
          [[name,'TBD','TBD','TBD','TBD'] for name in ['Global','S·V 없음','S만','V만','S+V']], row_h=0.58, font=14)
    box(s, 0.62, 5.20, 12.09, 0.49, fill=HEAD)
    text(s, 0.78, 5.26, 3.00, 0.35, 'Context · gate 설정', size=14, bold=True, ink=NAVY)
    text(s, 3.90, 5.26, 8.55, 0.35, 'TBD', size=16, ink=MUTED)
    takeaway(s)

    s = slide('동일 정제 조건의 기존 방법 비교', '기존 방법 비교')
    table(s, 0.62, 1.45, [3.29,2.20,2.20,2.20,2.20], ['방법','CIC2018','ToN-IoT','학습 표본','추론 비용'],
          [[name,'TBD','TBD','TBD','TBD'] for name in ['Global TabPFN','XGBoost','BoostPFN','DistPFN','제안 모델']], row_h=0.58, font=14)
    box(s, 0.62, 5.20, 12.09, 0.49, fill=HEAD)
    text(s, 0.78, 5.26, 3.00, 0.35, '공통 평가 조건', size=14, bold=True, ink=NAVY)
    text(s, 3.90, 5.26, 8.55, 0.35, 'TBD', size=16, ink=MUTED)
    takeaway(s)

    s = slide('확인된 효과와 다음 개선', '결론')
    for x,label in [(0.62,'Context 구성'),(4.73,'Expert 구분 능력'),(8.84,'최종 모델 성능')]:
        panel(s, x, 1.40, 3.87, 2.16, label)
    panel(s, 0.62, 3.87, 12.09, 1.82, '다음 개선')
    takeaway(s, '발표 결론')

prs.core_properties.title = '10월 2일 랩미팅 · 한국어 초안'
prs.core_properties.subject = '데이터 검토, context 구성, expert 역량, S/V, 기존 방법 비교'
prs.core_properties.comments = '1~6장 작성, 7~10장 후속 모델 실험 TBD' if FILL_SOTA else ('1~4장 작성, 후속 장 편집본 보존' if FILL_INTRO else '본문과 수치 TBD 초안')
prs.save(OUTPUT)

# Inspect the generated artifact, including text on each slide and geometry.
check = Presentation(OUTPUT)
assert len(check.slides) == 10
issues, dump = [], []
for i,s in enumerate(check.slides,1):
    entries = []
    for sh in s.shapes:
        if sh.left < 0 or sh.top < 0 or sh.left+sh.width > check.slide_width+10 or sh.top+sh.height > check.slide_height+10:
            issues.append([i,'outside slide',sh.name])
        if sh.top < Inches(6.90) and sh.top+sh.height > Inches(6.90):
            issues.append([i,'footer collision',sh.name])
        if sh.has_text_frame:
            entries.append(sh.text_frame.text)
        if sh.has_table:
            entries.extend(' | '.join(c.text for c in r.cells) for r in sh.table.rows)
    joined='\n'.join(entries)
    if (EDITING and i<=4) or (FILL_SOTA and i<=6):assert 'TBD' not in joined
    elif not FILL_INTRO:assert 'TBD' in joined
    assert '상대사례' not in joined and '상대 클래스 사례' not in joined
    if re.search(r'합니다|입니다|습니다|[—–]',joined):
        issues.append([i,'style',joined])
    dump.append(f'## {i}\n{joined}\n')
assert not issues,issues
qa=HERE/'kor/1002_draft_qa'
qa.mkdir(exist_ok=True)
(qa/'slide_text.md').write_text('\n'.join(dump))
(qa/'validation.json').write_text(json.dumps(dict(slides=10,source=str(OUTPUT if EDITING else SOURCE),output=str(OUTPUT),
     all_content_tbd=not EDITING,filled_slides=list(range(1,7)) if FILL_SOTA else ([1,2,3,4] if FILL_INTRO else []),
     preserved_slide_xml=list(range(5,11)) if FILL_INTRO else ([1,2,3,4,7,8,10] if FILL_SOTA else []),geometry_and_style_issues=issues),ensure_ascii=False,indent=2)+'\n')
print(OUTPUT)
