#!/usr/bin/env python3
"""Revise the introduction; preserve the supplied PDF's experiment pages.

One scene description produces editable PPTX shapes and browser-rendered PDF.
The supplied experiment pages are images in PPTX, original PDF pages in PDF.
"""
from pathlib import Path
import hashlib
import html
import json
import subprocess
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_AUTO_SIZE, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.xmlchemy import OxmlElement

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
SLIDES = ROOT/'docs/slides'
QA = SLIDES/'_support/qa/20261020_meeting'
QA.mkdir(parents=True, exist_ok=True)
SOURCE = SLIDES/'eng/26S_Meeting.pdf'
ARCH = SLIDES/'_support/qa/kor/1015_labmeeting_qa/v3_architecture_page.png'
PAPER = 'https://arxiv.org/html/2605.13986v1'
W, H = 13.333333, 7.5
NAVY, BLUE, INK, GREY, PALE = '00306C', '007BC6', '20262E', '617080', 'F1F5F9'

def text(x,y,w,h,s,size=22,bold=False,color=INK):
    h=max(h,len(s.splitlines())*size*1.14/72*1.06)
    return dict(kind='text',x=x,y=y,w=w,h=h,text=s,size=size,bold=bold,color=color)

def box(x,y,w,h,fill=PALE):
    return dict(kind='box',x=x,y=y,w=w,h=h,fill=fill)

def arrow(x,y,w=.35,h=.28):
    return dict(kind='arrow',x=x,y=y,w=w,h=h,fill=BLUE)

def arch():
    return dict(kind='image',x=.62,y=1.08,w=3.95,h=5.88,path=str(ARCH),crop=[.20,.082,.19,.285])

def make_scenes(lang):
    def t(ko,en): return ko if lang=='ko' else en
    scenes=[]
    def page(title,items,notes):
        scenes.append(dict(title=title,notes=notes,items=[text(.60,.33,12.1,.57,title,30,True,NAVY)]+items))

    items=[text(.62,1.12,12.0,.52,t('IDS · 통신 기록의 특징을 이용한 정상·공격 유형 분류',
        'IDS · Classifying network flows as benign or attack types'),22)]
    cards=[
        (t('초기 운영','Initial deployment'),t('정상 + 알려진 공격의\n라벨 사례 확보','Labelled benign traffic\nand known attacks')),
        (t('새 공격 사례 확보','Newly labelled attack'),t('처음 접한 공격 유형의\n소수 라벨 사례 확보','A small labelled sample\nof a newly observed attack')),
        (t('탐지기 갱신','Detector update'),t('새 공격 반영 +\n기존 정상·공격 구분 유지','Recognize the new attack\nand preserve earlier quality')),
    ]
    for i,(title,body) in enumerate(cards):
        x=.62+i*4.22
        items += [box(x,1.98,3.65,1.75),text(x+.20,2.15,3.25,.42,title,22,True,BLUE),text(x+.20,2.80,3.26,.68,body,19)]
        if i<2: items.append(arrow(x+3.81,2.70))
    items += [text(.65,4.21,2.05,.40,'XGBoost',23,True),
        text(2.85,4.21,9.80,.46,t('확보한 라벨 사례로 누적 재학습하는 비교군','Comparator: refit on all labelled examples available so far'),21),
        text(.65,5.00,2.05,.40,'TabPFN',23,True,BLUE),
        text(2.85,5.00,9.80,.46,t('사전학습 가중치 고정, 라벨 사례를 context로 제공','Fixed pretrained weights; labelled examples supplied as context'),21),
        box(.62,6.12,12.05,.77),text(.84,6.31,11.60,.40,t('비교 기준 · 신규 공격 탐지 · 기존 탐지 품질 · 갱신 비용',
        'Comparison · New-attack detection · Retained quality · Update cost'),20,True,NAVY)]
    page(t('신규 공격 사례를 반영하는 IDS','Adapting IDS to newly labelled attacks'),items,
        'Problem before dataset. A labelled flow is a tabular network record with an assigned class. '
        'Labels are assumed available after investigation; their acquisition is not evaluated here. '
        'XGBoost full refitting is the chosen comparator, not a claim that all tree updates require retraining. '
        'Class arrival is not a claim of zero-label novelty detection or cross-network robustness.')

    items=[text(.62,1.05,12.0,.40,t('합성 표 과제로 예측 규칙 사전학습 · TabPFN-3, arXiv 2026',
        'Prediction rules pretrained on synthetic tabular tasks · TabPFN-3, arXiv 2026'),19,color=GREY),
        box(.62,1.87,3.23,1.00),text(.82,2.04,2.85,.30,'Context',21,True,BLUE),
        text(.82,2.48,2.85,.25,t('통신 특징 + 정답 라벨','Flow features + labels'),17),
        box(.62,3.13,3.23,1.00),text(.82,3.30,2.85,.30,'Query',21,True,BLUE),
        text(.82,3.74,2.85,.25,t('분류할 통신 기록','Flow features to classify'),17),
        arrow(4.07,2.37,.47,.32),arrow(4.07,3.62,.47,.32),
        box(4.80,2.27,3.55,1.46,fill=NAVY),text(5.00,2.54,3.15,.39,'TabPFN',26,True,'FFFFFF'),
        text(5.00,3.16,3.15,.29,t('사전학습 가중치 고정','Pretrained weights fixed'),18,color='FFFFFF'),
        arrow(8.57,2.93,.47,.32),box(9.30,2.27,3.38,1.46),
        text(9.50,2.57,2.97,.33,t('클래스별 확률','Class probabilities'),21,True,BLUE),
        text(9.50,3.17,2.97,.30,t('정상 · 공격 A · 공격 B …','Benign · Attack A · Attack B …'),16),
        text(.65,4.64,2.63,.35,'In-context learning',20,True,BLUE),
        text(3.45,4.64,9.05,.36,t('가중치 변경 없이, context에 조건화한 query 예측',
        'Predict the query using its context, with no weight optimization'),19),
        text(.65,5.35,2.40,.35,'Fine-tuning',20,True,BLUE),
        text(3.45,5.35,9.05,.57,t('IDS 사례를 context/query로 나누고\nquery 정답의 손실을 역전파하여 가중치 갱신',
        'Split IDS examples into context/query; update weights\nby backpropagating the labelled-query loss'),19),
        box(.62,6.29,12.05,.61),text(.84,6.44,11.61,.31,t('현재 비교 · 초기 IDS fine-tuning 없이 사전학습 가중치 유지',
        'Current comparison · Original pretrained weights, with no IDS fine-tuning'),19,True,NAVY)]
    page(t('TabPFN · 가중치, context, query','TabPFN · Weights, context and query'),items,
        PAPER+' Sections 1, 2.2, 2.5. Synthetic tasks refer to model pretraining, not the IDS evaluation data. '
        'Context is a supplied labelled dataset, not information stored inside fixed pretrained weights. '
        'Fine-tuning is optional weight optimization and is not used in the supplied result pages. '
        'FT interface: https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_base.py')

    items=[arch()]
    rows=[
        ('1',t('입력값 표현 · Cell embedding','Encode values · Cell embedding'),t('패킷 길이 등 각 값을 벡터로 변환\n결측 여부도 함께 입력',
            'Convert each value, such as packet length, into a vector\nInclude an indicator for missing values')),
        ('2',t('열 정보 요약 · Feature embedding','Summarize columns · Feature embedding'),t('여러 통신 기록에서 각 특징의 분포 파악\ncontext의 라벨도 함께 참조',
            'Summarize each feature across multiple flows\nUse the labels supplied with the context')),
        ('3',t('행 정보 통합 · Feature aggregation','Combine features · Feature aggregation'),t('한 통신 기록의 특징들을 하나의 표현으로 통합\n다른 기록과의 관계를 비교할 준비',
            'Combine a flow’s features into one representation\nPrepare it for comparison with other flows')),
    ]
    for i,(num,title,body) in enumerate(rows):
        y=1.40+i*1.74
        items += [text(4.98,y,7.73,.45,num+'  '+title,22,True,BLUE),text(5.33,y+.68,7.25,.87,body,20)]
    page(t('v3 구조 · 통신 기록의 표현','v3 architecture · Representing network flows'),items,
        PAPER+' Figure 5, Section 2.1. Original figure on the left, without a visible caption per user instruction. '
        'Steps 1–3 follow cell embedding, column-wise feature embedding, and row-wise feature aggregation. '
        'The labelled context is external input; the diagram does not depict IDS weight fine-tuning.')

    items=[arch(),
        text(4.98,1.36,7.73,.45,t('4  사례 참조 · In-context learning','4  Consult examples · In-context learning'),22,True,BLUE),
        text(5.33,2.03,7.23,.78,t('query와 context 사례의 관계를 계산\n예측에 참고할 사례를 attention으로 선택',
        'Compare the query with labelled context examples\nWeight their relevance through attention'),20),
        text(4.98,3.02,7.73,.45,t('5  확률 출력 · Many-class decoder','5  Predict probabilities · Many-class decoder'),22,True,BLUE),
        text(5.33,3.69,7.23,.78,t('context 라벨의 가중합으로 클래스 확률 계산\n기존·신규 클래스의 라벨 사례 모두 필요',
        'Combine context labels with attention weights\nKeep examples of both existing and new classes'),20),
        box(4.98,5.11,7.70,1.57),text(5.20,5.31,7.25,.35,t('Context 갱신에도 계산 비용 발생','Context updates also require computation'),21,True,NAVY),
        text(5.20,5.96,7.23,.48,t('전처리 · 표현 계산 · 캐시 준비\n가중치 학습과 구별하여 갱신 시간 측정',
        'Preprocessing · Representations · Cache preparation\nMeasure update time separately from weight training'),18)]
    page(t('v3 구조 · 사례 참조와 갱신','v3 architecture · Context-based prediction'),items,
        PAPER+' Sections 2.2 and 2.4.2. The decoder averages one-hot context labels with learned attention weights. '
        'Cached prediction can reuse prepared context states; changing context may require recomputation. '
        'No claim that a context update is free, or universally faster than XGBoost refitting.')
    return scenes

def write_deck(lang,scenes,include_experiments):
    prs=Presentation();prs.slide_width=Inches(W);prs.slide_height=Inches(H)
    geometry=[]
    for page_no,scene in enumerate(scenes,1):
        slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.notes_slide.notes_text_frame.text=scene['notes']
        for item in scene['items']:
            x,y,w,h=[item[k] for k in ('x','y','w','h')]
            assert x>=0 and y>=0 and x+w<=W+.01 and y+h<=H+.01,item
            geometry.append(dict(page=page_no,**item))
            if item['kind']=='text':
                shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));frame=shape.text_frame
                frame.clear();frame.word_wrap=False;frame.auto_size=MSO_AUTO_SIZE.NONE;frame.vertical_anchor=MSO_ANCHOR.TOP
                frame.margin_left=frame.margin_right=frame.margin_top=frame.margin_bottom=0
                for i,line in enumerate(item['text'].split('\n')):
                    p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
                    p.space_before=p.space_after=Pt(0);p.line_spacing=1.14
                    r=p.add_run();r.text=line;r.font.name='Pretendard';r.font.size=Pt(item['size']);r.font.bold=item['bold'];r.font.color.rgb=RGBColor.from_string(item['color'])
                    properties=r._r.get_or_add_rPr();properties.set('lang','ko-KR' if lang=='ko' else 'en-US')
                    for tag in ('a:ea','a:cs'):
                        font_element=OxmlElement(tag);font_element.set('typeface','Pretendard');properties.append(font_element)
            elif item['kind']=='image':
                shape=slide.shapes.add_picture(item['path'],Inches(x),Inches(y),width=Inches(w),height=Inches(h))
                shape.crop_left,shape.crop_top,shape.crop_right,shape.crop_bottom=item['crop']
            else:
                shape=slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW if item['kind']=='arrow' else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
                shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(item['fill']);shape.line.fill.background()
    if include_experiments:
        for n in (4,5,6):
            slide=prs.slides.add_slide(prs.slide_layouts[6])
            slide.shapes.add_picture(str(QA/f'original-{n}.png'),0,0,width=prs.slide_width,height=prs.slide_height)
            slide.notes_slide.notes_text_frame.text=f'Unchanged supplied 26S_Meeting.pdf page {n}. PDF source only: original content is a page image in PPTX. The companion PDF preserves the original vector page. Experiment redesign is separately documented and has not been applied here.'
    stem='20261020_Meeting_en' if lang=='en' else '20261020_Meeting_intro_ko'
    dest=SLIDES/('eng' if lang=='en' else 'kor')/stem
    prs.save(dest.with_suffix('.pptx'))
    body=[]
    for scene in scenes:
        tags=[]
        for i in scene['items']:
            style=f'left:{i["x"]}in;top:{i["y"]}in;width:{i["w"]}in;height:{i["h"]}in;'
            if i['kind']=='text':
                style+=f'font-size:{i["size"]}pt;font-weight:{700 if i["bold"] else 400};color:#{i["color"]};'
                tags.append(f'<div class="text" style="{style}">{html.escape(i["text"])}</div>')
            elif i['kind']=='image':
                l,t,r,b=i['crop'];iw=i['w']/(1-l-r);ih=i['h']/(1-t-b)
                tags.append(f'<div class="clip" style="{style}"><img src="{Path(i["path"]).as_uri()}" style="position:absolute;width:{iw}in;height:{ih}in;left:{-l*iw}in;top:{-t*ih}in"></div>')
            else:
                style+=f'background:#{i["fill"]};'
                if i['kind']=='arrow':style+='clip-path:polygon(0 25%,60% 25%,60% 0,100% 50%,60% 100%,60% 75%,0 75%);'
                tags.append(f'<div class="shape" style="{style}"></div>')
        body.append('<section><div class="scene">'+''.join(tags)+'</div></section>')
    rendered=QA/f'{lang}_intro.html'
    rendered.write_text('<!doctype html><meta charset="utf-8"><style>'
        '@page{size:10in 5.625in;margin:0}'
        '*{box-sizing:border-box}html,body{margin:0;padding:0}body{font-family:Pretendard,sans-serif;-webkit-print-color-adjust:exact;print-color-adjust:exact}'
        'section{position:relative;width:10in;height:5.625in;page-break-after:always;overflow:hidden}'
        f'.scene{{position:relative;width:{W}in;height:{H}in;transform:scale(.75);transform-origin:top left}}'
        'section:last-child{page-break-after:auto}.text,.shape,.clip{position:absolute}.text{white-space:pre;line-height:1.14}.clip{overflow:hidden}'
        '</style>'+''.join(body))
    rendered_pdf=QA/f'{lang}_intro.pdf'
    command=['/opt/google/chrome/chrome','--headless','--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--no-pdf-header-footer',
        '--allow-file-access-from-files','--run-all-compositor-stages-before-draw','--virtual-time-budget=5000',f'--print-to-pdf={rendered_pdf}',rendered.as_uri()]
    with (QA/f'{lang}_render.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=log,check=True,timeout=60)
    if include_experiments:
        subprocess.run(['pdfunite',str(rendered_pdf),*[str(QA/f'original-{n}.pdf') for n in (4,5,6)],str(dest.with_suffix('.pdf'))],check=True)
    else:
        dest.with_suffix('.pdf').write_bytes(rendered_pdf.read_bytes())
    (QA/f'{lang}_scene.json').write_text(json.dumps(dict(scenes=scenes,geometry=geometry,output=str(dest.relative_to(ROOT))),ensure_ascii=False,indent=2)+'\n')
    return str(dest.relative_to(ROOT))

def main():
    # Original experiments are untouched in the PDF and preserved as images in PPTX.
    subprocess.run(['pdftoppm','-f','4','-l','6','-scale-to','2400','-png',str(SOURCE),str(QA/'original')],check=True)
    subprocess.run(['pdfseparate','-f','4','-l','6',str(SOURCE),str(QA/'original-%d.pdf')],check=True)
    outputs=[write_deck('ko',make_scenes('ko'),False),write_deck('en',make_scenes('en'),True)]
    manifest=dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        edited_source_pages=[1,2,3],revised_intro_pages=4,unchanged_source_pages=[4,5,6],new_experiment_page_numbers=[5,6,7],
        outputs=outputs,main_tex_changed=False,experiment_results_changed=False,
        pptx_limitation='PDF-only source: experiment slides are page images; introduction is editable',
        reference=PAPER)
    (QA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
