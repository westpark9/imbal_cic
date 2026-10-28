#!/usr/bin/env python3
"""Professor-2 review deck: few-shot evidence and three relevant papers."""
from copy import deepcopy
from pathlib import Path
import hashlib, importlib.util, json, re, subprocess, zipfile
from pptx import Presentation

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference_slides',HERE/'build_20261023_references.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
g.QA=g.SLIDES/'_support/qa/20261023_references'
g.ASSETS=g.QA/'assets'
g.OUT=g.SLIDES/'kor/20261023_References_ko'
g.N_SLIDES=7;g.SCENES=[]
T,B,P,C=g.txt,g.box,g.page,g.caption
NAVY,BLUE,INK,GREY,PALE,TEAL,AMBER=g.NAVY,g.BLUE,g.INK,g.GREY,g.PALE,g.TEAL,g.AMBER

def pdf_asset(src:Path,name:str,scale='2200'):
    out=g.ASSETS/name
    subprocess.run(['pdftoppm','-singlefile','-scale-to',scale,'-png',str(src),str(out)],check=True,capture_output=True)
    return out.with_suffix('.png')

def image(path,x,y,w,h):
    from PIL import Image
    iw,ih=Image.open(path).size;s=min(w/iw,h/ih)
    return dict(kind='image',x=x+(w-iw*s)/2,y=y+(h-ih*s)/2,w=iw*s,h=ih*s,path=str(path))

def add(title,items,summary,notes,sources=(),section='01 선행 실험'):
    P(title,items,summary,section,notes,list(sources))

def experiment_slides():
    g.ASSETS.mkdir(parents=True,exist_ok=True)
    figs=g.ROOT/'docs/lablog/figures'
    cic_curve=pdf_asset(figs/'exp87_quality_cic2018.pdf','few_curve_cic')
    ton_curve=pdf_asset(figs/'exp87_quality_toniot.pdf','few_curve_ton')
    cic_class=pdf_asset(figs/'references_20261022/class_cic2018.pdf','few_class_cic')
    ton_class=pdf_asset(figs/'references_20261022/class_toniot.pdf','few_class_ton')

    a=[T(.7,1.08,11.95,'공격 클래스만 평가 · 정상 제외 · 클래스별 같은 n개 학습 · 고정 attack test · 5 seeds',17,color=GREY),
       image(cic_curve,.66,1.52,6.0,3.62),image(ton_curve,6.7,1.52,6.0,3.62),
       T(.82,5.36,5.63,'CIC2018  n=5  +2.55%p\nn=10  +0.21%p',20,True,NAVY),
       T(6.89,5.36,5.62,'ToN-IoT  n=5  +6.46%p\nn=10  +2.58%p',20,True,NAVY),
       T(.82,6.18,11.62,'공격 종류를 구분하는 성능은 형성 · 모델 격차는 데이터와 n에 따라 축소·역전',20,True,TEAL)]
    add('극소수 공격 사례의 분류 가능성과 모델 차이',a,
        'TabPFN 우위는 일부 극소 구간에서 관찰 · 표본 수 전반에서 일관된 우위는 미확인',
        '선행실험은 정상 트래픽을 제외하고 공격 클래스끼리 구분할 수 있는지 확인했습니다. 각 공격 클래스에서 같은 수의 실제 train 행을 선택했고 XGBoost와 TabPFN에 동일한 행을 제공했습니다. 작은 n은 큰 n의 부분집합이며 seed 43부터 47까지의 평균과 표본 표준편차를 표시했습니다. CIC2018의 n=5에서 TabPFN과 XGBoost의 차이는 +2.55%p였지만 n=10에서는 +0.21%p였습니다. ToN-IoT에서는 각각 +6.46%p와 +2.58%p였습니다. 더 큰 n에서 두 곡선의 순위가 바뀌기도 했으므로 극소수 조건에서 TabPFN이 항상 우세하다고 결론 내리지 않습니다. 이 실험은 정상 오탐과 순차 갱신 후의 기존 공격 유지도 평가하지 않았습니다. 근거는 20261021_references 보고서 0절과 부록의 5-seed 학습곡선입니다.')

    a=[image(cic_class,.65,1.22,8.48,5.72),
       T(9.34,1.39,3.2,'n=10의 TabPFN F1',20,True,NAVY),
       T(9.34,1.98,3.2,'DDoS  99.6\nBot  99.4\nBrute force  100.0',21),
       B(9.25,3.45,3.35,.02,'CAD5DF'),
       T(9.34,3.77,3.2,'어려운 공격',20,True,AMBER),
       T(9.34,4.34,3.2,'Infiltration  84.3\nWeb attacks  4.3',21),
       T(9.34,5.56,3.2,'PFN−XGB\n+3.8%p / −3.1%p',18,True,TEAL)]
    add('CIC2018 · 평균 점수가 가리는 공격별 난이도',a,
        '쉬운 공격은 10개 이내에도 높은 F1 · Web attacks는 두 모델 모두 해결하지 못함',
        '셀 윗줄은 TabPFN의 클래스별 F1이고 괄호는 같은 조건의 TabPFN−XGBoost 차이입니다. 행은 고정 test 표본 수 내림차순입니다. CIC2018에서 DDoS, Bot, Brute force는 n=10에서도 거의 100%에 도달했지만 Web attacks는 TabPFN 4.3%였습니다. Infiltration은 84.3%였고 XGBoost보다 3.8%p 높았습니다. 반면 Web attacks는 XGBoost보다 3.1%p 낮았습니다. 평균 학습곡선이 비슷해도 어떤 공격이 해결됐는지는 다릅니다. 따라서 후속 IDS 평가는 전체 Macro-F1뿐 아니라 클래스별 precision, recall, F1과 정상 오탐을 함께 봐야 합니다.')

    a=[image(ton_class,.62,1.13,8.53,5.82),
       T(9.34,1.31,3.2,'n=10의 TabPFN F1',20,True,NAVY),
       T(9.34,1.88,3.2,'Backdoor  96.2\nDoS  87.2\nXSS  85.6',21),
       B(9.25,3.35,3.35,.02,'CAD5DF'),
       T(9.34,3.67,3.2,'어려운 공격',20,True,AMBER),
       T(9.34,4.24,3.2,'Scanning  46.4\nRansomware  28.6\nMITM  6.7',21),
       T(9.34,5.76,3.2,'PFN−XGB\n+12.0 / +6.3 / +1.6%p',18,True,TEAL)]
    add('ToN-IoT · 소수 사례의 효과도 공격마다 다름',a,
        '일부 공격은 TabPFN 이득 · MITM 등은 두 모델의 절대 성능이 계속 낮음',
        'ToN-IoT에서도 셀 윗줄은 TabPFN F1, 괄호는 TabPFN−XGBoost 차이입니다. n=10에서 Backdoor, DoS, XSS는 비교적 높았지만 Scanning, Ransomware, MITM은 각각 46.4%, 28.6%, 6.7%였습니다. 같은 조건에서 TabPFN은 XGBoost보다 각각 12.0, 6.3, 1.6%p 높았습니다. 상대적 이득이 있어도 MITM의 절대 성능은 낮습니다. Password와 Injection처럼 XGBoost가 더 높은 공격도 있습니다. 결론은 극소수 사례 자체가 항상 분류 불가능한 것은 아니지만 모델 간 평균 차이만으로 후속 설계를 정하기 어렵다는 것입니다. 순차 평가에서는 정상·기존 공격과 경쟁하는 조건에서 이 클래스별 차이가 유지되는지 확인해야 합니다.')

def paper_slides():
    # Reuse the paper-explanation slides already reviewed in the uploaded deck.
    old=json.loads((g.QA/'scenes.json').read_text())
    wanted=['LotBoNC · 원본 트래픽에서 평가 조건을 만드는 과정',
            'LotBoNC · 임시 정답과 클래스 표현의 반복 갱신',
            '데이터 스트림 · 예측 순서와 context 비용의 분리',
            'ICD · 모델 대신 context의 특징값 학습']
    by_title={s['title']:s for s in old}
    assert all(x in by_title for x in wanted)
    summaries={
      wanted[0]:'참고점  같은 출처에서 known·novel과 라벨 공개 범위를 명시 · 스트림 재현과 구분',
      wanted[1]:'참고점  라벨 없는 novel·긴 꼬리 학습 방식 · 우리 감독 분류 모델로 직접 채택하지 않음',
      wanted[2]:'평가 설계에 직접 참고  예측 → 라벨 확보 → 갱신 · 메모리와 처리 비용 분리',
      wanted[3]:'후속 방법 후보  큰 라벨 이력을 작은 context로 압축 · 극소 라벨 성능 근거와 구분'}
    for title in wanted:
        scene=deepcopy(by_title[title])
        for it in scene['items']:
            if it['kind']=='text' and it['y']>7.0:
                if it['x']<10:it['text']=summaries[title]
                elif it['x']<12:it['text']='02 관련 연구'
                else:it['text']=str(len(g.SCENES)+1)
        scene['notes']=scene['notes'].replace('우리의 시나리오에도 같은 출처의 어느 행을 언제 라벨 사례로 공개하는지 설명하는 방식을 적용할 수 있습니다.',
            '우리 시나리오에는 같은 출처의 어느 행을 언제 라벨 사례로 공개하는지 설명하는 방식을 적용합니다.')
        g.SCENES.append(scene)

def main():
    g.QA.mkdir(parents=True,exist_ok=True);g.ASSETS.mkdir(parents=True,exist_ok=True)
    experiment_slides();paper_slides();g.validate_scenes();g.pptx()
    prs=Presentation(g.OUT.with_suffix('.pptx'))
    prs.core_properties.title='극소수 공격 사례 실험과 TabPFN 관련 연구 검토'
    prs.core_properties.subject='교수 2 검토 자료 · 선행실험, LotBoNC, data streams, ICD'
    prs.save(g.OUT.with_suffix('.pptx'))
    checks=g.pdf()
    with zipfile.ZipFile(g.OUT.with_suffix('.pptx')) as z:
        shadows=sum(len(re.findall(rb'<a:(?:outerShdw|innerShdw|prstShdw)',z.read(n))) for n in z.namelist() if n.endswith('.xml'))
        assert shadows==0
    (g.QA/'scenes.json').write_text(json.dumps(g.SCENES,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(date='2026-10-23',purpose='Professor 2 reviews preliminary evidence and the literature basis before the evaluation explanation is shared with Professor 1',slides=7,groups={'few_shot_evidence':[1,2,3],'lotbonc':[4,5],'streams':[6],'icd':[7]},source_report='docs/lablog/reports/20261021_references.tex',sources=g.SOURCES,no_experiments_run=True)
    (g.QA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (g.QA/'validation.json').write_text(json.dumps(dict(passed=True,pages=checks,speaker_notes=7,pptx_shadow_count=0,pdf_sha256=g.sha(g.OUT.with_suffix('.pdf')),pptx_sha256=g.sha(g.OUT.with_suffix('.pptx')),visual_review_pending=True),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(pptx=str(g.OUT.with_suffix('.pptx')),pdf=str(g.OUT.with_suffix('.pdf')),slides=7),ensure_ascii=False))

if __name__=='__main__':main()
