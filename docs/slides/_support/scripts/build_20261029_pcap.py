"""Editable Korean PCAP research slides and matching PDF, with source/geometry QA."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').exists())
sys.path.insert(0,str(ROOT/'scripts/common'))
from experiment_paths import repo_root
import build_20261023_references as r

HERE=ROOT/'docs/slides'
QA=HERE/'_support/qa/20261029_pcap'
OUT=HERE/'kor/20261029_PCAP_연구방향'
EVIDENCE=ROOT/'docs/lablog/evidence/20261029_pcap'
RUN=ROOT/'tabpfn/results/v7/exp92/20261029_210839_cic2018_verified_pcap_4d62ceac'
SOURCE_DATA=json.loads((EVIDENCE/'sources.json').read_text())['sources']
r.QA=QA;r.ASSETS=HERE/'_support/assets/20261029_pcap';r.OUT=OUT;r.N_SLIDES=8;r.SCENES=[]
r.SOURCES={k:(v['title'],f"{v['venue']}, {v['year']}",v['url'],v['evidence']) for k,v in SOURCE_DATA.items()}
txt,box,arrow,block,page=r.txt,r.box,r.arrow,r.block,r.page
NAVY,BLUE,INK,GREY,PALE=r.NAVY,r.BLUE,r.INK,r.GREY,r.PALE
RED='B44948'

def card(x,y,w,h,title,body,size=21):
    return [box(x,y,w,h,PALE),box(x,y,w,.48,'E1ECF5'),txt(x+.16,y+.09,w-.32,title,21,True,NAVY),txt(x+.18,y+.72,w-.36,body,size)]

def table(x,y,widths,headers,rows,height=.55,size=20,colors=None):
    items=[]
    for ri,row in enumerate([headers]+rows):
        xx=x
        for ci,(w,value) in enumerate(zip(widths,row)):
            fill=NAVY if ri==0 else ('FFFFFF' if ri%2 else PALE)
            items.append(box(xx,y+ri*height,w,height,fill))
            color='FFFFFF' if ri==0 else (colors.get(ci,INK) if colors else INK)
            align='left' if ci==0 else 'right'
            items.append(txt(xx+.12,y+ri*height+.13,w-.24,str(value),size,ri==0,color,align))
            xx+=w
    return items

def asset_image(name,x,y,w,h):
    """Insert a generated chart without treating it as a literature figure."""
    from PIL import Image
    path=r.ASSETS/(name+'.png')
    iw,ih=Image.open(path).size
    scale=min(w/iw,h/ih)
    ww,hh=iw*scale,ih*scale
    return dict(kind='image',x=x+(w-ww)/2,y=y+(h-hh)/2,w=ww,h=hh,path=str(path))


def make_fewshot_asset():
    """Render the two requested EXP89 diagnostic views from the saved numeric CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np

    r.ASSETS.mkdir(parents=True,exist_ok=True)
    curve_path=ROOT/'docs/lablog/data/20261027_exp89_comparison_curves.csv'
    class_path=ROOT/'docs/lablog/data/20261027_exp89_class_comparison.csv'
    with curve_path.open() as f:
        curves=[x for x in csv.DictReader(f) if x['dataset']=='cic2018' and x['panel']=='1']
    with class_path.open() as f:
        classes=[x for x in csv.DictReader(f) if x['dataset']=='cic2018' and x['panel']=='1' and x['n']=='20']
    ns=[2,3,5,10,20]
    fp=FontProperties(fname='/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf')
    fp_bold=FontProperties(fname='/home/user/.local/share/fonts/Pretendard/Pretendard-Bold.otf')
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.0,4.1),dpi=190,gridspec_kw={'width_ratios':[1.0,1.20]})
    fig.patch.set_facecolor('white')
    colors={'xgb':'#B44948','pfn':'#007BC6'}
    labels={'xgb':'XGB','pfn':'TabPFN'}
    for method in ['xgb','pfn']:
        ys=[float(next(x['mean_percent'] for x in curves if x['method']==method and int(x['n'])==n)) for n in ns]
        ax1.plot(ns,ys,color=colors[method],lw=2.8,marker='o',ms=7,label=labels[method])
        ax1.annotate(f'{ys[-1]:.1f}',(ns[-1],ys[-1]),xytext=(7,5 if method=='pfn' else -16),textcoords='offset points',
                     color=colors[method],fontsize=10,fontproperties=fp_bold)
    ax1.set_title('공격 macro-F1',fontproperties=fp_bold,fontsize=15,pad=12)
    ax1.set_xlabel('클래스당 학습 라벨 수 n',fontproperties=fp,fontsize=11)
    ax1.set_ylabel('F1 (%)',fontproperties=fp,fontsize=11)
    ax1.set_xticks(ns);ax1.set_ylim(50,100.5);ax1.set_yticks([50,60,70,80,90,100])
    ax1.grid(axis='y',color='#D9E1E8',lw=.8);ax1.spines[['top','right']].set_visible(False)
    ax1.legend(loc='lower right',frameon=False,prop=fp,handlelength=2.8)
    ax1.text(.03,.05,'n=10 → 20\nXGB +2.9%p\nTabPFN +2.2%p',transform=ax1.transAxes,
             fontsize=10,fontproperties=fp,color='#334155',va='bottom')

    pretty={'ddos':'DDoS','brute_force':'Brute force','dos':'DoS','bot':'Bot','infiltration':'Infiltration','web_attacks':'Web attacks'}
    names=sorted({x['name'] for x in classes},key=lambda c:-int(next(x['support'] for x in classes if x['name']==c)))
    ypos=np.arange(len(names))
    h=.34
    for offset,method in [(-h/2,'xgb'),(h/2,'pfn')]:
        vals=[float(next(x['mean_percent'] for x in classes if x['name']==c and x['method']==method)) for c in names]
        ax2.barh(ypos+offset,vals,height=h,color=colors[method],label=labels[method])
        for y,v in zip(ypos+offset,vals):
            ax2.text(v+1.0,y,f'{v:.1f}',va='center',fontsize=8.5,fontproperties=fp,color=colors[method])
    ylabels=[f"{pretty[c]}  ({int(next(x['support'] for x in classes if x['name']==c)):,})" for c in names]
    ax2.set_title('수렴 구간 n=20의 클래스별 F1',fontproperties=fp_bold,fontsize=15,pad=12)
    ax2.set_yticks(ypos,ylabels,fontproperties=fp,fontsize=9.4);ax2.invert_yaxis()
    ax2.set_xlim(0,114);ax2.set_xlabel('F1 (%) · 괄호는 고정 attack test 표본 수',fontproperties=fp,fontsize=10)
    ax2.grid(axis='x',color='#D9E1E8',lw=.8);ax2.spines[['top','right','left']].set_visible(False)
    ax2.legend(loc='lower left',bbox_to_anchor=(0,-.02),ncol=2,frameon=False,prop=fp,handlelength=2.8)
    plt.tight_layout(pad=1.25,w_pad=2.8)
    fig.savefig(r.ASSETS/'fewshot_diagnostic.png',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def scenes():
    make_fewshot_asset()

    # 1. The presentation begins with the CSV actually used in prior work, then shows how CIC-IDS2017 produced it.
    a=[txt(.65,1.11,4.45,'기존 실험의 입력은 CSV였다',25,True,NAVY),
       txt(.65,1.64,4.45,'CIC-IDS2017은 PCAP과\n특징 CSV·라벨을 함께 제공',20)]
    a+=card(.65,2.50,4.45,1.26,'그동안의 실험','CSV의 flow 통계 벡터를\nXGB·TabPFN에 입력',19)
    a+=card(.65,4.00,4.45,1.26,'일반적인 IDS 표 분류 연구','제공된 CSV를 바로\n학습·평가 입력으로 사용',19)
    a += [txt(.72,5.78,4.28,'질문  이 CSV는 패킷에서\n어떤 정보를 남긴 표현인가?',22,True,BLUE),
          txt(5.58,1.11,6.75,'CIC-IDS2017에서 CSV가 만들어지는 과정',25,True,NAVY)]
    blocks=[(5.58,'실험망','정상·공격\n트래픽'),(7.38,'PCAP','시간순 원시\n패킷 기록'),(9.18,'CICFlowMeter','통신 단위로 묶고\n고정 통계량 계산'),(11.45,'CSV','숫자 벡터\n+ 라벨')]
    widths=[1.48,1.48,2.00,1.18]
    for (x,title,body),w in zip(blocks,widths):a+=block(x,1.94,w,1.37,title,body,body_size=15.5)
    for x,w in [(7.11,.18),(8.91,.18),(11.18,.18)]:a.append(arrow(x,2.45,w))
    a += [txt(9.18,3.49,4.02,'하나의 연결에서 오간 요청·응답 패킷을\n한 flow 레코드로 묶음',16,color=GREY,align='center'),
          txt(5.58,4.15,7.00,'CSV에 남는 특징의 예',20,True,NAVY)]
    examples=[('지속 시간','첫 패킷부터\n마지막 패킷까지'),('순방향 패킷 수','한 방향으로\n보낸 패킷 수'),('순방향 평균 길이','한 방향 패킷\n길이의 평균')]
    for x,(title,body) in zip([5.58,7.93,10.28],examples):a+=block(x,4.67,2.08,1.25,title,body,body_size=14)
    a += [txt(5.62,6.26,6.95,'[지속시간, 패킷 수, 평균 길이, …]  →  분류 모델 입력',18,True,BLUE)]
    page('CIC-IDS2017은 PCAP과 CICFlowMeter CSV를 함께 제공',a,
         'CSV 기반 기존 실험에서 시작해, 그 입력이 패킷을 어떻게 요약했는지 확인','01 기존 CSV와 진단',
         '왼쪽부터 설명한다. 기존 실험과 많은 IDS 표 분류 연구는 제공된 CSV의 flow 통계 벡터를 바로 분류기의 입력으로 썼다. 이어서 오른쪽에서 그 CSV가 실험망 트래픽의 PCAP을 CICFlowMeter가 통신 단위로 묶고, 미리 정의된 통계량을 계산한 결과임을 보여 준다. 여기서 flow는 같은 연결에서 오간 요청·응답 패킷을 하나의 레코드로 모은 것이다. CICFlowMeter는 공격에 중요한 특징을 학습해 고르는 모델이 아니라, 구현에 정해진 flow·통계 정의로 수치를 계산하는 도구다. 따라서 어떤 패킷 순서와 내용이 최종 벡터에서 사라지는지도 이후 검증 대상이 된다.', ['cic17'])

    # 2. Few-shot is a diagnostic for an unexpectedly high baseline, not the core operational scenario.
    a=[txt(.65,1.10,12.0,'일부 공격의 baseline이 이미 높았다  →  라벨을 극단적으로 줄여도 같은가?',22,True,NAVY)]
    a+=block(.65,1.69,3.78,.90,'출발점','일부 공격 F1 90%대',body_size=16)
    a+=block(4.78,1.69,3.78,.90,'진단 질문','클래스당 n=2까지 낮춰도 높은가?',body_size=16)
    a+=block(8.91,1.69,3.77,.90,'고정 조건','정제 전·정상 미포함 · 동일 attack test · 5 seeds',body_size=12.5)
    a += [asset_image('fewshot_diagnostic',.62,2.66,12.10,3.68),
          txt(.72,6.53,11.9,'few-shot은 방법의 주 성능 주장이 아니라, head 클래스의 조기 포화와 data ceiling을 드러내는 진단',17,True,NAVY)]
    page('few-shot은 높은 baseline을 점검한 진단',a,
         'n을 2개까지 낮춰도 일부 head 공격이 고점에 도달하는지 확인','01 기존 CSV와 진단',
         'few-shot을 먼저 한 이유는 TabPFN의 우위를 주장하려는 것이 아니라, 기존 baseline이 일부 공격에서 이미 매우 높았기 때문이다. 클래스당 라벨을 n=2, 3, 5, 10, 20개로 줄이고 동일한 attack-only test로 분류해 조기 포화 여부를 본다. macro-F1은 n=2에서 n=20으로 XGB 60.0에서 78.1, TabPFN 65.1에서 78.3으로 올라가므로 전체 변화가 없다고 해석하지 않는다. 다만 n=10에서 n=20의 증가는 XGB 2.9%p, TabPFN 2.2%p로 둔화하고 DDoS·Bot은 n=20에서 98–99%에 도달한다. Web attacks는 11–12%에 머물러, 모든 공격이 쉬웠다는 뜻도 아니다. 이 결과는 정상 포함 IDS 성능이 아니며, 왜 특정 head가 쉽게 분리되는지와 tail의 낮은 성능 원인을 별도로 점검해야 한다.', [])

    # 3. Put observed exact conflicts before the literature explanation, without asserting they are all wrong labels.
    a=[txt(.65,1.08,12.0,'기존 입력 CSV의 46개 숫자가 모두 같아도 라벨은 둘 이상',24,True,NAVY)]
    metrics=[('20,115,529','감사한 전체 행'),('20,609','상이 라벨 벡터 그룹'),('26.6%','상이 라벨 그룹에 속한 행')]
    for x,(n,label) in zip([.65,3.15,5.65],metrics):
        a += [box(x,1.67,2.15,1.15,PALE),txt(x+.1,1.87,1.95,n,25,True,BLUE,'center'),txt(x+.1,2.36,1.95,label,14,color=GREY,align='center')]
    a += [txt(8.25,1.79,4.2,'감사 대상  기존 실험 입력의\nCIC2018 기반 46개 flow 통계 벡터',18,True,NAVY),
          txt(8.25,2.50,4.2,'공식 CICFlowMeter CSV 그대로의\n중복률 측정과는 구분',15,color=GREY)]
    a+=block(.65,3.35,3.02,1.37,'같은 특징 벡터 x','TCP · 60B/40B\n각 1 packet · 1 ms',body_size=17)
    a += [arrow(3.88,3.86,.35),arrow(3.88,4.42,.35),
          box(4.48,3.35,3.38,.61,'FBE8E7'),txt(4.65,3.54,3.04,'Brute force  386,780행',19,True,RED,'center'),
          box(4.48,4.11,3.38,.61,'E6F1F8'),txt(4.65,4.30,3.04,'DoS  105,550행',19,True,BLUE,'center'),
          txt(8.30,3.46,4.14,'한 개의 특징 벡터가\n두 공격에 반복 등장',20,True,NAVY),
          txt(8.30,4.22,4.14,'Brute force 67.2% · DoS 34.8%\n각 클래스 행이 이 충돌에 포함',16,color=GREY)]
    a += [box(.65,5.47,12.03,.02,'C8D5E1'),
          txt(.72,5.76,5.86,'Infiltration–Benign\n20,378개 벡터 · infiltration의 75.1%',18,True,NAVY),
          txt(6.78,5.76,5.80,'Web attacks–Benign\n298개 벡터 · web의 69.5%',18,True,NAVY),
          txt(.72,6.53,11.9,'같은 x에는 하나의 결정적 예측만 가능  →  기존 full XGB에서도 DoS F1 0.782, Infiltration F1 0.429',16,color=GREY)]
    page('기존 입력 CSV에는 동일 벡터·상이 라벨이 실제로 존재',a,
         '정확히 같은 입력을 서로 다른 정답으로 학습시키는 구조를 먼저 계량','01 기존 CSV와 진단',
         '이 슬라이드의 감사는 기존 실험에 사용한 CIC2018 기반 46개 flow 통계 벡터를 대상으로 했다. 따라서 공식 CICFlowMeter CSV 원본 자체의 중복률이라고 바꾸어 말하지 않는다. 전체 2,011만 행 중 26.6%가 상이한 라벨을 가진 동일 벡터 그룹에 속했고, 그런 그룹은 20,609개였다. 예시 하나는 TCP의 매우 짧은 요청·응답 형태로, 정확히 같은 벡터가 brute force 386,780행과 DoS 105,550행에 붙어 있다. Infiltration과 Web attacks도 각각 benign과 상당수의 동일 벡터를 공유한다. 이것은 모든 라벨이 잘못되었다는 증거가 아니라, 현재 통계 표현만으로는 두 정답을 동시에 분리할 수 없거나 라벨 근거를 점검해야 한다는 증거다. 실제 full XGB에서 DoS·Infiltration이 상대적으로 낮게 남은 이유를 분석할 때 이 구조를 먼저 봐야 한다.', [])

    # 4. The literature explains possible mechanisms after, not before, the observed audit.
    a=[txt(.65,1.08,12.0,'선행연구는 CIC의 flow 생성·라벨 규칙에서 가능한 원인을 제시',22,True,NAVY)]
    papers=[('engelen','Engelen et al. (IEEE SPW/WTMC, 2021)','첫 FIN 뒤 잔여 패킷이 별도 flow로 분리되고\nRST를 무시한 flow 경계 문제를 분석',1.70),
            ('liu','Liu et al. (IEEE CNS, 2022)','CIC2017·2018의 생성·특징·라벨 오류를 재검토\nCIC2018 FTP closed-port 사례를 Attempted로 구분',3.28),
            ('rosay','Rosay et al. (ICISSP, 2022)','세션 종료와 특징 계산을 점검\n유사 flow에 상이한 라벨이 생기는 경로를 논의',4.86)]
    for key,head,body,y in papers:
        a += [box(.65,y,12.03,1.23,PALE),txt(.86,y+.14,4.62,head,18,True,NAVY,url=SOURCE_DATA[key]['url']),txt(5.70,y+.22,6.65,body,18)]
    a += [txt(.72,6.35,11.8,'문헌은 가능한 원인, 슬라이드 3은 현재 입력에서 관측한 현상  ·  모든 exact collision을 한 원인으로 단정하지 않음',16,True,BLUE)]
    page('문헌은 CIC의 flow 생성·라벨 규칙 결함을 지적',a,
         '현재 감사의 exact collision을 문헌의 가능 원인과 구분해 해석','01 기존 CSV와 진단',
         '앞 슬라이드가 우리 입력에서 직접 센 현상이라면, 이 슬라이드는 가능한 원인을 확인한 문헌 근거다. Engelen 등은 CICIDS2017의 TCP 종료 처리와 flow 경계 문제를 분석했다. Liu 등은 CIC2017·2018의 공격 수행·특징·라벨 오류를 재검토하고, 닫힌 FTP 포트로 인해 실제 brute force가 진행되지 않은 사례를 구분했다. Rosay 등은 세션 종료와 특징 계산에서 유사 flow에 다른 라벨이 생길 수 있는 경로를 논의했다. 이 논문들은 현재 46개 값이 일치한 모든 사례가 동일한 한 가지 오류 때문이라고 증명하지 않는다. 그래서 다음 단계는 PCAP에서 packet·flow·label 근거를 다시 연결해 표현 부족과 라벨 문제를 분리하는 것이다.', ['engelen','liu','rosay'])

    # 5. Correct CICFlowMeter rationale: it calculates a fixed summary, it does not learn the task-specific representation.
    a=[txt(.65,1.08,12.0,'고정 flow 통계 요약을 PCAP 인코더 표현으로 다시 만들고 두 결함을 재검증',22,True,NAVY)]
    a+=card(.65,1.70,5.76,1.55,'CICFlowMeter CSV','미리 정한 flow 경계와 통계식으로\n지속시간·패킷 수·길이·IAT·flag 등을 계산',19)
    a+=card(6.83,1.70,5.76,1.55,'PCAP 사전학습 인코더','패킷 순서·시간·헤더 패턴을 입력으로\n인코더별 표현 벡터를 생성',19)
    a += [txt(.80,3.55,5.46,'고정 통계량은 학습으로 고른 특징이 아니다.\n패킷 순서·상호작용은 요약에서 사라질 수 있다.',17,color=GREY),
          txt(7.02,3.55,5.46,'새 임베딩도 라벨 오류를 자동으로 고치지 않는다.\nPCAP·시간·host·port 근거를 따로 보존한다.',17,color=GREY)]
    steps=[(.65,'PCAP·라벨','패킷·시각·키'),(3.15,'flow 추적','원시 통신·라벨'),(5.65,'고정 인코더','표현 벡터 추출'),(8.15,'차원 축소','학습 분할만'),(10.65,'XGB · TabPFN','같은 벡터로 비교')]
    for x,title,body in steps:a+=block(x,4.36,2.0,1.27,title,body,body_size=15.5)
    for x in [2.79,5.29,7.79,10.29]:a.append(arrow(x,4.84,.24))
    a += [txt(.72,5.90,5.73,'① 동일 표현·상이 라벨 비율과\nPCAP 라벨 근거를 다시 감사',18,True,NAVY),
          txt(6.83,5.90,5.73,'② baseline·few-shot에서\nhead의 조기 포화가 다시 생기는지 측정',18,True,NAVY),
          txt(.72,6.70,11.9,'인코더 후보  netFound (arXiv, 26) · ET-BERT (WWW, 22) · NetMamba (ICNP, 24)',14,color=GREY)]
    page('PCAP 인코더로 새 학습 벡터를 만들 이유',a,
         '표현을 바꾼 뒤 동일 벡터·상이 라벨과 쉬운 head의 조기 포화를 다시 측정','02 PCAP 재구축',
         'CICFlowMeter는 원시 패킷을 flow로 묶고 구현에 미리 정해진 통계량을 계산한다. 공격을 가장 잘 구분할 특징을 데이터에서 학습해 선택하는 인코더는 아니다. 따라서 현재 CSV에서 빠진 패킷 순서와 시간·헤더 상호작용이 분류에 필요한지 직접 확인할 여지가 있다. PCAP 인코더는 이런 정보를 사용해 새로운 벡터를 만들지만, 어떤 정보가 쓰이는지는 인코더에 따라 다르며 라벨 오류를 자동으로 고치지 않는다. 그래서 PCAP과 label rule을 분리해 보존하고, 새 벡터에서도 동일 입력·상이 라벨 비율과 baseline의 조기 포화를 각각 다시 측정한다. PCA는 벡터 차원을 맞추는 변환이며, 적합 자료는 학습 분할로 제한한다. 현재 제시한 인코더들은 앙상블이 아니라 같은 재사용 파이프라인에서 교체해 비교할 후보들이다.', ['netfound','etbert','netmamba'])

    # 6. Keep data selection tied to packet-label traceability.
    a=[txt(.65,1.13,12.0,'PCAP 제공 여부뿐 아니라, 새 flow에 라벨 근거를 다시 연결할 수 있는지가 기준',20,color=GREY)]
    rows=[['CIC2017 · 우선','PCAP + flow CSV + 공격 일정','flow 경계·라벨 규칙 대조'],
          ['CIC2018 · 우선','호스트별 PCAP + 로그 + CSV','중복 캡처·시간·host 확인'],
          ['IoT-23 · 후속','PCAP + 라벨 Zeek conn.log','작은 시나리오부터 연결'],
          ['UNSW-NB15','PCAP + GT·공격 이벤트','시간·5-tuple join'],
          ['TON_IoT','네트워크 PCAP + Zeek·CSV','네트워크와 센서·OS 자료 구별'],
          ['CICIoT2023','PCAP + CSV·처리 코드','flow별 라벨 재구성 검증']]
    xs=[.65,3.45,7.64];ws=[2.8,4.19,5.04]
    for ri,row in enumerate([['데이터','공개 원시 자료','새 벡터를 위한 검증']]+rows):
        for ci,value in enumerate(row):
            y=1.82+ri*.62
            a += [box(xs[ci],y,ws[ci],.62,NAVY if ri==0 else PALE if ri%2==0 else 'FFFFFF'),
                  txt(xs[ci]+.12,y+.17,ws[ci]-.24,value,17,ri==0,'FFFFFF' if ri==0 else INK,
                      url=SOURCE_DATA[['cic17','cic18','iot23','unsw','ton','iot2023'][ri-1]]['url'] if ri and ci==0 else None)]
    a += [txt(.72,6.45,11.9,'CIC2017·2018을 먼저 재구축하고, IoT-23처럼 작은 PCAP 시나리오로 파이프라인·라벨 연결을 추가 검증',18,True,NAVY)]
    page('PCAP·라벨을 함께 제공하는 IDS 데이터 후보',a,
         'CIC2017·2018을 우선하고, PCAP과 라벨을 다시 연결할 수 있는 데이터로 확장','02 PCAP 재구축',
         '데이터 후보는 단순히 PCAP을 제공하는지보다, 새로 만든 flow와 원래 라벨 근거를 다시 연결할 수 있는지로 고른다. CIC2017과 CIC2018은 우선 대상이다. IoT-23은 개별 scenario의 PCAP과 라벨 Zeek connection log가 있어 작은 규모에서 연결 규칙을 먼저 검증할 후속 후보다. UNSW-NB15, TON_IoT, CICIoT2023은 원시 패킷과 부가 로그 또는 ground truth를 제공하지만, 각 자료의 label join 규칙은 별도 구현·감사가 필요하다. 이 표는 모든 데이터를 이미 다운로드하거나 재구축했다는 주장이 아니다.', ['cic17','cic18','iot23','unsw','ton','iot2023'])

    # 7. Retain the completed pipeline pilot but make its scope explicit.
    prep=json.loads((RUN/'prepare.json').read_text());projection=json.loads((RUN/'projection.json').read_text())
    audit=json.loads((RUN/'audit.json').read_text());stat=next(iter(prep['raw_stats'].values()))
    a=card(.65,1.35,5.76,3.14,'CIC2018 PCAP 파일럿',
           f"2018-02-14 · 대상 host 1대\n패킷 {stat['raw_packets']:,}개 → flow {stat['flows']:,}개\n주석·분할별 표본 {projection['rows']:,}개 인코딩\n512차원 → 64차원 벡터",21)
    a+=card(6.83,1.35,5.76,3.14,'확인한 처리 계약',
            f"학습·검증·시험 flow 그룹 중복 0\n변환 적합: 학습 {projection['pca_fit_rows']:,}행만 사용\n공개 checkpoint 가중치 고정\n미확정 라벨은 별도 보존",20)
    a += [txt(.72,4.94,11.9,'현재는 PCAP → 학습 벡터 생성의 구현 검증\n자연 분포 분류 benchmark와 few-shot 비교는 다음 단계',19,True,NAVY),
          txt(.72,5.75,11.9,f"라벨 미확정 {audit['unresolved_rows']:,}행은 정상으로 추정하지 않음  ·  공격 범위를 넓힌 새 CIC2018 구축 작업으로 확장",19),
          txt(.72,6.38,11.9,'완료 파일럿은 단일 날짜·host·제한된 주석 표본  ·  정상 오탐률·분류 성능·TabPFN 우위는 아직 측정하지 않음',17,color=GREY)]
    page('구현 파일럿: PCAP부터 학습용 벡터까지',a,
         '표현 생성 파이프라인은 검증했고, 새 CIC2018의 분류 평가는 공격 범위를 확장한 뒤 수행','02 PCAP 재구축',
         f'완료 실행은 {RUN.relative_to(ROOT)}이다. CIC2018의 2018-02-14 대상 host 한 대에서 패킷을 읽고 flow·라벨·분할 정보를 보존한 채 netFound embedding과 64차원 변환 벡터를 만들었다. 그룹 중복 없이 분리하고, 차원 축소 변환은 학습 행만으로 적합했다. 이 표본은 자연 클래스 분포 benchmark가 아니며 일부 라벨은 미확정으로 보존했다. 따라서 현재 파일럿으로 분류 성능, 정상 오탐률, TabPFN의 우위를 주장하지 않는다. 새 CIC2018 전체에서 공격군과 정상 범위를 확장한 뒤, 슬라이드 5의 두 감사와 기존 baseline·few-shot 조건을 같은 분할에서 반복하는 것이 다음 단계다.', ['aws','rules'])

    # 8. This is explicitly a historical appendix, not the discarded 2x2 plan.
    report=(ROOT/'docs/lablog/reports/20260904.tex').read_text()
    section=report.split(r'\subsection*{CIC2018 · 클래스별 F1 · test 3,042,473행}',1)[1].split(r'\bottomrule',1)[0]
    sota=[]
    for line in section.splitlines():
        if not line.startswith('{}'):continue
        fields=[v.strip() for v in line[2:].split('&')]
        sota.append([fields[0],fields[1],*[f'{float(fields[i])*100:.2f}' for i in [2,4,5,6]]])
    assert len(sota)==7 and sum(int(x[1].replace(',','')) for x in sota)==3042473
    assert [int(x[1].replace(',','')) for x in sota]==sorted([int(x[1].replace(',','')) for x in sota],reverse=True)
    a=[txt(.65,1.13,12.0,'CIC2018 · 정제 후·정상 포함 · 공통 학습 100,000행 · 전체 test 3,042,473행 · F1 (%)',18,color=GREY)]
    a+=table(.65,1.84,[2.1,1.6,2.09,2.08,2.08,2.08],['평가 클래스','평가 수','TabPFN v3','BoostPFN','LoCalPFN','DistPFN v3'],sota,height=.53,size=17)
    a += [txt(.72,6.27,11.9,'주요 공격의 높은 F1과 Infiltration·Web의 낮은 F1이 공존',22,True,NAVY),
          txt(.72,6.69,11.9,'부록 · few-shot 및 새 PCAP 표현과는 준비·라벨 예산·평가 범위가 달라 직접 수치 비교 불가',14,color=GREY)]
    page('부록: 기존 SOTA 비교에서도 보인 클래스별 격차',a,
         'historical comparison은 문제의식의 근거이며, 새 PCAP 평가와 직접 합치지 않음','부록',
         '이 표는 기존 SOTA 비교의 역사적 결과다. 정제 후·정상 포함·100,000행 공통 학습·전체 test 조건으로, 본문 few-shot의 정제 전·정상 미포함·클래스당 n개 조건과 다르다. 주요 공격은 높지만 Infiltration과 Web은 낮아, 평균 수치만으로 데이터와 모델의 한계를 판단할 수 없었다. BoostPFN과 LoCalPFN은 TabPFN v1 기반이고 LoCalPFN은 validation fine-tuning을 사용했다. 이 수치를 새 PCAP 표현 또는 현재 제안 모델의 결과로 사용하지 않는다. 기존의 2×2 A–D 비교 계획 슬라이드는 이번 자료에서 제거했다. 아직 실행하지 않은 라벨 교정 arm을 현재 결론처럼 보이게 하고, CSV의 관측 문제와 PCAP 재구축의 동기를 끊었기 때문이다.', [])

def main():
    QA.mkdir(parents=True,exist_ok=True);r.ASSETS.mkdir(parents=True,exist_ok=True)
    scenes();r.validate_scenes();r.pptx()
    from pptx import Presentation
    prs=Presentation(OUT.with_suffix('.pptx'))
    prs.core_properties.title='CIC 데이터 결함 검토와 PCAP 기반 학습 벡터 재구축'
    prs.core_properties.subject='문헌 근거, 인코더·데이터 후보, EXP92 구현 검증'
    prs.save(OUT.with_suffix('.pptx'))
    checks=r.pdf()
    # pdftoppm emits page-1.png; keep zero-padded stable QA asset names.
    for i in range(1,len(checks)+1):
        src=QA/f'page-{i}.png';dst=QA/f'page-{i:02d}.png'
        if src.exists():src.replace(dst)
    from PIL import Image,ImageDraw
    for old in QA.glob('contact-*.png'):old.unlink()
    paths=sorted(QA.glob('page-*.png'))
    for start in range(0,len(paths),5):
        canvas=Image.new('RGB',(1500,1335),'#DCE3EB')
        for j,p in enumerate(paths[start:start+5]):
            im=Image.open(p);im.thumbnail((730,411));x=(j%2)*750+10;y=(j//2)*445+8
            canvas.paste(im,(x,y));ImageDraw.Draw(canvas).text((x,y+414),str(start+j+1),fill='black')
        canvas.save(QA/f'contact-{start//5+1}.png')
    (QA/'scenes.json').write_text(json.dumps(r.SCENES,ensure_ascii=False,indent=2)+'\n')
    visible='\n'.join(it.get('text','') for scene in r.SCENES for it in scene['items'])
    assert 'NF-v3' not in visible and 'NF-CIC' not in visible
    manifest=dict(date='2026-10-30',slides=len(r.SCENES),source_evidence=str(EVIDENCE.relative_to(ROOT)),
                  run=str(RUN.relative_to(ROOT)),old_results='EXP89 uncurated attack-only, five seeds; not a new PCAP benchmark',
                  validation='shared scenes for editable PPTX and Chrome PDF; geometry/font/text checks',
                  visual_review='2026-10-30 rendered PDF and all eight pages inspected',pages=checks,
                  pptx_sha256=hashlib.sha256(OUT.with_suffix('.pptx').read_bytes()).hexdigest(),
                  pdf_sha256=hashlib.sha256(OUT.with_suffix('.pdf').read_bytes()).hexdigest())
    (QA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(pptx=str(OUT.with_suffix('.pptx')),pdf=str(OUT.with_suffix('.pdf')),slides=len(r.SCENES)),ensure_ascii=False))

if __name__=='__main__':main()
