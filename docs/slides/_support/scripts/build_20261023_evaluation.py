#!/usr/bin/env python3
"""Unified Korean explanation of the existing evaluation and proposed refinements."""
from pathlib import Path
import importlib.util, json, csv, hashlib, zipfile
from PIL import Image
from pptx import Presentation

spec=importlib.util.spec_from_file_location('reference_slides',Path(__file__).with_name('build_20261023_references.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
g.QA=g.SLIDES/'_support/qa/20261023_evaluation'
g.ASSETS=g.QA/'assets'
g.OUT=g.SLIDES/'kor/20261023_Evaluation_ko'
g.N_SLIDES=4;g.SCENES=[]
g.QA.mkdir(parents=True,exist_ok=True);g.ASSETS.mkdir(parents=True,exist_ok=True)
g.SOURCES.update({
 'cic':('CSE-CIC-IDS2018 on AWS','UNB CIC official dataset documentation','https://www.unb.ca/cic/datasets/ids-2018.html','Collection environment and attack schedule'),
 'nf':('NetFlow V3 Datasets','University of Queensland dataset documentation','https://staff.itee.uq.edu.au/marius/NIDS_datasets/','NF-CSE-CIC-IDS2018-v3 / NF-ToN-IoT-v3; preprocessing and split counts follow local experiment records'),
})
T,B,A,P,C=g.txt,g.box,g.arrow,g.page,g.caption
NAVY,BLUE,INK,GREY,PALE,TEAL,AMBER=g.NAVY,g.BLUE,g.INK,g.GREY,g.PALE,g.TEAL,g.AMBER
OLD,NEW='6D8EAB','B87828'

def tag(text):return g.chip(9.55,1.02,3.1,text,fill=PALE,color=TEAL,size=15)
def add(title,items,summary,notes,keys=(),section='평가 시나리오'):
    P(title,items,summary,section,notes,list(keys))
def line(x,y,w,color='DCE5ED'):return B(x,y,w,.015,color)
def fig(path,x,y,w,h):
    iw,ih=Image.open(path).size;s=min(w/iw,h/ih)
    return dict(kind='image',x=x+(w-iw*s)/2,y=y+(h-ih*s)/2,w=iw*s,h=ih*s,path=str(path))

def learning_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=FontProperties(fname=str(g.FONT));plt.rcParams.update({'axes.unicode_minus':False,'font.size':13})
    values={'CIC2018':([63.24,72.24,80.37,80.43],[66.34,74.78,80.58,80.59],[6.32,4.66,1.91,.97],[10.44,4.29,.73,1.64]),
            'ToN-IoT':([45.03,52.29,57.88,64.74],[45.77,58.75,60.46,69.13],[3.27,5.62,5.08,2.18],[6.04,3.67,3.78,4.49])}
    fig,axs=plt.subplots(1,2,figsize=(12.4,3.9),dpi=200,sharey=True)
    for ax,(name,(x,p,xsd,psd)) in zip(axs,values.items()):
        xx=range(4)
        ax.errorbar(xx,x,yerr=xsd,color='#303640',marker='s',lw=2,capsize=3,label='XGB')
        ax.errorbar(xx,p,yerr=psd,color='#007BC6',marker='o',lw=2,capsize=3,label='TabPFN')
        for j,(a,b) in enumerate(zip(x,p)):
            ax.text(j,a-6,f'{a:.1f}',ha='center',color='#303640',fontsize=11)
            ax.text(j,b+5,f'{b:.1f}',ha='center',color='#007BC6',fontsize=11)
        ax.set_title(name,fontproperties=fp,fontsize=17,color='#00306C',weight='bold')
        ax.set_xticks(list(xx),['2','5','10','20']);ax.set_ylim(32,94);ax.set_xlim(-.22,3.25)
        ax.set_xlabel('공격 클래스당 라벨 사례 수',fontproperties=fp,fontsize=13)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    axs[0].set_ylabel('공격만 평가한 Macro-F1 (%)',fontproperties=fp,fontsize=13)
    axs[1].legend(loc='lower right',frameon=False)
    fig.tight_layout();path=g.ASSETS/'fewshot.png';fig.savefig(path,bbox_inches='tight',facecolor='white');plt.close(fig)
    (g.QA/'preliminary_plot_values.json').write_text(json.dumps(dict(source='docs/lablog/reports/20261021_references.tex, Appendix learning-curve tables',seeds=[43,44,45,46,47],n=[2,5,10,20],values=values),ensure_ascii=False,indent=2)+'\n')
    return path

def slides():
    a=[T(.72,1.13,11.95,'침입 탐지 시스템 (IDS) · 통신 기록의 특징으로 정상·공격 유형을 구분',20,color=GREY),
       T(.76,1.98,11.8,'새 공격의 정답 사례를 조금 확보했을 때',29,True,NAVY),
       T(.8,2.64,11.6,'기존 정상·공격 구분을 유지하면서 탐지기에 반영할 방법',24),
       B(.8,3.5,5.64,1.7,PALE),T(1.02,3.7,5.2,'XGB',24,True,NAVY),
       T(1.02,4.24,5.2,'확보한 라벨 사례로\n트리 모델 재학습',22),
       B(6.84,3.5,5.64,1.7,PALE),T(7.06,3.7,5.2,'TabPFN',24,True,BLUE),
       T(7.06,4.24,5.2,'사전학습 가중치는 고정\n라벨 사례를 context로 제공',22),
       T(.82,5.46,11.7,'Context = 현재 IDS 문제에서 참조할 특징·라벨 사례',20,color=NAVY),
       T(.82,6.13,11.7,'연구 질문  같은 사례로 얻는 탐지 품질·기존 성능 유지·처리 비용',22,True,TEAL)]
    add('소수 신규 공격 라벨을 반영하는 IDS 평가',a,
        '재학습 여부와 비용 우위는 별개 · 같은 사례와 평가 집합으로 비교',
        '이 발표는 기존 평가 시나리오를 이해시키는 데 중심을 둡니다. IDS는 flow 특징으로 정상과 공격 유형을 구분하는 분류기입니다. 새 공격의 라벨을 조금 확보했을 때 어떤 방식으로 반영할지 비교합니다. XGB는 여기서 선택한 기준선으로 매 단계 재학습합니다. 모든 트리 모델이 반드시 전체 재학습해야 한다는 주장은 아닙니다. TabPFN은 가중치를 고정하지만 context 전처리와 캐시 구성, query 추론에 비용이 듭니다. 기존 실행 설명, 별도 요청으로 수행한 극소수 선행 실험, 아직 실행하지 않은 보완 설계안을 구별합니다.',('cic','streams'))

    a=[T(.72,1.13,11.95,'CIC2018과 ToN-IoT의 사례 선택·평가를 각각 독립 수행',22,True,TEAL)]
    for x,name,tr,te,src in [(.72,'CIC2018','8,666,430','3,042,473','cic'),(6.94,'ToN-IoT','4,669,119','2,271,723','nf')]:
        a+=g.block(x,1.82,5.66,1.08,name+' 원본 트래픽','실험 환경에서 수집한 네트워크 기록',NAVY,18)
        a+=[A(x+2.57,3.0,.36,.35,down=True),T(x+.14,3.49,5.38,'NF-v3 flow 표 → 기존 정제·고정 분할',19,True,NAVY)]
        a+=g.block(x,4.12,2.77,1.17,'학습 후보 집합',tr+'행',BLUE,19)
        a+=g.block(x+2.93,4.12,2.73,1.17,'고정 test',te+'행',TEAL,19)
        a+=[T(x+.12,5.59,2.6,'초기·신규 라벨 사례 선택',16,color=BLUE),T(x+3.02,5.59,2.55,'예측 채점에만 사용',16,color=TEAL)]
    a+=[T(.84,6.23,11.72,'행의 출처와 분할 유지 · 다른 데이터셋의 공격을 가져와 합치는 과정 없음',21,True,NAVY)]
    add('데이터 출처 · 두 벤치마크를 섞지 않는 구성',a,
        '모델에 제공한 사례는 해당 벤치마크 train의 부분집합 · 특징값 합성 없음',
        '청중이 A와 B라는 별도 데이터셋으로 이해한 부분에 대한 첫 답변입니다. 두 열 사이에는 행 이동이 없습니다. CIC도 ToN도 자신의 NF-v3 자료와 고정 분할을 사용합니다. 원본은 통제된 실험망에서 수집한 트래픽이며 무가공 운영망을 뜻하지 않습니다. 실제 모델 입력은 기존 파이프라인의 46개 특징입니다. 상충 라벨을 가진 동일 특징 그룹을 제거했고 정제 후에도 원래 split membership을 유지했습니다. 표의 수는 context 크기가 아니라 정제 후 전체 후보 train/test 크기입니다. 이 오프라인 정제는 test 라벨도 활용한 기존 감사에 기반하므로 독립적인 배포 검증이라고 주장하지 않습니다. 현재 사례 집합을 만들 때 행 선택과 라벨 공개 조건을 설계했으며 다른 출처의 표를 합치거나 특징값을 합성하지 않았습니다.',('cic','nf'))

    a=[T(.72,1.1,11.9,'CIC2018 예시 · 모든 사례는 같은 CIC 학습 후보 집합에서 선택',20,color=GREY)]
    a+=g.block(.76,1.75,3.45,1.65,'초기 L₀ · 1,000개','정상 750\nBrute force 125 · DoS 125',NAVY,19)
    names=['DDoS','Web attacks','Infiltration','Bot'];colors=[BLUE,'2B8EA8',TEAL,AMBER]
    for i,(name,color) in enumerate(zip(names,colors)):
        x=4.57+i*2.03
        a+=g.block(x,1.75,1.81,1.65,name,'라벨 n개',color,18)
        if i<3:a.append(A(x+1.85,2.46,.16,.18))
    a+=[T(.8,3.82,11.6,'n = 신규 공격 클래스별 확보 라벨 수',25,True,NAVY),
        T(.82,4.43,11.65,'기존 실험  5 · 10 · 20 · 30 · 50 · 100개',22),
        T(.82,5.02,11.65,'n=100 예시  누적 1,000 → 1,100 → 1,200 → 1,300 → 1,400개',22,True,BLUE),
        T(.82,5.72,11.65,'“현재 데이터셋 B” = 현재까지 공개한 라벨 사례 Lₜ\n100개는 비교용 조건 · 하루 공격량이나 최적값을 뜻하지 않는 수',21)]
    add('현재 사례 집합 · 원본 train에서 라벨을 단계별 공개',a,
        '공격 도입 순서는 평가 조건 · 원본 날짜별 전체 트래픽 재생과 구별',
        'A에서 B를 어떻게 만드는지를 숫자로 설명하는 장입니다. A는 이 예시에서 CIC 학습 후보 집합입니다. B라고 불린 것은 별도 데이터셋이 아니라 현재까지 공개한 라벨 사례 L_t입니다. 초기 1000개에 같은 train의 DDoS 100개를 더하면 1100개가 됩니다. 모든 모델이 이 동일 행을 사용합니다. 100은 실생활 하루 발생량이나 최적값이 아니라 기존 발표에서 선택한 한 조건이며 5부터100까지 비교했습니다. 시간 정보는 공격 클래스 도입 순서의 근거였지만 행 전체를 시간순으로 재생한 실험은 아닙니다. 각 n 조건의 표본은 기록된 seed와 행 ID로 재현합니다. ToN의 자체 클래스 도입 순서는 마지막 부록에서 별도로 설명합니다.',('cic','nf'))

    a=[T(.74,1.1,11.94,'첫 신규 공격 100개를 확보한 예시 · 블록 하나는 100행',20,color=GREY)]
    for y,title,oldcount,total in [(1.85,'누적',10,'1,100행'),(3.43,'버퍼',9,'1,000행')]:
        a+=[T(.77,y,1.4,title,22,True,NAVY),B(2.1,y-.08,5.22,1.05,'FFFFFF','9AAEC1')]
        for j in range(oldcount):a.append(B(2.23+j*.437,y+.14,.39,.35,OLD))
        a.append(B(2.23+oldcount*.437,y+.14,.39,.35,NEW))
        a+=[T(2.24,y+.63,2.99,'기존 사례 + 신규 사례',15,color=GREY),T(5.43,y+.63,1.63,total,15,True,NAVY,align='right')]
    a+=[T(.8,4.84,6.48,'버퍼  기존 100행 삭제 → 신규 100행 전량 편입\n삭제 규칙  FIFO / 무작위 / 유사도',19),
        T(.82,5.79,6.42,'색 비율은 개념도 · 실제 클래스 구성은 정책별로 변화',15,color=GREY),
        A(7.47,2.75,.48,.42),
        T(8.12,1.78,4.45,'같은 선택 행을 두 모델에 제공',21,True,NAVY),
        B(8.12,2.4,4.43,1.18,PALE),T(8.33,2.55,4.0,'XGB  선택 행으로 재학습\nTabPFN  context·캐시 갱신',20),
        A(10.12,3.72,.35,.36,down=True),
        B(8.12,4.23,4.43,1.12,PALE),T(8.33,4.43,4.0,'같은 고정 test로 평가\nTabPFN 가중치 FT 없음',20),
        T(.82,6.38,11.6,'확보한 라벨 수 Lₜ와 실제 보관량 Mₜ를 따로 기록',23,True,TEAL)]
    add('누적·고정 버퍼 · 같은 사례로 갱신 방식 비교',a,
        'XGB에도 동일 버퍼 제공 · 사례 구성의 효과와 모델 효과를 분리',
        '테두리는 실제 모델에 제공되는 학습 집합 또는 context의 범위입니다. 회색은 기존 사례, 주황은 신규 라벨 사례입니다. 누적은1100행을 모두 사용하고 버퍼는100행을 삭제해1000행을 유지합니다. 그림은 삭제 정책의 실제 클래스별 결과가 아닌 개념도입니다. 기존 정책은 신규 사례를 전량 편입하고 이미 표현된 클래스 최소1행을 보존했습니다. EXP85에서 XGB도 PFN의 각 정책과 같은 행 ID 및 순서를 받아 모든 단계에서 새로 학습했습니다. 모델 효과를 설명할 때 PFN 버퍼와 XGB 누적만 비교해서는 안 됩니다. TabPFN은 추가 가중치 학습 없이 context에 따른 전처리와 캐시를 준비합니다. 이 동작에도 계산 비용이 있습니다.',('streams',))

    a=[T(.74,1.1,11.92,'기존 실험의 평가 범위 · 최종 품질과 갱신 과정의 측정을 구분',21,color=GREY)]
    rows=[('탐지 품질','최종 모델 · 해당 벤치마크 전체 test','전체 Macro-F1 · 신규 공격군 평균 F1'),
          ('기존 클래스 유지','초기 정상·공격 2종의 고정 평가 패널','같은 표본에서 최종 F1 − 초기 F1'),
          ('갱신 비용','2단계 이후 선택·삭제·fit/cache 시간 합계','초기 준비·모든 추론과 별도로 집계')]
    for i,(head,one,two) in enumerate(rows):
        y=1.85+i*1.45
        a+=[B(.8,y,2.8,1.13,NAVY),T(.98,y+.32,2.42,head,23,True,'FFFFFF'),T(3.94,y+.08,8.45,one,22,True,NAVY),T(3.94,y+.63,8.45,two,20)]
    a+=[T(.84,6.36,11.7,'신규 F1도 정상·기존 공격이 함께 있는 test의 오탐을 포함해 계산',21,True,TEAL)]
    add('기존 실험의 측정 항목과 해석 범위',a,
        '최종 F1만으로 갱신 중 적응을 설명할 수 없으며 갱신 시간은 전체 처리 비용과 별개',
        'CIC 최종전체test는3042473행,ToN은2271723행입니다. 신규 공격군 평균은 최종 혼동행렬에서 CIC4개,ToN7개 공격F1을 같은 비중으로 평균한 값입니다. 신규공격행만 따로 잘라정상오탐을제외한점수가 아닙니다. 기존 유지지표는 클래스당최대2000개 고정패널을 사용했고 초기3클래스의최종−초기차이입니다. 이패널과전체test의모집단은서로다릅니다. 기존갱신비용은초기와모든추론을제외한후속업데이트합계이며운영전체비용의우위로해석하지않습니다. 보완할부분은나중에도입된공격도다음업데이트에서추적하고같은query량을전제로추론비용까지보는것입니다.')

    a=[T(.74,1.08,11.9,'별도 선행실험 · 모든 공격에 같은 수의 라벨 제공 · 정상 제외 · 5 seeds',19,color=GREY),
       fig(learning_plot(),.72,1.65,11.89,3.85),
       T(.85,5.73,11.65,'소수 사례로 공격 유형 구분 가능 · 평균 격차는 데이터·사례 수에 따라 변화',21,True,NAVY),
       T(.85,6.27,11.65,'다음 확인  정상·기존 공격과 경쟁할 때의 신규 공격 성능과 유지',21,True,TEAL)]
    add('극소수 선행실험 · 분류 가능성 확인과 적용 범위',a,
        '공격만 구분한 결과와 정상 포함 IDS 성능을 구별 · 우세한 표본 수만 선택하지 않는 설계',
        '이 선행실험은 기존 평가시나리오 설명 요청과 별도로, 극소수context로분류가가능한지확인하라는요청에대한답변입니다. 모든공격class에동일n행을제공했고공격test만평가했습니다. 두모델은동일행을사용했고seed43부터47의평균과표준편차를표시합니다. 이후더큰n으로곡선을연장했지만이장은극소구간의전제확인만보여줍니다. CICn10에서80.37대80.58,ToNn10에서57.88대60.46입니다. 상대격차와실용적으로충분한탐지성능은다릅니다. 정상거부능력이나기존공격유지를입증하지않습니다. n1에서기존XGB설정상분할이없었던문제때문에이그림은n2부터표시했고작은표본설정진단도필요합니다. 자료: docs/lablog/reports/20261021_references.tex §0 및학습곡선부록.')

    a=tag('보완 설계안 · 미실행')
    for x,head,color in [(.76,'관점 1  라벨 가용성',BLUE),(6.88,'관점 2  갱신 과정',TEAL)]:
        a+=[B(x,1.75,5.68,.6,color),T(x+.18,1.87,5.32,head,23,True,'FFFFFF')]
    a+=[T(.96,2.64,5.28,'LotBoNC에서 참고',22,True,NAVY,url=g.SOURCES['lotbonc'][2]),
        T(.96,3.22,5.28,'Known·novel의 구분\n라벨 공개 여부와 클래스 불균형',21),
        T(.96,4.37,5.28,'우리 비교\n새 공격 라벨 2·5·10·20개를 제공\n정상·기존 공격과 함께 분류\n모두 보관해 라벨 효율부터 확인',20),
        T(.96,6.01,5.28,'비라벨 신규 군집 발견과는 다른 과제',17,True,BLUE),
        T(7.08,2.64,5.28,'Rethinking에서 참고',22,True,NAVY,url=g.SOURCES['streams'][2]),
        T(7.08,3.22,5.28,'예측 → 라벨 공개 → 갱신\n메모리 정책·갱신·추론 비용 구분',21),
        T(7.08,4.37,5.28,'우리 비교\n같은 라벨 이력을 단계별로 공개\n공격별 적응과 이후 성능 유지 추적',21),
        T(7.08,6.01,5.28,'통제한 라벨 공개 순서 · 원본 재생과 구분',17,True,TEAL)]
    add('보완 설계 · 라벨 가용성과 갱신 과정의 두 관점',a,
        '두 논문을 그대로 재현하는 계획이 아닌 정보 조건과 평가 순서의 분리',
        '두논문은다른문제를다루므로실험두개를그대로복제하지않습니다. LotBoNC는label없는새botnet군집을찾았지만XGB와TabPFN은여기서양의라벨사례를제공한분류기입니다. 따라서가져오는것은known/novel구성및라벨접근조건을명확하게쓰는방식입니다. Rethinking은예측후라벨을받고갱신하는순서와메모리및인코딩비용의구분을참고합니다. 관점1은최종접근가능라벨이정해진상태의분류효율이고관점2는같은최종라벨을순서대로받는동안의변화입니다. 이제안은기존실험과같은결과라고표현하지않으며새로실행하지않았습니다. XGB만으로모든스트림학습기를대표한다고주장할수없고실제온라인우위에는적합한스트림기준선이추가로필요합니다.',('lotbonc','streams'),section='보완 설계안')

    a=tag('보완 설계안 · 미실행')
    a+=[T(.78,1.74,11.82,'공통 조건  같은 초기 사례·신규 행·최종 라벨 예산·보관 정책',22,True,NAVY),
        T(.82,2.49,2.55,'관점 1  한 시점',22,True,BLUE),T(3.48,2.49,8.95,'초기 + 신규 공격별 n개를 모두 보관 → 분류',21),
        line(.8,3.12,11.72),T(.82,3.52,2.55,'관점 2  공개 과정',22,True,TEAL)]
    for j,label in enumerate(['DDoS 2개','누적 5개','누적 10개','누적 20개']):
        x=3.48+j*2.14
        a+=g.chip(x,3.44,1.91,label,fill=PALE,color=NAVY,size=18)
        if j<3:a.append(A(x+1.94,3.58,.17,.18))
    a+=[T(3.5,4.12,8.95,'각 갱신 뒤 고정 panel 평가 → 다음 공격 도입',20),
        T(.84,4.96,11.65,'현재 공격  라벨 증가에 따른 F1·recall\n기존 공격  후속 갱신에서의 성능 변화 · 정상 오탐률',22),
        T(.84,6.13,11.65,'동일 품질에 필요한 라벨 수·갱신 비용·같은 query량의 총 처리 비용 비교',20,True,TEAL)]
    add('보완 설계의 공통 예산과 단계별 평가',a,
        '성능 차이를 만드는 조건 탐색보다 차이가 생기는 공격·정보·비용 조건의 확인',
        '이장은아직실행하지않은구체안입니다. 관점1은동일한초기1000행과각신규공격n행을한번에관찰한상태를비교합니다. 관점2는같은행을2,5,10,20의누적라벨수로공개하고같은최종할당에맞춥니다. 2에서5가될때3개를추가하며5개를새로더하는뜻이아닙니다. 새train기록을라벨공개전에예측한결과와갱신후고정test패널점수는별도집계입니다. 새로운이름의출력class가없을때의0점은구조적으로결정되므로0에서오른것을noveldetection성과로삼지않습니다. 핵심곡선은라벨확보후시작합니다. 기존공격은초기두종뿐아니라중간에도입된공격까지다음갱신들에서추적합니다. 누적양쪽의최종행이같으면최종성능차이는순서효과의증거가아니므로과정점수와비용이필요합니다. 보존량효과는같은획득라벨pool에서따로비교합니다. 현재1000버퍼를기준으로추가크기민감도는별도설정하며이자료작성으로실험을실행하지않습니다.',('streams',),section='보완 설계안')

    # Existing matched cumulative results provide an explicitly historical reference.
    rows=list(csv.DictReader((g.ROOT/'docs/lablog/data/exp85_summary.csv').open()))
    a=[T(.76,1.1,11.9,'기존 EXP85 · 신규 공격당 10개 · 누적 · 3 seeds · 최종 전체 test',20,color=GREY)]
    for x,ds,name in [(.8,'cic2018','CIC2018'),(6.9,'toniot','ToN-IoT')]:
        a+=[T(x,1.82,5.55,name,25,True,NAVY)]
        vals=[]
        for model,y,color in [('xgb',2.6,INK),('pfn',3.6,BLUE)]:
            rr=next(r for r in rows if r['dataset']==ds and r['n']=='10' and r['model']==model and r['policy']=='accumulate')
            val=float(rr['macro_f1_pct_mean']);sd=float(rr['macro_f1_pct_sd']);vals.append(rr)
            a+=[T(x,y,1.22,'XGB' if model=='xgb' else 'TabPFN',20,True,color),B(x+1.48,y,2.48*val/100,.36,color),T(x+4.1,y-.02,1.36,f'{val:.2f}',24,True,color,align='right'),T(x+4.1,y+.47,1.36,f'± {sd:.2f}',15,color=GREY,align='right')]
        a+=[line(x+1.48,4.39,2.48),T(x+1.43,4.52,.4,'0',12,color=GREY),T(x+3.59,4.52,.4,'100',12,color=GREY,align='right'),T(x,5.12,5.55,'후속 갱신 합계  XGB / TabPFN',18,color=GREY),T(x,5.58,5.55,f'{float(vals[0]["update_seconds_mean"]):.2f}초 / {float(vals[1]["update_seconds_mean"]):.2f}초',23,True,NAVY)]
    a+=[T(.82,6.39,11.6,'단위  Macro-F1 (%) · 오차는 seed 표준편차 · 시간은 추론을 제외한 합계',17,color=GREY)]
    add('부록 · 기존 순차 실험의 비교 기준',a,
        '이미 수행한 비교의 참조값 · 두 관점의 보완 설계를 실행한 결과와 구별',
        '자료는EXP85의같은행대조를포함한통합CSV입니다. 이장은누적n10의전체MacroF1과후속갱신시간을기존결과참조로보여줍니다. 시드43,44,45의평균및표본표준편차이며새프로토콜을돌린값이아닙니다. 두모델의격차가크지않다는것과갱신비용차이를함께보되해당시간은초기와모든추론을제외했습니다. PFN추론이더오래걸릴수있기때문에총운영비용절감이라고주장하지않습니다. 원시근거docs/lablog/data/exp85_summary.csv;상세보고docs/lablog/reports/20261021_exp85.tex.',section='기존 결과 부록')

    a=[T(.76,1.12,11.9,'ToN-IoT 자체 train에서 선택 · CIC2018과 행 결합 없음',21,True,TEAL)]
    stages=[('초기','정상 750 + Scanning 125 + DoS 125','1,000'),('2단계','Injection n + DDoS n','1,000 + 2n'),('3단계','Password n','1,000 + 3n'),('4단계','XSS n','1,000 + 4n'),('5단계','Ransomware n + Backdoor n','1,000 + 6n'),('6단계','MitM n','1,000 + 7n')]
    a+=[B(.8,1.78,11.7,.5,NAVY),T(1,1.88,1.6,'공개 단계',18,True,'FFFFFF'),T(3.0,1.88,6.3,'해당 단계의 라벨 공급량',18,True,'FFFFFF'),T(10.04,1.88,2.1,'누적 보유량',18,True,'FFFFFF',align='right')]
    for j,(s,l,n) in enumerate(stages):
        y=2.43+j*.51
        a+=[B(.8,y-.04,11.7,.48,PALE if j%2==0 else 'FFFFFF'),T(1,y,1.65,s,18),T(3.0,y,6.4,l,18),T(9.9,y,2.25,n,18,True,NAVY,align='right')]
    a+=[T(.85,5.78,11.5,'n=100  최종 누적 1,700개 / 고정 버퍼 1,000개',22,True,BLUE),T(.85,6.34,11.5,'최종 고정 test  2,271,723개 · 단계 순서와 test 표본 수 정렬은 별개',18,color=GREY)]
    add('부록 · ToN-IoT의 단계와 공급량',a,
        '단계별 공급량과 실제 보관량을 구분 · 신규 공격은 총 7개 클래스',
        'CIC의과정을ToN에같은공격명으로복사한것이아닙니다. ToN의기존초기클래스는정상ScanningDoS이고새공격이동시에들어오는단계는해당두class모두n행씩제공합니다. 신규7개class이므로n100에서1700행이고n10에서1070행입니다. 고정버퍼는각갱신에서새행을모두넣고동일수기존행을삭제해1000행을유지합니다. 위표는공급단계순서이며평가표본수내림차순의성능표와용도가다릅니다. 데이터출처와정제규칙은본문2장과동일합니다.',('nf',),section='설정 부록')

def main():
    slides()
    revised_notes={
      5: 'CIC의 최종 전체 test는 3,042,473행이고 ToN은 2,271,723행입니다. 신규 공격군 평균은 최종 혼동행렬에서 CIC 4개, ToN 7개 신규 공격의 F1을 같은 비중으로 평균한 값입니다. 신규 공격 행만 잘라 정상 오탐을 제외한 점수가 아닙니다. 기존 성능 유지는 클래스당 최대 2,000개 고정 패널에서 초기 세 클래스의 최종 F1과 초기 F1을 비교했습니다. 이 패널과 전체 test는 모집단이 다릅니다. 기존 갱신 비용은 초기 준비와 모든 추론을 제외한 후속 갱신 시간의 합계입니다. 이를 전체 운영 비용의 우위로 해석하지 않습니다. 보완할 부분은 중간에 도입된 공격도 다음 갱신에서 추적하고, 동일한 query 수를 전제로 추론 비용까지 보는 것입니다.',
      6: '이 선행실험은 기존 평가 시나리오의 설명 요청과 별도로, 극소수 context로 분류가 가능한지 확인하라는 요청에 답한 것입니다. 모든 공격 클래스에 동일한 n개 사례를 제공하고 공격 test만 평가했습니다. 두 모델은 동일한 행을 사용했고, seed 43부터 47까지의 평균과 표본 표준편차를 표시했습니다. 이후 더 큰 n으로 곡선을 연장했지만 이 장은 극소 구간의 전제만 보여줍니다. CIC의 n=10에서 XGB와 TabPFN은 각각 80.37%, 80.58%, ToN에서는 57.88%, 60.46%였습니다. 상대적인 격차와 실용적으로 충분한 탐지 품질은 다릅니다. 이 결과는 정상 거부 능력이나 기존 공격의 유지 성능을 입증하지 않습니다. n=1에서 기존 XGB 설정상 실제 트리 분할이 없었던 문제가 있어 이 그림은 n=2부터 표시했습니다. 작은 표본에 적합한 XGB 설정 진단도 필요합니다. 근거는 docs/lablog/reports/20261021_references.tex의 0절과 학습곡선 부록입니다.',
      7: '두 논문은 다른 문제를 다루므로 두 실험을 그대로 복제하지 않습니다. LotBoNC는 라벨 없는 새 botnet 군집을 찾았지만, 여기서는 XGB와 TabPFN에 새 공격의 정답 사례를 제공합니다. 참고하는 부분은 known/novel 구성과 라벨 접근 조건을 명확하게 정의하는 방식입니다. Rethinking에서는 예측 후 라벨을 받고 갱신하는 순서, 메모리 유지와 context 인코딩 비용의 구분을 참고합니다. 관점 1은 최종 접근 가능한 라벨이 정해진 상태의 분류 효율이며, 관점 2는 같은 최종 라벨을 순서대로 받는 동안의 변화입니다. 이 제안은 아직 실행하지 않았고 기존 결과와 구분합니다. XGB만으로 모든 스트림 학습기를 대표할 수는 없으므로, 실제 온라인 우위에 관한 주장은 적합한 스트림 기준선을 추가한 뒤 검토해야 합니다.',
      8: '이 장은 아직 실행하지 않은 구체안입니다. 관점 1은 동일한 초기 1,000개와 각 신규 공격의 n개 사례를 한 번에 관찰한 상태를 비교합니다. 관점 2는 같은 행을 2, 5, 10, 20개의 누적 라벨 수로 공개해 동일한 최종 할당에 맞춥니다. 2개에서 5개가 될 때는 3개를 추가합니다. 새 train 기록의 라벨 공개 전 예측과 갱신 후 고정 test 패널 점수는 별도 집계합니다. 출력 클래스가 아직 없는 상태의 새 공격 분류 0점은 구조적으로 결정되므로, 0점에서 올랐다는 사실을 신규 공격 발견의 성과로 삼지 않습니다. 핵심 적응 곡선은 양의 라벨을 확보한 뒤 시작합니다. 기존 공격은 초기 두 종뿐 아니라 중간에 도입한 공격까지 후속 갱신에서 추적합니다. 누적 모델의 최종 행이 같다면 최종 점수만으로 순서의 효과를 설명하기 어려우므로 중간 점수와 비용이 필요합니다. 보관량의 효과는 같은 확보 라벨 pool에서 따로 비교합니다. 현재 1,000개 버퍼를 기준으로 추가 크기 민감도는 별도 설정이며, 이 자료를 작성하면서 실험을 실행하지는 않았습니다.',
      9: 'EXP85의 동일 행 대조를 포함한 통합 CSV에서 누적 조건 n=10을 가져왔습니다. 이 장은 최종 전체 Macro-F1과 후속 갱신 시간을 기존 결과의 참조값으로 보여줍니다. seed 43, 44, 45의 평균과 표본 표준편차이며, 새 프로토콜을 실행한 결과가 아닙니다. 두 모델의 품질 격차와 갱신 비용 차이를 함께 읽되 해당 시간에서 초기 준비와 모든 추론이 제외됐다는 점을 설명합니다. PFN의 추론이 더 오래 걸릴 수 있으므로 전체 운영 비용 절감을 주장하지 않습니다. 수치 출처는 docs/lablog/data/exp85_summary.csv, 상세 보고서는 docs/lablog/reports/20261021_exp85.tex입니다.',
      10: 'CIC의 과정에 ToN 사례를 합친 것이 아닙니다. ToN의 초기 클래스는 정상, Scanning, DoS입니다. 두 공격이 함께 들어오는 단계에서는 각 클래스에 n개씩 제공합니다. 신규 공격은 총 7개이므로 n=100에서 최종 1,700개, n=10에서 1,070개가 됩니다. 고정 버퍼는 각 갱신에서 새 사례를 모두 넣고 같은 수의 기존 사례를 삭제해 1,000개를 유지합니다. 이 표는 공급 단계 순서이므로 평가 표본 수 내림차순으로 정렬하는 성능표와 용도가 다릅니다. 데이터 출처와 정제 규칙은 본문 2장과 같습니다.'}
    for number,notes in revised_notes.items():
        scene=g.SCENES[number-1]
        scene['notes']='발표 설명\n'+notes+'\n\n원문 근거\n'+scene['notes'].split('\n\n원문 근거\n',1)[1]
    # 교수 1의 설명 요청에는 기존 실험의 문제·출처·표본 공개·갱신 흐름만 사용한다.
    # 극소수 선행실험과 문헌 검토는 별도 자료에서 먼저 검토한다.
    g.SCENES[:]=g.SCENES[:4]
    g.validate_scenes();g.pptx()
    p=Presentation(g.OUT.with_suffix('.pptx'));p.core_properties.title='IDS 평가 시나리오 · 데이터 구성과 갱신 절차';p.core_properties.subject='기존 평가 시나리오의 데이터 출처, 단계별 라벨 공개, 누적·버퍼 비교';p.save(g.OUT.with_suffix('.pptx'))
    checks=g.pdf()
    with zipfile.ZipFile(g.OUT.with_suffix('.pptx')) as z:
        assert not any(b'<a:outerShdw' in z.read(n) for n in z.namelist() if n.endswith('.xml'))
    (g.QA/'scenes.json').write_text(json.dumps(g.SCENES,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(date='2026-10-23',purpose='Explain the existing evaluation scenario and resolve the A/B-dataset misunderstanding',slides=4,main_slides=[1,2,3,4],inputs=['docs/slides/kor/20261021_Meeting_ko.pptx','docs/lablog/reports/20261021_exp84.tex','docs/lablog/reports/20261021_exp85.tex'],source_links=g.SOURCES,manuscript='docs/latex/main.tex',no_experiments_run=True,review_order='Professor 2 reviews all three artifacts before the evaluation explanation is shared with Professor 1')
    (g.QA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (g.QA/'validation.json').write_text(json.dumps(dict(passed=True,pages=checks,speaker_notes=4,pptx_shadow_count=0,pdf_sha256=g.sha(g.OUT.with_suffix('.pdf')),pptx_sha256=g.sha(g.OUT.with_suffix('.pptx')),visual_review_pending=True),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(pdf=str(g.OUT.with_suffix('.pdf')),pptx=str(g.OUT.with_suffix('.pptx')),slides=4),ensure_ascii=False))

if __name__=='__main__':main()
