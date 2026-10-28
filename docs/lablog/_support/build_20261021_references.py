"""Build the Korean review in the user's 0--4 order (revised 2026-10-22).

Source papers and experiment outputs are read-only. No manuscript/slides are
changed, and compiled report PDFs are temporary.
"""
from pathlib import Path
import sys, json, hashlib, datetime
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/lablog.py').exists())
sys.path.insert(0,str(ROOT/'scripts/common'))
from lablog_format import esc, table
from lablog import build_report, check

BASE=ROOT/'docs/lablog'
EVID=BASE/'evidence/references/20261021'
REPORT=BASE/'reports/20261021_references.tex'
RUN=ROOT/'tabpfn/results/v7/exp87/20261022_163739_full_learning_curve_five_seeds_c59213f5'
EXT=ROOT/'tabpfn/results/v7/exp87/20261022_164549_class_capped_extension_seed43_43f6dd6c'
ASSET='figures/references_20261022/'
parts=[]
SOURCES={
 'survey':('Tabular Foundation Models: A Systematic Survey','https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7531718'),
 'boost':('Prior-Fitted Networks Scale to Larger Datasets When Treated as Weak Learners','https://arxiv.org/abs/2503.01256'),
 'local':('Retrieval & Fine-Tuning for In-Context Tabular Models','https://papers.nips.cc/paper_files/paper/2024/file/c40daf14d7a6469e65116507c21faeb7-Paper-Conference.pdf'),
 'dist':('Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment','https://arxiv.org/abs/2605.04363'),
 'tabfm':('TabFM: A Zero-Shot Foundation Model for Tabular Data','https://arxiv.org/abs/2609.37959'),
 'pfn35':('TabPFN-3.5: Technical Report','https://arxiv.org/abs/2609.17895'),
 'imbalance':('Correcting Class Imbalance in Prior-Data Fitted Networks for Tabular Classification','https://arxiv.org/abs/2605.21742'),
 'ensemble':('Ensembling Tabular Foundation Models — A Diversity Ceiling And A Calibration Trap','https://arxiv.org/abs/2605.18696'),
 'iot':('Optimizing IoT Intrusion Detection with Tabular Foundation Models for Smart City Forensics','https://arxiv.org/abs/2604.11394'),
 'tabarena':('TabArena: A Living Benchmark for Machine Learning on Tabular Data','https://arxiv.org/abs/2506.16791'),
 'lotbonc':('LotBoNC: Novel Botnet Traffic Classification under Long-tailed Distributions','https://doi.org/10.1145/3779208.3785276'),
 'streams':('Rethinking Tabular Foundation Models On Data Streams','https://arxiv.org/abs/2610.05352'),
 'icd':('In-Context Data Distillation with TabPFN','https://arxiv.org/abs/2402.06971'),
}
def raw(s): parts.append(s+'\n')
def section(s): raw(r'\section{'+esc(s)+'}')
def sub(s): raw(r'\subsection{'+esc(s)+'}')
def para(title,text): raw(r'\paragraph{'+esc(title)+'} '+esc(text)+'\n')
def p(s): raw(esc(s)+'\n')
def discussion(s): para('Discussion.',s)
def tab(rows): raw(table([[esc(str(v)) for v in row] for row in rows]))
def link(url,label): return r'\href{'+url+'}{'+esc(label)+'}'
def cite(key,where):
    title,url=SOURCES[key]
    raw(r'{\small\textbf{원문 확인:} '+link(url,title)+'; '+esc(where)+r'.}\par')
def fig(path,caption,width='1.0',height='.65'):
    raw(r'\begin{figure}[H]\centering\includegraphics[width='+width+r'\linewidth,height='+height+r'\textheight,keepaspectratio]{'+path+r'}\caption{'+caption+r'}\end{figure}')
def paperfig(key,number,page,caption,width='.86',height='.47'):
    source='streams' if key=='streams_cache' else key
    fig(ASSET+key+'.pdf',esc(caption)+' '+link(SOURCES[source][1],f'원문 {number}, PDF p.{page}')+' 발췌.',width,height)
def steps(items):
    raw(r'\begin{enumerate}[leftmargin=*,itemsep=3pt]')
    for item in items: raw(r'\item '+esc(item))
    raw(r'\end{enumerate}')

def results():
    section('극소 샘플로 실험한 결과')
    p('소수 라벨만으로 공격 유형을 구분할 수 있는지는 context 갱신 실험에 앞서 확인할 전제다. 이를 확인하기 위해 CIC2018과 ToN-IoT를 각각 사용하여 모든 공격 클래스에서 같은 수의 실제 사례를 선택하고 XGBoost (XGB)와 TabPFN을 비교했다. 처음에는 공격당 1–20개를 살폈고, 이후 두 모델이 언제 비슷해지고 추가 표본의 이득이 얼마나 남는지 확인하도록 학습곡선의 오른쪽을 연장했다. 많은 사례를 다시 요구하는 연구로 바꾼 것이 아니라, 극소수 구간의 결과를 해석할 비교 범위를 확보한 것이다.')
    p('본 절은 EXP86의 후속인 EXP87 완료 결과를 사용한다. 연구 버전은 v7이고, 균형 조건 200개 실행이 모두 완료되어 모든 점에서 seed 43–47의 5회 평균과 표본 표준편차를 비교한다. XGB가 분할을 만들지 못했던 1개 조건을 제외해 2개부터 표시한다. 학습과 평가 모두 정상(benign)을 제외한 공격 유형 분류이며, 새 공격을 순차적으로 추가하는 실험은 아니다.')
    sub('공통 설정과 학습곡선')
    tab([['데이터','공격 클래스 수','클래스당 n / 전체 제공량','고정 공격 test'],
         ['CIC2018','6','2, 5, 10, 20, 40, 80, 160, 320, 450 / 6n행','390,270'],
         ['ToN-IoT','9','2, 5, 10, 20, 40, 80, 160, 320, 640, 1,280, 2,151 / 9n행','797,421']])
    p('같은 seed·n에서 두 모델에 동일한 train 행을 제공했고, 작은 n의 행은 큰 n에 포함되도록 선택했다. TabPFN은 기존 v3 checkpoint와 4 estimators를 사용하며, 가중치 fine-tuning (FT) 없이 모든 선택 행을 context로 제공했다. XGB는 300개 트리, 깊이 8 등의 기존 설정으로 각 n에서 새로 학습했다. 점수는 각 seed에서 고정 공격 test의 클래스별 F1을 동일 가중치로 평균한 Macro-F1이며, 서로 다른 seed의 예측을 합산하지 않았다.')
    raw('데이터의 수집·배포 배경은 '+link('https://www.unb.ca/cic/datasets/ids-2018.html','CSE-CIC-IDS2018 공식 설명')+' 및 '+link('https://staff.itee.uq.edu.au/marius/NIDS_datasets/','UQ의 NIDS 데이터·NetFlow 배포 설명')+'에서 확인할 수 있다. 실제 실험은 프로젝트의 기존 정제·분할을 유지했으며, 위 숫자는 원본 전체 크기가 아닌 이번 공격-only 평가 범위다.\n')
    fig('figures/exp87_quality_cic2018.pdf',esc('CIC2018 학습곡선. 점의 숫자는 5-seed 평균 F1(%), 오차 막대는 표본 표준편차, 하단은 TabPFN−XGB(%p)다. x축은 로그 간격이다.'))
    fig('figures/exp87_quality_toniot.pdf',esc('ToN-IoT 학습곡선. CIC와 동일한 집계·표시 기준이다. 두 모델의 점수가 가까워지는 시점과 각각의 성능이 포화되는 시점은 구분한다.'))
    # Class results deliberately precede detailed numerical tables.
    sub('클래스별 결과: 평균 점수 뒤에 남는 차이')
    p('아래 셀의 배경색과 윗줄 숫자는 TabPFN의 클래스별 F1(%)이다. 아랫줄 괄호는 동일한 5개 seed에서 계산한 TabPFN−XGB 차이(%p)이며, 양수는 짙은 녹색, 음수는 짙은 적색으로 표시했다. 차이의 부호가 흐려지지 않도록 배경색은 밝은 청색 범위로 제한했다. 행은 고정 test의 클래스별 표본 수 내림차순이고, 대괄호에 평가 표본 수를 기재했다.')
    fig(ASSET+'class_cic2018.pdf',esc('CIC2018: TabPFN F1과 두 모델 차이. 셀마다 평균 F1 / (평균 F1 차이)를 표시한다.'))
    p('CIC에서는 공격당 10개만으로도 DDoS·bot·brute force의 TabPFN F1이 약 99–100%였지만 web attacks는 4.3%였다. 같은 n에서 infiltration은 TabPFN 84.3%로 XGB보다 3.8%p 높았고, web attacks는 XGB보다 3.1%p 낮았다. 따라서 전체 평균이 비슷하다는 결과와 모든 공격이 충분히 분류된다는 주장은 다르다. Web의 test는 127개로 작고 낮은 precision의 영향도 받으므로, 후속 개선은 recall과 혼동 대상을 함께 확인해야 한다.')
    fig(ASSET+'class_toniot.pdf',esc('ToN-IoT: TabPFN F1과 두 모델 차이. 이 표는 자연 빈도의 test에 대한 평가이며, 학습 context만 클래스별 균형이다.'))
    p('ToN의 공격당 5개 조건에서 TabPFN은 ransomware와 scanning에서 각각 약 19.2·20.8%p 높은 F1을 보였지만 password는 3.7%p 낮았다. 160개에서도 MITM은 TabPFN 35.5%, XGB 43.2%로 남았고, ransomware는 TabPFN이 약 15.7%p 높았다. 작은 context의 장점과 약점은 공격별로 달랐으며, 모든 tail 공격을 하나의 평균으로 설명하기 어렵다. Context의 클래스 수가 균형이어도 이런 차이가 남았으므로 불균형 비율만을 원인으로 단정하지 않는다.')
    sub('비용: 준비·학습/캐시 구성과 추론을 분리')
    fig('figures/exp87_cost.pdf',esc('5개 seed의 평균 비용. 준비는 새 모델 생성과 XGB fit 또는 PFN context/cache 구성을 포함한다. 추론은 각 데이터의 고정 공격 test 전체이며, 후속 단계의 갱신 시간은 아니다.'))
    p('CIC의 공격당 10개 조건에서 XGB와 TabPFN의 준비 시간은 각각 약 0.290·0.312초였고, 390,270개 공격 test 추론은 0.142·11.759초였다. ToN의 160개 조건에서는 준비 1.277·0.333초, 797,421개 test 추론 0.545·27.003초였다. 이 측정에서 TabPFN의 준비 비용은 비교적 작았지만 전체 추론은 XGB보다 오래 걸렸다. 준비 구간에서 빠진 계산이 추론 구간에 포함될 수도 있으므로, FT가 없다는 이유만으로 전체 운영 비용이 작다고 결론 내리지 않는다.')
    sub('결론과 다음 평가에 사용할 해석')
    p('두 모델은 적은 공격 사례만으로도 상당수 유형을 구분했다. CIC에서는 n=10·20의 평균 차이가 약 0.21·0.16%p로 작았지만, n=10에서 450까지 XGB는 80.37%에서 86.25%, TabPFN은 80.58%에서 85.42%로 더 올랐다. ToN도 n=160의 82.76·84.95%에서 n=2,151의 92.75·91.12%까지 개선됐다. 따라서 CIC 10개와 ToN 160개를 확정적인 수렴점으로 쓰기는 어렵고, 해당 지점은 모델 간 격차와 표본 효율을 비교할 기준점으로 해석한다.')
    p('균형 곡선의 끝점은 성능 포화가 아니라 희귀 공격 train 가용량인 CIC web 450개, ToN ransomware 2,151개로 정해졌다. 이를 넘긴 단일 seed 탐색에서는 희귀 공격을 전부 유지하고 다른 공격만 늘렸으므로 클래스 비율이 달라졌다. 해당 결과는 균형 5-seed 곡선에 이어 붙이지 않고 부록에 별도로 정리했다. 이 탐색에서도 큰 context가 TabPFN의 점수를 항상 높이지는 않았다.')
    p('이전 순차 시나리오의 신규 공격당 100개는 일부 쉽게 구분되는 공격에서 극소수 구간의 차이를 희석했을 가능성이 있다. 그러나 정상과 기존 공격이 함께 있는 조건으로 바뀌면 요구량도 달라지므로, 모든 공격에 100개가 충분했다고 일반화하지 않는다. 후속 시나리오를 10개 미만, 예를 들어 2·3·5·8개로 좁혀 보는 목적은 소수 사례에서 관찰된 TabPFN의 이득이 정상·기존 공격과 경쟁하는 조건에서도 유지되는지 검증하는 데 있다. 특히 n=5의 평균 차이인 CIC +2.55%p와 ToN +6.46%p는 이 가설의 근거이지만, n=2의 큰 변동과 클래스별 역전도 함께 고려한다.')
    p('다음 실험에서는 두 모델에 같은 초기 행·신규 행·버퍼 정책을 적용하고, 신규 공격 F1, 기존 공격의 전후 변화, 정상 오탐, 갱신과 추론 비용을 함께 비교하는 것이 적절하다. 결과에 따라 필요한 라벨 수를 줄일 수 있는 공격과 추가 표현·선택 개선이 필요한 공격을 구분한다. 이 문서 개정에서는 새 실험을 실행하지 않았다.')

def related():
    raw(r'\clearpage')
    section('TabPFN 계열 논문: 분류 체계와 관련 연구')
    sub('서베이 Figure 3을 읽는 두 축')
    p('표 데이터 파운데이션 모델(Tabular Foundation Model, TFM)은 사전학습 자료의 출처와 새 표에 지식을 전달하는 방식으로 나눌 수 있었다. 제공된 Systematic Survey의 Figure 3은 열에서 합성 prior와 실제 표를, 행에서 네 가지 전이 방식을 구분했다. 실제 표는 본문에서 정제된 표 모음·웹 수집 표·지식베이스로 더 세분했다. 따라서 TabPFN 이후 모델을 모두 같은 구조의 후속 버전으로 묶는 대신, 예측 시 무엇을 입력하고 무엇을 학습하는지 비교해야 한다.')
    paperfig('survey','Figure 3',6,'서베이의 기반 모델 분류도. 노드 크기는 인용 빈도, 색은 발표 시기이며 성능 순위가 아니다.','.88','.66')
    tab([['전이 방식','새 표에서 하는 일','이번 검토와의 연결'],
         ['In-context learning (ICL) / PFN','라벨 행을 context로 주고 고정 가중치로 query 예측','현재 TabPFN, TabFM 및 context·사후확률 확장'],
         ['Fine-tuned encoder','사전학습 표현을 가져와 새 표에서 가중치 최적화','가중치 FT와 context 변경의 구별'],
         ['Text generation','행을 텍스트로 직렬화하고 언어 생성으로 예측','이번 실험의 숫자 flow context와 입력 방식이 다름'],
         ['Weight generation / hypernetwork','입력 표에 맞는 작은 모델의 가중치를 생성','매번 context를 읽는 ICL과 추론 구조가 다름']])
    p('Prior-data fitted network (PFN)은 여러 생성 과제에서 예측 절차를 사전학습하고, 목표 표의 라벨 행을 context로 받아 예측하는 계열이다. 서베이는 146편 중 기반 모델 42개를 분류했고, 그중 합성 사전학습은 26개, ICL/PFN 계열은 28개였다. 검색 목록의 전체 건수와 최종 포함 논문 수는 구분한다. 아래에서는 모델 구조·앙상블, 네트워크 데이터, long-tail (LT) 또는 클래스 불균형, 기존 SOTA 비교 모델이라는 네 기준과 관련된 논문만 골랐다. TabFM·스트림 논문과 보안 응용은 서베이의 2026-09-15 수집 범위를 보완하는 별도 검토로 표시한다.')
    cite('survey','§3.1, Figure 3, pp.5–6; §4 및 §6.3')
    tab([['검토 대상','선정 이유','바꾸는 대상'],
         ['LoCalPFN / BoostPFN / DistPFN','기존 SOTA 비교군; 지역성·앙상블·불균형','query별 context와 FT / context ensemble / 출력 확률'],
         ['TabFM+ / 앙상블 분석','제안 모델의 앙상블 필요성','입력 view·결합 가중치 / 오류 다양성·보정'],
         ['TabPFN-3.5','기반 모델 교체의 효과를 분리할 최신 비교 후보','입력 표현·모델 용량·추론 구성'],
         ['Class imbalance correction','LT와 관련된 prior·판정 편향','context 비율·분류 임계값'],
         ['IoT forensic pipeline','실제 보안 데이터 응용','TabPFN 선별과 트리 앙상블의 연계'],
         ['LotBoNC / Streams / ICD','평가 시나리오·context 설계','각각 §3과 §4에서 상세 설명']])
    discussion('분류도는 연구 후보를 고르는 지도이며, 우리 데이터에서 우열을 정해 주지 않는다. 이번에는 새 checkpoint를 추가하는 것보다 극소수 라벨, 오류가 남는 공격, context의 구성과 비용에 직접 연결되는 방법을 우선한다. 아래의 수치는 각 원문의 자체 프로토콜 결과로, §0의 IDS Macro-F1과 직접 비교하지 않는다.')

    sub('[LoCalPFN] (NeurIPS, 2024)')
    para('목적과 방법.', '큰 표에서 무작위 context가 query 주변의 분류 경계를 충분히 담지 못하는 문제를 다뤘다. 먼저 query와 가까운 학습 행을 k-nearest neighbors (kNN)로 찾아 지역 context를 구성하고 TabPFN에 넣었다. 이어 학습 시에도 지역 이웃을 context와 query로 나누고 query의 예측 오차로 TabPFN 가중치를 FT했다. 그림 오른쪽은 한 기준점의 이웃을 공유해 학습 배치마다 검색하는 비용을 줄이는 구조다. FT 없이 검색만 하면 TabPFN-kNN이고, FT까지 포함한 것이 LoCalPFN이다.')
    paperfig('local','Figure 3',5,'왼쪽은 query별 이웃 검색, 오른쪽은 지역 context를 공유하는 fine-tuning 배치 구성이다.')
    para('실험과 결과.', 'TabZilla/OpenML의 95개 분류 자료를 사용했고, 47개 소규모와 48개 중·대규모 자료를 구분했다. 원문은 80:10:10 train/validation/test 분할과 10-fold 평가, 비교군의 validation AUC 기반 튜닝을 기술했다. 전체 자료의 평균 AUC는 TabPFN 0.867, TabPFN-kNN 0.891, XGB 0.892, LoCalPFN 0.908이었다. 개선에는 검색과 가중치 FT가 함께 기여했으며, 단순히 context를 작게 한 것만의 성과는 아니었다.')
    cite('local','§2.2–2.4, Figure 3 pp.3–5; §4.1–4.2 및 Table 1 pp.6–7')
    discussion('우리 연구에서는 query마다 필요한 공격 사례가 다를 수 있다는 관점을 참고할 수 있다. 다만 극소 라벨에서는 이웃 후보 자체가 적고, 비슷한 정상·공격이 겹치면 검색만으로 해결되지 않을 수 있다. 검색만 / 검색+FT를 분리하고, 같은 라벨 pool에서 클래스별 이득과 query별 cache·검색 비용을 확인해야 한다.')

    sub('[BoostPFN] (AISTATS, 2025)')
    para('목적과 방법.', '한 번에 큰 train 전체를 context로 쓰기 어려운 PFN을 여러 약한 예측기로 결합했다. 매 라운드에 train의 일부를 context로 뽑아 PFN 예측기를 만들고, 현재 결합 예측의 오차를 이용해 다음 context의 표본 가중치를 바꿨다. 이렇게 얻은 예측들을 학습한 결합 계수로 합쳤다. PFN 본체는 고정하며, 업데이트하는 것은 표본 선택 가중치와 앙상블의 결합 방식이다.')
    paperfig('boost','Figure 1',2,'서로 다른 context를 갖는 PFN 약한 예측기를 순차 결합한다.')
    para('실험과 결과.', '기존 OpenML 30개와 더 큰 30개 자료를 사용했고, 50:50 train/test 및 5개 seed로 평가했다. 큰 자료에서는 train 5,000·50,000개와 전체 자료 조건도 비교했다. Table 3의 train 5,000개 조건 평균 AUC는 XGB 0.849, TabPFN 0.885, 단순 bagging 0.874, BoostPFN 0.908이었다. 50,000개에서는 단일 TabPFN이 메모리 부족을 겪었고 BoostPFN은 0.910이었지만, 50,000개 초과 조건에서 XGB 0.916보다 BoostPFN 0.913이 낮아 모든 규모에서 우세하지는 않았다.')
    cite('boost','§4, Figure 1 pp.2–4; §6 및 Table 3 pp.5–6; 네트워크 예시 KDDCup99는 Appendix D')
    discussion('현재 앙상블에 참고할 것은 단순히 expert 수를 늘리는 대신 이전 expert가 틀린 공격 사례를 다음 context에 반영하는 방식이다. 다만 이 논문은 큰 라벨 pool의 활용이 목적이어서 라벨이 2–5개뿐인 조건과 다르다. 적은 예산에서는 오류가 상호 보완되는 context가 실제로 만들어지는지부터 확인하고, 반복 예측 비용을 포함해 단일 PFN과 비교해야 한다.')

    sub('[DistPFN] (ICML, 2026)')
    para('목적과 방법.', 'Context와 평가 자료의 클래스 비율이 달라질 때 PFN의 예측이 context 다수 클래스에 치우치는 현상을 다뤘다. Context에서 계산한 클래스 비율과 PFN이 예측한 분포의 비를 이용해 출력 확률을 재조정했다. 원문은 개별 query 예측과 query 배치의 평균 분포를 모두 사용할 수 있다고 설명했고, 실험 기본값은 배치 평균이었다. DistPFN-T는 두 분포의 차이에 따른 온도 조절을 추가했다. 어느 방식도 기본 절차에 PFN 가중치 FT를 요구하지 않았다.')
    paperfig('dist','Figure 4',4,'모델 출력 이후에 클래스 확률을 보정한다. 그림의 단일 사례 예시와 실험의 배치 평균 분포 사용을 구분한다.')
    para('실험과 결과.', 'OpenML 분류 253개에서 기본 분할과 label-shift 변형을 비교했다. §4.4는 train을 고정하고 test 분포를 바꾸는 구성을 기술했다. Table 5에서 shift 강도 β=2일 때 TabPFN-v2의 평균 accuracy는 0.759, DistPFN 적용 시 0.783, DistPFN-T 적용 시 0.789였다. Shift가 없는 조건에서는 세 방법이 0.818로 비슷했으므로, 효과의 핵심은 기본 분류력 향상보다 분포 차이에 대한 보정이었다.')
    cite('dist','§4.2–4.4, Eq.(3)–(5), Algorithm 1 pp.4–5; Table 5 p.6')
    discussion('우리의 균형 context와 자연 빈도의 test 사이에도 prior 차이가 생기므로 보정 대조군으로 관련이 높다. 그러나 특징이 겹치는 tail 공격에 새로운 구분 정보를 만들어 주지는 않는다. 적용 시 query 배치를 어디까지 허용하는지 고정하고, 개별 예측을 제곱하는 형태와 배치 분포를 사용하는 형태를 혼동하지 않아야 한다. 원문의 Figure 5 캡션과 §4.4 본문은 어느 분할을 고정하는지 표현이 엇갈려, 정확한 재현에는 실행 코드·분할을 확인해야 한다.')

    sub('[TabFM / TabFM+] (arXiv 기술보고서, 2026)')
    para('목적과 기반 구조.', 'TabFM은 합성 표로 사전학습한 transformer 기반 ICL 모델이며 TabPFN의 버전명이 아닌 별도 모델이다. 셀별 입력을 임베딩한 뒤 열과 행의 attention을 거쳐 행 표현을 만들고, 라벨이 있는 context와 query 사이의 관계로 예측했다. Figure 2의 왼쪽은 셀→행 표현→in-context predictor의 흐름이며, 오른쪽은 열·행 attention을 분리하는 구조다. 보고서의 기본 checkpoint는 400M 파라미터로 설명됐다.')
    paperfig('tabfm','Figure 2',2,'TabFM의 기본 모델 구조. TabFM+는 이 고정 모델을 여러 입력 view에 반복 적용하는 확장이다.','.86','.45')
    para('TabFM+가 결합하는 것.', 'TabFM+는 입력 특징의 변환·순열·교차 특징·SVD 등을 적용한 32개 view에서 예측을 얻었다. 내부 validation의 정답을 이용해 비음수 최소제곱으로 결합 가중치를 구하고, 그 가중치 75%와 균등 가중치 25%를 섞었다. 분류에는 사후 확률 보정도 적용했다. 따라서 본체 가중치는 고정됐지만, 결합기와 보정에 validation 라벨을 사용한 계산은 존재했다.')
    para('실험과 결과.', 'TabArena 51개 자료인 분류 38개·회귀 13개에서 평가했다. 원문 Table 2의 분류 Elo는 TabFM 1,768.6, TabFM+ 1,838.0이었고, 회귀 Elo는 각각 2,055.2·2,189.2였다. Table 2는 repeated 10-fold로 표기했으므로 §2에서 설명하는 TabArena 원 논문의 기본 outer 분할과 구분한다. 이 결과는 동일 모델의 입력 view 다양화가 도움이 된 사례이지만, Elo는 서로 다른 과제의 승패를 집계한 값이며 IDS의 F1 증가량이 아니다.')
    cite('tabfm','§2와 Figure 2 p.2; §3 Table 2; §4 및 추가 구현 설명')
    discussion('우리 앙상블도 같은 특징을 같은 방식으로 본 expert의 반복이라면 다양성 이득이 작을 수 있다. 입력 표현·지역 context·prior 보정 중 어떤 변화가 상호 보완 오류를 만드는지 분리해 볼 수 있다. 다만 공격당 2–5개에서 별도 validation을 얼마나 확보할 수 있는지까지 라벨 예산에 포함해야 하며, TabFM+의 32회 예측을 저비용 갱신의 근거로 바로 사용하지 않는다.')

    sub('[TFM 앙상블 다양성 분석] (arXiv, 2026)')
    para('목적과 방법.', '이 연구는 여러 강한 TFM을 합치면 항상 이득이 생기는지, 정확도와 확률 보정·비용을 함께 평가했다. TabPFN 계열, TabICL 계열, LimiX, Orion-MSP 등 6개 모델에 가중 평균·온도 조절·stacking·greedy selection·cascade 등의 6개 결합 전략을 적용했다. 새 backbone을 제안해서가 아니라, 우리의 앙상블 구조가 필요한지 판단하는 진단 연구로 포함했다.')
    paperfig('ensemble','Figure 1',5,'153개 과제의 평균 accuracy와 평균 fit 시간. 오른쪽으로 갈수록 준비 비용이 크다.','.72','.40')
    para('실험과 결과.', 'OpenML 분류 153개에서 모델 간 오류의 높은 유사성과 일부 결합기의 보정 저하를 보고했다. 논문이 강조한 cascade는 TabICL-v2보다 accuracy가 약 0.18%p 높았지만 계산량은 약 253배였다. Logistic-regression stacking도 accuracy와 AUC만 보았을 때와 log-loss까지 보았을 때 평가가 달랐다. 이는 해당 모델 pool과 프로토콜의 관찰이며 모든 앙상블의 개선 가능성이 없다는 증명은 아니다.')
    cite('ensemble','§3–5, Figure 1 p.5 및 Discussion')
    discussion('제안 앙상블을 개선하기 전 단일 expert의 오류 교집합과 oracle 상한을 계산하는 것이 우선이다. 어느 expert든 맞힌 사례를 선택하는 oracle도 단일 모델과 비슷하면 결합기보다 expert의 context·표현을 바꿀 필요가 있다. Oracle 여지가 크면 gating과 보정을 분리해 검증하고, 각 expert의 context를 줄인 영향이 앙상블 효과로 섞이지 않게 해야 한다.')

    sub('[TabPFN-3.5] (arXiv 기술보고서, 2026)')
    para('구조 변화와 목적.', 'TabPFN-3.5는 v3의 큰 틀을 유지하면서 입력 수치 표현과 모델 용량을 확장했다. Figure 8은 원래 수치의 선형·Fourier 표현에 context 내 순위를 나타내는 경험적 누적분포(ECDF) 등의 정보를 더하는 셀 인코딩을 보여준다. 행을 모으는 CLS token 수를 4개에서 8개로 늘렸고 in-context transformer 폭은 512에서 1024로 커졌다. 분류·회귀를 하나의 checkpoint로 학습했으며, 기본 8 estimators와 Fast 4 estimators를 구분했다.')
    paperfig('pfn35','Figure 8',10,'수치의 크기뿐 아니라 주기적 표현과 context 내 순위 등을 함께 입력하는 인코더.','.69','.46')
    para('실험과 결과.', '기술보고서는 TabArena뿐 아니라 시간·그룹 분할, 다양한 자료형과 관계형·시계열 과제도 평가했다. 기본형의 성능 개선과 Fast의 속도 개선, 별도 Plus·Thinking 모드를 구별해 보고했다. Fast는 v3 대비 최대 약 3배 빠르다고 보고했지만, 이 비율은 해당 실험 구성의 결과이며 현재 IDS의 v3 비용을 같은 비율로 줄일 근거는 아니다.')
    cite('pfn35','§3.1–3.4 pp.9–11, Figure 8; §4 및 초록')
    discussion('현재 EXP86·87은 v3를 그대로 사용했으므로 결과 변화에 새 checkpoint 효과는 섞이지 않았다. v3.5를 후속 비교한다면 같은 라벨 행과 context 정책으로 비교하고 estimator 수 또는 계산 예산을 맞춰야 한다. 입력 표현 개선만으로 tail 오류가 줄어드는지 먼저 확인하면 복잡한 앙상블이 필요한 부분을 좁힐 수 있다.')

    sub('[PFN 클래스 불균형 보정] (ITW 표기 원고, 2026)')
    para('목적과 방법.', '이 논문은 고정 가중치 ICL에서 클래스 불균형을 어떻게 보정할지 비교했다. 다수 클래스 downsampling, oversampling, 합성 증강, 클래스 prior를 이용한 판정 임계값 조절을 평가했다. 이진 분류에서 기본 0.5 대신 소수 클래스의 context 비율에 대응하는 임계값을 쓰면, 다수 클래스 prior의 영향으로 낮아진 소수 클래스 점수도 탐지에 활용할 수 있었다.')
    paperfig('imbalance','Figure 4',4,'불균형 정도별 임계값 변화와 다수·소수 클래스 정확도. 한 threshold의 이동이 서로 다른 클래스에 반대 효과를 낼 수 있다.','.67','.46')
    para('실험과 결과.', 'TabPFN-2.5와 OpenML 이진분류 11개를 사용했고, test는 클래스당 500개로 균형을 맞췄다. Table II는 context 100·500·1,000개와 여러 prior 조건을 평균했다. 평균 balanced accuracy는 무보정 0.710, 임계값 조절 0.793, downsampling 0.779였다. Worst-class accuracy는 각각 0.446·0.729·0.737로, 전체 평균과 가장 약한 클래스의 순위가 달랐다.')
    cite('imbalance','§II–III pp.2–3; Figure 4 p.4; Table II p.5. ITW는 저자 arXiv 메타데이터의 표기')
    discussion('이 결과는 context의 양을 줄이면서 prior 편향을 줄이는 대조군의 근거가 된다. 다만 이진·균형 test 조건이므로 다중 공격의 자연 빈도에서 동일한 이득을 보장하지 않는다. 우리의 tail F1 저하를 prior 문제와 특징 구분 문제로 나누고, 정상 오탐을 포함한 평가와 validation 기반 보정을 함께 검토해야 한다.')

    sub('[TabPFN 기반 IoT 포렌식 pipeline] (arXiv, 2026)')
    para('목적과 방법.', '이 보안 응용은 ToN-IoT의 7개 장치 telemetry에 TabPFN-2.5를 적용했다. 시간 특징·범주 인코딩·결측 보정 등의 전처리 후 PFN으로 의심 사례를 선별하고, 뒤의 random forest와 gradient boosting으로 공격 유형을 분석하는 pipeline을 제안했다. 우리 실험의 flow 특징 표와는 입력과 대상 단위가 다르다.')
    paperfig('iot','Figure 1',5,'PFN 선별 후 트리 앙상블로 세부 공격을 분류하는 원문 제안 흐름.','.56','.40')
    para('실험과 결과.', '원문은 stratified 80/20 분할과 5-fold 평가를 설명했다. Table 2의 장치 평균 이진 accuracy는 RF 99.48%, PFN 97.2%였고, 다중분류는 RF 99.44%, PFN 96.8%였다. 이 표는 개별 모델 결과이므로 그림의 전체 cascade가 단일 RF보다 우수하다는 근거로 바꿀 수 없다. Table 3의 PFN training time 0도 context 처리와 추론의 비용이 없다는 의미는 아니다.')
    cite('iot','§3 pp.4–6; Tables 2–3 p.7')
    discussion('관련성은 보안 자료에서 TFM과 트리 모델을 역할별로 결합하려 했다는 점이다. 우리 모델에는 앞 단계가 놓친 공격을 뒤 단계가 복구할 수 있는지, 두 모델의 오류가 상호 보완적인지 확인하는 근거가 필요하다. 동일한 ToN 명칭이나 제안 pipeline 그림만으로 데이터·분할·전체 성능이 동일하다고 판단하지 않는다.')

def tabarena():
    raw(r'\clearpage')
    section('TabArena 데이터와 평가방식')
    p('TabArena는 하나의 거대한 표가 아니라 여러 분야의 분류·회귀 자료와 공통 평가 pipeline을 묶은 벤치마크다. NeurIPS 2025 Datasets and Benchmarks 트랙 논문의 v0.1은 51개 자료, 분류 38개·회귀 13개로 구성됐다. 금융·의료·재료 등 서로 다른 행과 특징을 가진 표들을 각각 별도 과제로 평가한다. 아래 설명과 파일 예시는 이 버전 및 내려받은 공식 배포 자료를 기준으로 하며, 수시로 바뀌는 현재 리더보드 전체와 동일하다고 가정하지 않는다.')
    cite('tabarena','§2.2–2.3, Figure 3, Appendix B 및 C')
    raw(link('https://huggingface.co/spaces/TabArena/leaderboard','공식 리더보드')+' · '+link('https://github.com/TabArena/tabarena','공식 코드')+' · '+link('https://huggingface.co/datasets/TabArena/BeyondArena','공식 데이터 배포 저장소')+'\n')
    sub('자료의 규모와 대표 예시')
    p('로컬에 보관한 v0.1 metadata의 전체 행 수는 자료별 748–150,000개였고, target을 제외한 입력 특징은 4–1,776개였다. 열 수는 원본 metadata가 target까지 세므로 보고서에서는 1을 뺐다. 직접 열어 보기 위한 예시는 이진분류·다중분류·회귀·결측과 불균형이 있는 자료에서 4개를 골랐다. 이 네 예시는 모두 비교적 작은 자료이므로 전체 벤치마크의 최대 규모를 대표하지 않는다.')
    tab([['대표 유형 / 자료','행 × 입력 특징','내용 / target','평가 / 실제 예시 열'],
         ['금융 이진 / credit-g','1,000 × 20','대출 고객; good/bad = 700/300','ROC-AUC / checking_status, duration_months'],
         ['의료 3분류 / maternal health','1,014 × 6','산모 위험; low/mid/high = 406/336/272','log-loss / Age, SystolicBP, DiastolicBP'],
         ['재료 회귀 / concrete','1,030 × 8','배합·양생 조건 → 압축강도','RMSE / Cement, BlastFurnaceSlag, FlyAsh'],
         ['의료 8분류 / MIC','1,699 × 111','심근경색 합병증·생존; 최다 1,428 / 최소 12','log-loss / AGE, SEX, INF_ANAM; 결측 15,965셀']])
    p('다음은 다운로드한 공식 배포본에서 변환한 data.csv의 처음 세 행과 일부 열을 그대로 가져온 것이다. 설명을 위해 만든 가상 행이 아니며, 표시하지 않은 나머지 입력 열도 원본 파일에 있다. 숫자와 target 문자열을 보존하고, 표의 행 번호는 이 로컬 CSV의 1부터 센 데이터 행 번호로 표시했다.')
    manifest=json.loads((ROOT/'data/tabarena_examples/_metadata/download_manifest.json').read_text())
    entries=manifest.get('datasets',[])
    if not entries: entries=next(v for v in manifest.values() if isinstance(v,list) and v and isinstance(v[0],dict) and 'name' in v[0])
    columns={'credit_g':['checking_status','duration_months','good_or_bad_customer'],
             'maternal_health_risk':['Age','SystolicBP','DiastolicBP','RiskLevel'],
             'concrete_compressive_strength':['Cement','BlastFurnaceSlag','FlyAsh','ConcreteCompressiveStrength'],
             'mic':['AGE','SEX','INF_ANAM','LET_IS']}
    for e in entries:
        name=e['name']; d=pd.read_csv(ROOT/'data/tabarena_examples'/name/'data.csv')
        para(name.replace('_',' '),'선택한 열의 실제 CSV 미리보기.')
        tab([['행']+columns[name]]+[[str(i+1)]+[str(d.iloc[i][c]) for c in columns[name]] for i in range(3)])
        raw(link(e['upstream_url'],'원자료 제공처')+' · '+link('https://huggingface.co/datasets/TabArena/BeyondArena/tree/'+e['hf_revision']+'/'+name,'확인한 공식 배포 revision')+r'\par')
        raw(r'\sourcepath{data/tabarena_examples/'+name+'/data.csv}')
    p('원자료 압축본과 공식 배포 parquet·metadata도 각 예시 폴더에 보관했다. OpenML 접속이 되지 않아 공식 Hugging Face 배포본을 사용했으며, v0.1과 행·특징 수가 맞는 것은 확인했지만 OpenML 파일과 행 순서·값이 완전히 동일하다고 검증한 것은 아니다. Maternal health의 중복 562행, concrete의 중복 25행처럼 자료 품질 정보도 manifest에 남겼다. 자료를 실제 새 실험에 쓸 때에는 해당 중복과 분할이 일반화 평가에 미치는 영향을 따로 확인해야 한다.')
    sub('점수와 순위가 만들어지는 과정')
    steps([
      '외부 평가 분할: 클래스별 층화한 3-fold outer cross-validation을 사용했다. 전체 2,500행 미만은 10회 반복, 그 외에는 3회 반복했다.',
      '모델 선택: 기본 설정, hyperparameter 튜닝, 튜닝된 설정들의 앙상블을 구분했다. 내부에는 기본적으로 8-fold validation과 그 예측을 결합하는 절차를 사용했다.',
      'TFM 처리: 모든 모델이 8개 내부 모델을 그대로 결합하는 것은 아니다. Foundation model은 저자 권장에 따라 선택 후 train과 validation을 합쳐 refit하는 방식을 사용했다고 명시했다.',
      '과제별 점수: 이진분류는 ROC-AUC, 다중분류는 log-loss, 회귀는 root mean squared error (RMSE)를 사용했다.',
      '과제 간 집계: 모델 간 승패로 Elo를 계산했다. Elo는 과제별 성능 차이의 크기를 직접 평균하는 점수가 아니므로, 순위와 실제 정확도·비용을 함께 읽어야 한다.'
    ])
    cite('tabarena','§2.1 pp.3–4; §2.3 p.6; Appendix C.3–C.4, D.1–D.2')
    tab([['구분','TabArena v0.1','현재 IDS 연구'],
         ['질문','여러 분야의 표에서 일반 예측 성능은 어떠한가','소수 공격 라벨로 어느 공격을 구분하고 유지할 수 있는가'],
         ['분할','IID 중심의 반복 outer CV','기존 고정 train/test와 라벨 수 조절; 단계 평가 별도'],
         ['표본 예산','주로 train 500–250,000의 큐레이션 범위','공격당 2–수십 개; 확보량과 보존량 구분'],
         ['핵심 지표','과제별 AUC/log-loss/RMSE → Elo','Macro-F1, 클래스별 품질, 정상 오탐, 비용'],
         ['모델 비교','default / tuned / ensemble의 잠재력','동일 사례·동일 버퍼의 XGB와 고정 가중치 PFN']])
    discussion('IDS도 숫자·범주 표라는 점에서 TabArena의 강한 트리 기준선과 공정한 validation 절차를 참고할 수 있다. 반면 이 논문은 few-shot·시간 변화 같은 범위를 기본 벤치마크 밖으로 명시했으므로, Elo 우위가 곧 소수 공격 적응이나 저비용 운영의 근거가 되지는 않는다. 우리 평가는 데이터 이름보다 확보 라벨·보존량·평가 시점·비용의 차이를 명확히 드러내야 한다.')

def lotbonc():
    raw(r'\clearpage')
    section('평가 시나리오 고안에 참고할 논문')
    sub('[LotBoNC] (ACM ASIA CCS, 2026)')
    p('LotBoNC는 이미 라벨이 알려진 botnet과 이름을 아직 모르는 botnet이 섞인 트래픽에서 새로운 군집을 찾는 문제를 다뤘다. 핵심은 실제 트래픽에서 라벨이 보이는 집합과 보이지 않는 집합을 나누고, 비라벨 집합에 불균형을 부여한 다음 학습하는 구성이었다. 우리의 신규 공격별 소수 정답 사례를 제공하는 분류와는 목표가 다르지만, 원본 자료에서 평가 조건을 만드는 이유와 절차를 설명하는 데 직접 참고할 수 있다. 아래 페이지는 첫 장을 1로 센 PDF 기준이다.')
    cite('lotbonc','§4 pp.3–6; §5 pp.6–8; 인쇄 pp.1013–1027')
    para('§4.1: 무엇을 알고 시작하는가.', '라벨 집합 L에는 known 공격의 이름과 트래픽이 있다. 비라벨 집합 U에는 known 공격과 novel 공격이 섞여 있지만, 학습 알고리즘에는 그 정답 이름을 주지 않는다. 모델은 known 공격의 구분을 유지하면서 novel 공격끼리도 서로 다른 군집으로 묶어야 한다. 평가자는 숨겨 둔 정답으로 군집이 실제 공격 유형과 맞는지 채점한다.')
    raw(r"""\[
       L=\{(x_i,y_i): y_i\in\mathcal Y_{\mathrm{known}}\},\qquad
       U=\{x_j: y_j\in\mathcal Y_{\mathrm{known}}\cup\mathcal Y_{\mathrm{novel}}\}.
    \]""")
    p('여기서 U의 y 표기는 평가자가 보유한 정답의 범위를 설명하기 위한 것이다. 알고리즘이 학습 시 U의 정답을 받는다는 뜻이 아니다. 이 차이 때문에 LotBoNC의 novel discovery 성능을 우리 소수 라벨 공격 분류 F1과 바로 비교할 수 없다.')
    para('§4.2: PCAP을 모델 입력으로 바꾸는 과정.', '원시 패킷 캡처에서 양방향 flow를 복원하고, 한 방향으로 이어지는 패킷을 burst로 나눴다. 프로토콜 header를 제거한 payload를 연결한 뒤, 인접한 2-byte bigram을 token으로 바꿨다. 여기에 CLS·SEP·MASK·PAD 같은 특수 token과 위치·구간 정보를 더해 transformer encoder에 입력했다. 따라서 LotBoNC의 입력은 숫자 flow 통계 표가 아니라 패킷 바이트에서 만든 token열이었다.')
    steps([
      'PCAP → flow: 통신을 같은 연결 단위로 묶고 방향을 식별한다.',
      'Flow → burst: 한 방향으로 연속 전송되는 구간을 하나의 입력 단위로 나눈다.',
      'Burst → token열: payload 바이트를 bigram token으로 바꾸고 특수·위치·구간 정보를 붙인다.',
      '두 view 구성: 같은 token열에 독립적인 dropout을 적용해 두 표현을 만든다. 원문 §4.2의 증강은 입력 바이트를 임의로 변조하는 방식이 아니었다.'
    ])
    cite('lotbonc','§4.2–4.3, pp.3–4; Figure 2–3')
    para('§4.3: 먼저 어떤 표현을 사전학습했는가.', 'Masked-token 예측과 대조 학습을 함께 사용했다. Masked 예측은 가려진 바이트 token을 주변 내용으로 맞히게 했고, 대조 학습은 같은 트래픽의 두 dropout 표현을 가깝게 하고 다른 트래픽 표현을 멀게 했다. 이는 뒤 단계에서 소수·미지 공격의 유사성을 이용할 표현을 준비하는 절차였다.')
    raw(r"""\begin{align}
       \mathcal L_{\mathrm{mask}}&=-\sum_{i\in M}\log p_\theta(x_i\mid x_{\setminus M}),\\
       \mathcal L_{\mathrm{contrast},i}
        &=-\log\frac{\exp(\operatorname{sim}(h_i,h_i^+)/\tau)}
                         {\sum_j\exp(\operatorname{sim}(h_i,h_j)/\tau)}.
    \end{align}""")
    p('M은 가린 token 위치이고, 첫 식은 정답 token의 확률이 작으면 큰 손실을 준다. 두 번째 식에서 h_i와 h_i⁺는 같은 입력의 다른 view이며, 분모는 비교 대상 표현들이다. τ는 유사도 차이를 얼마나 강하게 반영할지 조절하는 온도다. 위 식은 원문 Eq.(1)–(2)의 핵심을 설명용 표기로 옮긴 것이며, 구현 시에는 positive·negative의 정확한 집계 범위를 원문과 코드에 맞춰야 한다.')
    para('§4.4–4.7: 비라벨 데이터에 임시 정답을 붙이고 표현을 다시 학습.', '모델의 현재 예측만 최대값으로 골라 pseudo-label을 붙이면 다수 공격으로 표본이 몰릴 수 있다. LotBoNC는 각 클래스가 차지할 비율을 추정하고, 그 비율을 제약으로 사용해 batch 전체의 pseudo-label을 함께 배정했다. 그다음 배정 결과로 클래스 대표 벡터와 encoder를 갱신하고 이 과정을 반복했다.')
    paperfig('lotbonc','Figure 4',5,'E-step에서 비라벨 표본을 배정하고, M-step에서 prototype·prior·표현을 갱신하는 반복 과정.','.75','.40')
    para('E-step: 최적 수송으로 batch 전체를 배정.', 'm개 비라벨 사례와 K개 후보 클래스가 있을 때 P는 현재 모델이 예측한 m×K 확률 행렬이다. A는 각 사례가 각 클래스에 속할 양을 담는 soft assignment 행렬이다. 모델이 높은 확률을 준 배정은 비용이 작고, batch의 클래스별 총량은 추정 prior r에 맞춘다. 이때 prior가 불균형이면 배정도 그 불균형을 허용하며, 항상 클래스 수를 같게 만드는 절차가 아니다.')
    raw(r"""\begin{align}
       C_{ik}&=-\log P_{ik},\qquad H(A)=-\sum_{i,k}A_{ik}\log A_{ik},\\
       A^\star&=\arg\min_{A\ge0}\ \sum_{i,k}A_{ik}C_{ik}-\eta H(A),\\
       A\mathbf1_K&=w,\quad A^\top\mathbf1_m=r,\quad
       w_i=1/m,\quad \sum_k r_k=1.
    \end{align}""")
    p('첫 항은 모델 예측에 맞는 배정을 선호하고, 엔트로피 항은 처음부터 지나치게 한 클래스에 확정하지 않도록 배정을 부드럽게 한다. w는 각 사례의 질량이고 r는 클래스별 질량이다. 예를 들어 10개 사례에 r=(0.6, 0.3, 0.1)이면 전체적으로 6·3·1개에 해당하는 양을 배정하되, 각 행의 배정은 확률처럼 나뉠 수 있다. 최적 수송(optimal transport, OT)은 이 행·열 합 제약을 동시에 만족시키는 배정 문제를 푸는 장치다.')
    para('Sinkhorn 반복을 쉽게 읽기.', '모델 확률로 초기 배정 행렬을 만든 다음, 행의 합과 열의 합을 번갈아 원하는 값으로 맞춘다. 여기서 반복하는 것은 모델 전체를 새로 학습하는 것이 아니라 현재 batch의 배정 행렬을 정규화하는 계산이다. 다음은 원문 Eq.(3)의 목적함수에 맞춰 행·열 차원을 통일한 설명식이다.')
    raw(r"""\[
      K_{\mathrm{OT}}=\exp(-C/\eta)=P^{1/\eta},\qquad
      u\leftarrow w/(K_{\mathrm{OT}}v),\qquad
      v\leftarrow r/(K_{\mathrm{OT}}^\top u),\qquad
      A=\operatorname{diag}(u)K_{\mathrm{OT}}\operatorname{diag}(v).
    \]""")
    p('나눗셈은 원소별 연산이며 u는 m차원, v는 K차원이다. 각 행을 합이 1인 학습용 pseudo-label로 쓰려면 위 질량 표현에서 mA를 사용한다. 원문 인쇄식의 kernel 부호와 α·β 차원, Algorithm 1의 행·열 갱신에는 서로 맞지 않는 표기가 있어 이 보고서에서는 Eq.(3)에 일관된 설명 표기를 사용했다. 이를 원문 구현을 그대로 재현한 수식으로 제시하지 않으며, 확인 지점은 부록에 모았다.')
    cite('lotbonc','§4.4 Eq.(3)–(5), PDF p.5; Algorithm 1 p.6')
    para('M-step: 대표 벡터와 클래스 비율을 갱신.', 'Prototype c_k는 k번째 클래스의 대표 표현이다. Encoder가 만든 사례 표현 v_i를 pseudo-label에 맞는 prototype 쪽으로 끌어당기고, 다른 prototype과 구별되도록 학습한다. 동시에 현재 배정된 클래스 비율 z로 prior를 갱신하고, 해당 클래스의 평균 표현으로 prototype을 갱신한다. 급격한 변동을 줄이기 위해 이전 값과 새 값을 지수 이동평균으로 섞는다.')
    raw(r"""\begin{align}
       z_k&=\frac1m\sum_i \mathbf1\{\arg\max_j P_{ij}=k\},\\
       r_k&\leftarrow \mu r_k+(1-\mu)z_k,\\
       c_k&\leftarrow \mu c_k+(1-\mu)\bar v_k .
    \end{align}""")
    p('μ가 크면 이전 추정을 오래 유지하고, 작으면 현재 batch의 비율과 표현을 빠르게 반영한다. 처음의 균일 prior가 이 과정에서 불균형한 prior로 바뀔 수 있다는 것이 핵심이다. Prototype은 실제 보관 중인 패킷 한 개를 뜻하지 않고 encoder 공간의 대표 벡터다. 해당 클래스에 배정된 표본이 없는 batch의 처리 등 구현 세부는 별도 확인이 필요하다.')
    para('Prototype loss와 전체 학습.', '설명상 prototype 분류의 핵심은 pseudo-label 가중치로 정답 prototype의 softmax 확률을 높이는 것이다. 원문 Eq.(6)은 여기에 prior 항을 더한 손실을 제시했다. 다만 Eq.(6)과 Appendix A.1의 Eq.(23)에서 prior 항의 부호가 다르게 읽혀, 아래에는 일치하는 prototype 분류 핵심만 분리해 적는다.')
    raw(r"""\begin{align}
       q_{ik}&=\frac{\exp(v_i^\top c_k)}{\sum_j\exp(v_i^\top c_j)},&
       \mathcal L_{\mathrm{proto,core}}&=-\sum_{i,k}\widetilde A_{ik}\log q_{ik},\\
       \mathcal L_{\mathrm{overall}}
       &=\mathcal L_{\mathrm{ins}}+\lambda_{\mathrm{proto}}\mathcal L_{\mathrm{proto}}
                              +\lambda_{\mathrm{sup}}\mathcal L_{\mathrm{sup}}.
    \end{align}""")
    p('L_ins는 같은 비라벨 입력의 두 view를 가깝게 만드는 instance 대조 손실이고, L_sup는 정답이 있는 같은 클래스 사례를 가깝게 만드는 supervised 대조 손실이다. L_proto는 pseudo-label과 prototype을 이용해 비라벨 사례의 구조를 학습한다. λ는 각 목표를 얼마나 강하게 반영할지 정한다. 이 전체 손실로 encoder 가중치 θ를 실제로 학습하므로, LotBoNC는 고정된 TabPFN의 context만 교체하는 방식과 다르다.')
    cite('lotbonc','§4.5–4.7 Eq.(6)–(13), pp.5–6; Appendix A.1 Eq.(22)–(23), p.15')
    para('데이터 구성: 무엇을 원본에서 선택했는가.', '원문의 사전학습 출처는 CTU-MAL-CAP·VNAT·ISCX-VPN-2016·ToN-IoT였고, 평가 구성에는 IoT-23·ISCX-Bot-2014·CTU-13을 사용했다. §5.1은 비-IP 제어 트래픽을 제외하고 TCP/UDP의 최소 5패킷 flow를 유지했다고 설명했다. 각 평가 데이터 안에서 known의 균형 라벨 집합과 known+novel의 불균형 비라벨 집합을 따로 만들었다. 즉 원본 트래픽은 관측 자료지만, 어떤 클래스의 라벨을 공개하고 표본을 몇 개 선택할지는 실험자가 설계했다.')
    tab([['구성 단계','LotBoNC의 선택','우리 시나리오에 참고할 설명'],
         ['원본→학습 단위','PCAP→flow→burst/token','flow 정의·제외 기준·추출 특징을 밝힘'],
         ['라벨 가용성','known만 라벨; novel은 정답 비공개','초기/신규 공격의 라벨 공개 단계를 명시'],
         ['비율 조절','비라벨 known/novel 비 γ=1, 5, 10','새 공격당 n과 retained buffer B를 따로 정의'],
         ['채점','known 분류 + novel 군집 정합도','우리의 이름 있는 공격별 F1·기존 성능 유지와 구분']])
    p('γ는 비라벨 known 표본 수를 novel 표본 수로 나눈 값으로 정의됐다. Novel의 군집 번호는 임의이므로 Hungarian matching으로 정답 클래스와 가장 잘 대응시킨 뒤 정확도를 계산했다. 5개 seed를 사용했으며 Table 2의 γ=1, IoT-23 전체 accuracy는 LotBoNC 94.48±0.41%, RoNeTC 91.34±0.30%였다. γ=5의 IoT-23 unlabelled-aware 결과는 75.73±0.20% 대 51.50±0.65%로 보고됐다. 이 값들은 우리 attack-only Macro-F1과 다른 지표다.')
    cite('lotbonc','§5.1–5.4 pp.7–8; Table 2 p.8')
    raw('직접 PCAP을 검토할 때의 출처: '+link('https://www.stratosphereips.org/datasets-iot23','IoT-23 PCAP·라벨 flow')+' · '+link('https://www.stratosphereips.org/datasets-ctu13','CTU-13 공개 범위')+' · '+link('https://www.unb.ca/cic/datasets/botnet.html','ISCX botnet 구성 설명')+'.\n')
    discussion('우리에게 가장 유용한 부분은 같은 원본 안에서 학습 단위·라벨 공개 범위·클래스별 수를 정의한 평가 구성이다. 이를 따라 원본 데이터 A에서 현재 사용 사례 B를 만드는 과정을 행 ID로 보여주면, 서로 다른 벤치마크를 합치는 것으로 오해하는 일을 줄일 수 있다. 다만 실제 timestamp 재생, 정답이 없는 공격 발견, 라벨 확보 후 분류는 별개 질문이므로 선택한 목표에 맞춰 설명해야 한다. PCAP을 직접 쓸 경우에는 capture/session 중복과 분할 독립성부터 점검하고, 실제 flow 선택과 합성 prototype을 구별해 기록한다.')

def streams():
    sub('[Rethinking Tabular Foundation Models On Data Streams] (arXiv, 2026)')
    p('이 논문은 데이터가 계속 도착할 때 TFM의 context를 어떻게 유지할지, 정확도와 처리 비용이 어떤 관계를 갖는지 연구했다. 고정된 최종 test만 채점하는 대신 매 시점에서 먼저 예측하고 나중에 정답을 받아 갱신하는 test-then-learn 평가를 사용했다. 메모리 정책, 기반 모델, label delay, 갱신 주기와 query batch 크기를 나누어 실험한 점이 핵심이다.')
    paperfig('streams','Figure 1',2,'스트리밍 학습기·AutoML·ICL 모델을 같은 예측→라벨→갱신 프로토콜에서 비교한다.')
    para('한 시점의 동작을 풀어 보면.', '시점 t의 x_t가 도착하면, 그때까지 라벨을 받은 사례만 담은 메모리 M_t로 예측한다. 정답 y_t는 d개 사례 뒤에 도착하고, 도착한 정답만 모델 또는 context 갱신에 쓴다. 미래 정답을 지금의 context에 넣지 않는 것이 prequential 평가의 핵심이다. 원문의 기본값은 warm-up 100개, 보존 메모리 B=1,000, 즉시 라벨 d=0, query 1개였다.')
    raw(r"""\[
       \widehat y_t=f_\theta(x_t;M_t),\qquad
       M_{t+1}=\operatorname{Keep}_{B}
       \bigl(M_t\cup\{\text{시점 }t\text{에 정답을 받은 사례}\}\bigr).
    \]""")
    p('위 식은 고정 가중치 ICL의 메모리 갱신을 설명한 것으로, 모든 비교 모델이 같은 방식으로 학습한다는 뜻은 아니다. 트리 기반 온라인 학습기는 정답 도착 시 내부 학습 상태를 바꾸고, ICL 모델은 라벨 사례 메모리를 바꾼다. 현재 우리의 단계별 신규 클래스 제공·고정 test 평가를 이 프로토콜과 동일하다고 부르면 안 된다.')
    para('어떤 데이터를 흘려 보냈는가.', '실제 도착 순서가 있는 실데이터 5개와 생성기로 drift를 만든 합성 stream 4개를 사용했다. 실데이터는 전력·날씨·스마트미터·영상 표현·자세 자료였고, 합성 자료는 abrupt·gradual·incremental·recurring 변화의 사례였다. 서로 다른 drift 유형에 서로 다른 생성기를 사용했으므로, 단순한 점수 차이를 drift 종류 하나의 효과로 해석할 수는 없다.')
    tab([['실데이터','행 × 특징 / 클래스','합성 stream','행 × 특징 / 클래스'],
         ['electricity','45,312 × 8 / 2','agr_a, abrupt','30,000 × 9 / 2'],
         ['noaa','18,159 × 8 / 2','sea_g, gradual','100,000 × 3 / 2'],
         ['meter','22,948 × 96 / 10','rbf_m, incremental','100,000 × 10 / 5'],
         ['rialto','82,250 × 27 / 10','tree_r, recurring','60,000 × 10 / 2'],
         ['posture_no8','163,477 × 3 / 10','—','—']])
    para('무엇을 비교했는가.', 'TabPFN v1·drift 사전학습 DR-TabPFN·TabICL v2·TabFM에 단순 FIFO를 적용했다. FIFO는 오래된 라벨 사례부터 내보내 최근 B개를 남겼다. DualFIFO는 최근 사례와 클래스 균형을 고려한 별도 메모리를, CURE는 중복을 줄이는 선택을 이용했다. 비교군에는 adaptive random forest (ARF), streaming random patches (SRP), Hoeffding adaptive tree (HAT), 온라인 AutoML 등도 포함됐다.')
    p('Backbone을 고정한 FIFO 대 DualFIFO 또는 CURE 비교와, backbone 자체를 바꾼 비교를 나누어 읽어야 한다. 원문은 B=100·250·500·1,000, label delay 0·100·1,000, query batch 1·10·100 등의 별도 실험으로 요인을 분리했다. 기본 점수는 1,000개 단위의 시간창에서 추적했으며 accuracy 외에 balanced accuracy와 시간적 기준선 대비 κ_t도 사용했다.')
    raw(r"""\[
       \kappa_t=\frac{\mathrm{Accuracy}-\mathrm{Accuracy}_{\mathrm{NoChange}}}
                         {1-\mathrm{Accuracy}_{\mathrm{NoChange}}}.
    \]""")
    p('NoChange는 직전에 관측한 라벨이 계속 나온다고 예측하는 기준선이다. 같은 라벨이 긴 구간 이어지는 stream에서는 이런 단순 기준선도 높아질 수 있으므로, 높은 accuracy가 실제 적응을 뜻하는지 확인하는 지표로 κ_t를 사용했다.')
    cite('streams','§2–3 및 Table 1 pp.2–3; Appendix A Tables 7–8')
    para('실험 결과: 좋은 backbone과 단순 메모리가 강했다.', '9개 stream 평균에서 TabFM+FIFO의 accuracy는 0.869, balanced accuracy는 0.829였다. TabICL+FIFO는 0.860·0.820, 같은 backbone의 CURE는 0.857·0.819였다. TabPFN+FIFO는 0.805·0.762, DualFIFO는 0.804·0.770으로, 복잡한 정책이 전체 평균을 일관되게 높이지는 않았다. 다만 라벨이 오래 지연되는 smart-meter에서는 CURE의 이득이 나타나 조건별 효과는 남았다.')
    tab([['원문 방법','평균 accuracy','평균 balanced acc.','시간 기하평균 (ms/사례)'],
         ['TabFM + FIFO','0.869','0.829','167.00'],
         ['TabICL v2 + FIFO','0.860','0.820','58.40'],
         ['CURE (TabICL)','0.857','0.819','87.30'],
         ['TabPFN v1 + FIFO','0.805','0.762','103.00'],
         ['DualFIFO (TabPFN)','0.804','0.770','127.00'],
         ['ARF','0.791','0.742','6.30'],['SRP','0.803','0.749','8.47'],['HAT','0.694','0.654','0.32']])
    p('TFM은 더 높은 예측 성능을 보인 대신 단일 사례 처리 비용이 컸다. 위 시간은 TFM의 GPU 실행과 비교군의 CPU 실행을 포함한 원문 환경의 결과이며 동일 하드웨어에서 알고리즘만 바꾼 비교가 아니다. 값은 단순 산술평균 시간이 아니라 stream별 시간의 기하평균이다.')
    cite('streams','Table 2 p.4; Table 3 및 Figure 6 pp.6–7; Appendix A Table 8')
    para('왜 context 갱신 빈도를 줄여도 충분히 빨라지지 않았는가.', '원문에서 메모리의 행을 교체하는 refresh와 예측 시 그 메모리를 다시 encoding하는 작업은 달랐다. Refresh 간격을 늘려도 각 예측 호출이 context 전체를 다시 처리하면 큰 비용은 그대로 남았다. 예를 들어 TabICL의 meter에서 refresh 간격 1→1,000은 70.3→60.7ms/사례에 그쳤지만 accuracy는 0.906→0.710으로 낮아졌다. 반면 같은 context에 query를 100개씩 묶으면 meter의 TabICL 추론이 88.17→1.89ms/사례로 줄었고 accuracy는 0.906→0.875였다.')
    raw(r"""\[
       t(q)\approx A+q\gamma,\qquad
       t(q)/q\approx A/q+\gamma.
    \]""")
    p('A는 context encoding·전처리·전송·호출 등의 고정 비용이고 γ는 query 한 개를 더 처리하는 비용이다. Batch q가 커지면 고정 비용을 나눌 수 있지만, 앞의 query를 기다리게 하거나 갱신을 늦추는 문제가 생길 수 있다. 식은 원문의 실측 회귀 모형이며 A 전체가 encoding 비용이라는 정확한 분해는 아니다.')
    paperfig('streams_cache','Figure 7',8,'거의 같은 context를 매번 다시 encoding하는 방식과 이전 표현을 cache에서 재사용하는 방식의 차이.')
    para('캐시 실험이 시사하는 구조 변화.', '저자들은 평가 stream이 아닌 합성 prior로 작은 causal-attention 모델을 사전학습하고, 같은 checkpoint의 재encoding 경로와 ring cache 경로를 비교했다. Causal 구조에서는 도착한 행을 한 번 표현하고 이후 query가 저장된 표현을 읽을 수 있었다. 적절히 전이되는 조건에서 cache의 사례당 비용은 메모리가 커져도 거의 일정했지만, 일부 stream에서는 모델 전이가 잘되지 않았다. 이 결과는 현재 사용하는 TabPFN-v3의 cache 교체가 그대로 무상 incremental update라는 뜻은 아니다.')
    cite('streams','§4 Tables 4–6, Figures 7–8 pp.7–8; Appendix D.4–D.6')
    raw(link('https://github.com/TechyNilesh/StreamTFM-Study','저자 코드·실험 자료')+'\n')
    discussion('우리 연구는 먼저 극소 라벨의 분류 가능성과 정상·기존 공격 유지 여부를 다루고, 실제 도착순 평가를 추가할지는 별도로 결정해야 한다. 스트림으로 확장한다면 같은 B에서 단순 FIFO를 강한 기준선으로 두고, 정보가 늘어나는 이득과 cache 비용을 분리하는 것이 좋다. 현재 v3의 구현에서 행 선택·전처리·context encoding·query 추론을 실제로 계측한 뒤 구조 개선이 필요한 병목을 판단해야 한다.')

def icd():
    raw(r'\clearpage')
    section('Context 선택·최적화에 참고할 논문')
    sub('[In-Context Data Distillation with TabPFN] (ICLR ME-FoMo Workshop, 2024)')
    p('In-context data distillation (ICD)은 큰 라벨 train의 정보를 작은 context에 압축하도록 context의 특징값 자체를 학습했다. 기존 TabPFN이 작은 표에서는 유망했지만 큰 train을 그대로 넣기 어려웠고, 무작위 1,000개만 선택하면 정보를 버리게 된다는 문제에서 출발했다. 원문은 기본 PFN을 미세조정하는 대신, 고정 PFN이 전체 train을 잘 예측하게 만드는 소수의 합성 사례를 최적화했다. 제공된 논문은 ICLR 2024 ME-FoMo 워크숍 원고이며 ICLR 본회의 논문으로 표기하지 않는다.')
    cite('icd','Introduction, §2–3 pp.1–3')
    raw(link('https://openreview.net/pdf/b0713981c817e0198b59d2cd1dba5911acb076e3.pdf','사용자 제공본과 대응하는 OpenReview PDF')+'\n')
    paperfig('icd','Figure 2',3,'일반 dataset distillation의 내부 모델 학습과 외부 데이터 학습을, 고정 TabPFN에 대한 context 최적화 한 단계로 바꾼다.')
    para('무엇을 고정하고 무엇을 바꾸는가.', 'θ는 이미 사전학습된 TabPFN 가중치이고 학습 중 바꾸지 않는다. D_train은 실제 라벨 자료 전체이고, D_dist=(X_dist, y_dist)는 예측에 넣을 작은 context다. 원문 실험은 클래스별 균형이 되도록 y_dist를 고정하고 X_dist만 연속값으로 최적화했다. 최적화 뒤의 X_dist는 원본 flow를 골라 보관한 표가 아니라 모델에 유용하도록 이동한 합성 특징점이다.')
    raw(r"""\begin{align}
       D_{\mathrm{dist}}&=(X_{\mathrm{dist}},y_{\mathrm{dist}}),\qquad \theta\ \text{고정},\\
       X_{\mathrm{dist}}^\star
       &=\arg\min_{X_{\mathrm{dist}}}
          \mathbb E_{(x,y)\sim D_{\mathrm{train}}}
             [-\log p_\theta(y\mid x,D_{\mathrm{dist}})],\\
       X_{\mathrm{dist}}&\leftarrow X_{\mathrm{dist}}
                  -\alpha\nabla_{X_{\mathrm{dist}}}\mathcal L.
    \end{align}""")
    p('실제 train의 한 사례 x를 query로 넣고, 현재 작은 context를 보고 정답 y를 맞히도록 한다. 틀렸다면 그 오차를 query→TabPFN→context 입력으로 역전파해 X_dist를 움직인다. 모델 가중치로 미분값이 흐를 수 있는 연산 경로와 그 가중치를 optimizer로 갱신하는 것은 다르다. 여기서는 optimizer가 context 특징값만 바꾼다.')
    steps([
      '큰 라벨 train과 별도 validation/test를 확보한다. 이때 사용한 모든 라벨은 획득 예산에 포함된다.',
      '정해진 크기의 context와 클래스별 고정 라벨을 준비한다. 초기 점의 구체적인 설정은 원문의 그림·실험 구성을 구별한다.',
      '실제 train 일부를 query로 삼아 frozen TabPFN에 현재 context와 함께 넣고 정답의 음의 로그확률 손실을 계산한다.',
      '손실의 gradient로 context 특징 X_dist만 갱신한다. 여러 train minibatch를 반복해 전체 train 정보를 압축한다.',
      'Validation AUC로 중단·선택하고, 최종 합성 context로 독립 test를 예측한다. 이 마지막 단계는 추가 가중치 학습이 필요하지 않다.'
    ])
    para('왜 일반 distillation보다 구조가 단순해졌는가.', '일반 dataset distillation에서는 합성 사례로 작은 모델을 여러 단계 학습한 뒤 그 모델의 성능으로 합성 사례를 수정한다. 합성 자료→모델 학습→최종 예측을 모두 미분해야 해서 두 겹 최적화가 된다. TabPFN은 context를 읽는 forward 자체가 새 표에 대한 예측 절차이므로, 중간에 task-specific 모델 학습을 펼칠 필요가 없다. 다만 context를 찾는 반복 최적화의 시간과 메모리는 여전히 필요하다.')
    cite('icd','§3 Eq.(1)–(3), Figure 2 p.3')
    para('실험은 극소 라벨 확보가 아니라 큰 train의 압축.', 'TabZilla/OpenML에서 2,000–200,000개 사례를 가진 결측 없는 48개 자료를 골랐다. Train/validation/test는 80/10/10으로 나눴고 validation AUC로 early stopping했다. 합성 context는 항상 총 1,000개였으며, Adam 학습률 0.01로 X_dist를 최적화했다. 비교용 TabPFN은 무작위 1,000개 실제 사례를 사용했다. 그림의 two-moons 예시에서 클래스당 8개 합성 점을 썼다는 사실은 실제 라벨을 8개밖에 확보하지 않았다는 뜻이 아니다.')
    tab([['원문 Table 1: 48개 자료의 중앙값','AUC','F1','Accuracy'],
         ['Tuned XGBoost','0.969','0.921','0.923'],
         ['TabPFN-ICD','0.967','0.899','0.902'],
         ['Default XGBoost','0.953','0.893','0.894'],
         ['TabPFN (무작위 context)','0.951','0.847','0.844']])
    p('ICD는 기본 TabPFN과 기본 XGB보다 높았으나 튜닝한 XGB를 넘지는 못했다. 이 수치는 자료별 점수의 중앙값이고, 원문 F1을 우리 실험의 attack-only Macro-F1과 같은 지표라고 가정하지 않는다. 따라서 이 논문이 뒷받침한 주장은 고정 context 크기에서 큰 라벨 pool의 정보를 더 잘 쓰는 것이었다.')
    cite('icd','§4, Table 1 p.4; Appendix A의 자료별 결과')
    para('선택·증류·FT·cache를 구분하면.', 'FIFO·무작위·유사도 정책은 실제 행 중 어떤 것을 남길지 정한다. ICD는 실제 라벨 train을 보고 가상의 context 특징을 학습한다. FT는 TabPFN 가중치 θ를 학습하고, cache 구성은 정해진 context의 중간 표현을 계산해 저장한다. 이 네 작업은 최적화 대상과 비용 발생 시점이 다르다.')
    tab([['방법','바뀌는 것','추가 학습/계산','반드시 셀 예산'],
         ['실제 행 선택','보관할 행 ID','선택·전처리·cache','사용한 라벨 pool / 보관 행 수'],
         ['ICD','합성 context의 특징값','반복 gradient 최적화·cache','전체 압축 원본 라벨 / 합성 context / validation'],
         ['가중치 FT','모델 θ','gradient·optimizer·재준비','FT에 사용한 라벨 / context / validation'],
         ['cache 준비','중간 표현','forward·저장','동일 context라도 재계산 여부와 빈도']])
    discussion('현재 질문이 실제 확보 가능한 공격 사례 2–5개의 효율이라면 ICD를 바로 주 방법으로 붙이는 것은 맞지 않을 수 있다. 큰 라벨 이력을 얻은 뒤 작은 메모리로 압축·유지하는 문제가 된다면, 같은 전체 라벨 pool과 context 크기에서 무작위·대표성 선택·ICD를 비교할 수 있다. 그때 증류 시간과 validation 라벨을 숨기지 않고 갱신 비용에 포함해야 한다. 숫자 flow 특징을 합성한다면 범주·정수·범위의 일관성을 확인하고, 합성 점을 실제 네트워크 패킷으로 해석하지 않아야 한다.')
    para('이 보고서에서 연결되는 후속 순서.', '먼저 §0의 극소 라벨 구간을 정상·기존 공격이 있는 단계적 시나리오에서 확인한다. 같은 보관량의 XGB와 PFN을 비교해 행 선택 효과와 모델 효과를 분리한다. 그다음 클래스별 실패가 prior·지역성·표현·expert 오류 중 어디에 해당하는지 진단한다. 라벨 이력이 커져 압축이 필요해졌을 때 ICD와 같은 context 최적화를 별도 비용 대조로 추가하는 순서가 타당하다.')

def appendix():
    raw(r'\clearpage\appendix')
    section('학습곡선의 상세 수치')
    p('주 곡선의 모든 점을 동일한 seed 43–47로 집계했다. F1과 표준편차는 %, 모델 차이는 %p, 시간은 초 단위다. 표준편차는 seed 간 변동이며 통계적 동등성 또는 신뢰구간을 의미하지 않는다. 클래스별 셀의 미반올림 값과 출처는 evidence/references/20261021/report_class_values_20261022.json에 보관했다.')
    summary=pd.read_csv(RUN/'summary.csv')
    for ds,title in [('cic2018','CIC2018'),('toniot','ToN-IoT')]:
        sub(title+' — 평균±표본 표준편차와 비용')
        rows=[['공격당 n','XGB F1 (%)','PFN F1 (%)','PFN−XGB (%p)']]
        costs=[['n','XGB 준비 (초)','PFN 준비 (초)','XGB 추론 (초)','PFN 추론 (초)']]
        for n in sorted(summary[summary.dataset==ds].n.unique()):
            q=summary[(summary.dataset==ds)&(summary.n==n)].set_index('method');x=q.loc['xgb'];t=q.loc['pfn']
            assert x.seeds==t.seeds==5
            rows.append([f'{n:,}',f'{100*x.mean_f1:.2f} ± {100*x.sd_f1:.2f}',f'{100*t.mean_f1:.2f} ± {100*t.sd_f1:.2f}',f'{100*(t.mean_f1-x.mean_f1):+.2f}'])
            costs.append([f'{n:,}',f'{x.mean_preparation:.3f}',f'{t.mean_preparation:.3f}',f'{x.mean_inference:.3f}',f'{t.mean_inference:.3f}'])
        tab(rows);tab(costs)
    sub('균형 상한을 넘긴 추가 탐색 — seed 43만, 별도 조건')
    p('희귀 공격은 train 전량을 유지하고 다른 공격만 늘린 조건이다. CIC의 web은 450개, ToN의 ransomware는 2,151개로 고정했다. ToN MITM은 목표 3,200에서 3,200개, 목표 6,400에서 가용 전량 3,285개였다. 나머지 공격은 목표 n개씩 제공하므로 이 표는 균형 학습곡선과 클래스 비율이 다르다.')
    tab([['데이터','목표 n','전체 train/context','XGB F1 (%)','PFN F1 (%)'],
         ['CIC2018','800','4,450','90.43','86.31'],
         ['CIC2018','1,600','8,450','93.04','86.42'],
         ['CIC2018','3,200','16,450','92.98','86.67'],
         ['ToN-IoT','3,200','27,751','93.06','93.03'],
         ['ToN-IoT','6,400','50,236','94.47','92.00']])
    p('CIC의 XGB는 목표 1,600→3,200에서 거의 유지됐고 PFN도 86%대였지만, 단일 seed와 변경된 비율만으로 최종 수렴을 확정하지 않는다. ToN은 목표 6,400에서 XGB가 더 높아졌고 PFN은 낮아졌다. 큰 context에서의 오분류와 표현·prior 효과를 따로 진단할 필요가 있으며, 단순 표본 증량만을 해법으로 가정하지 않는다.')

    raw(r'\clearpage')
    section('원문 재확인 위치와 재현 근거')
    sub('LotBoNC에서 구현 전에 대조할 표기')
    tab([['위치','원문에서 확인한 문제','이 보고서의 처리'],
         ['§4.4 Eq.(3)–(5), Algorithm 1','OT 목적함수와 kernel의 부호; α·β의 행/열 차원이 서로 맞지 않음','Eq.(3) 기준으로 차원 일치하는 설명식을 제시; 원문 구현의 정정본이라고 주장하지 않음'],
         ['§4.5 Eq.(6), Appendix A.1 Eq.(23)','Prior 항이 본문과 부록에서 반대 부호로 읽힘','Prototype 분류 핵심과 prior 갱신을 분리해 설명; 최종 loss 재현은 코드 확인 필요'],
         ['§5.1, Table 1 p.7','CTU-13 표 2.8M, 본문 1.3M; known/novel 비 γ와 클래스별 배분식 불명확','표본 수·지수 배분을 우리 설정에 그대로 복제하지 않음'],
         ['§4.2, §5.3','Payload/burst 입력 설명과 64-byte·header/payload 처리 표현의 대조 필요','PCAP 처리기는 별도로 검증하고 실제 입력 예시·제외율을 남김']])
    p('특히 prior를 M-step 안에서 상수로 취급한다면 softmax 밖에 더한 log-prior 항은 encoder gradient를 직접 바꾸지 않는다. 따라서 원문이 말하는 prior 보정 효과의 경로는 배정 A와 prior 갱신, 실제 loss 구현을 함께 확인해야 한다. 이 지적은 모델의 효과를 부정하는 결론이 아니라 재현 시 잘못된 식을 구현하지 않기 위한 확인 항목이다.')
    raw(link('https://github.com/watsonhu-hpi/LotBoNC','원문 기재 코드 주소')+' — 2026-10-21 검토 기록에서는 공개 접근이 되지 않아 원문 수식과 구현을 대조하지 못했다.\n')
    sub('실험과 자료의 원시 출처')
    raw(r'\sourcepath{'+str(RUN.relative_to(ROOT))+'}')
    p('summary.csv는 주 곡선·부록 표, per_class.csv는 클래스별 heatmap, values.csv와 results.json은 개별 실행 근거다. EXP86의 동일 2–20개 표본 연결은 exp86_continuity_check.json에서 확인했다. 정상 query 제거에 따른 batch 구성 차이로 기존 점수와 최대 약 0.0069%p 차이가 있어, 이 보고서는 EXP87의 일관된 attack-only 집계를 사용한다.')
    raw(r'\sourcepath{'+str(EXT.relative_to(ROOT))+'}')
    raw(r'\sourcepath{data/tabarena_examples/_metadata/download_manifest.json}')
    raw(r'\sourcepath{docs/lablog/evidence/references/20261021/report_figure_sources_20261022.json}')
    p('논문 그림은 위 JSON에 원본 파일 hash·PDF 페이지·Figure 번호·발췌 좌표와 링크를 남겼다. 학회가 확인된 논문은 학회명을, 프리프린트는 arXiv 또는 기술보고서로 표시했다. 검토 대상 논문과 데이터의 링크는 각 설명 및 그림 캡션에 붙였다. 파일명은 최초 작성일인 20261021_references를 유지하며, 본 개정일과 EXP87 분석일은 2026-10-22다.')

def main():
    results(); related(); tabarena(); lotbonc(); streams(); icd(); appendix()
    body='\n'.join([
        f'% !TeX root = {REPORT.name}','% !LW recipe=lablog XeLaTeX',
        r'\input{_support/style.tex}',
        r'\setlength{\parskip}{4pt}\setlength{\intextsep}{8pt}',
        r'\begin{document}',
        r'\reporttitle{최초 작성 2026-10-21 · 개정 2026-10-22}{극소수 공격 실험과 TabPFN 관련 연구 검토}',
        r'\recordnote{구성: 0. 극소수 실험 결과 → 1. 관련 모델 → 2. TabArena → 3. 평가 시나리오 → 4. Context 최적화. 학습곡선의 상세 수치는 부록에 배치했다.}',
        r'\setcounter{section}{-1}',*parts,r'\end{document}',''])
    REPORT.write_text(body)
    validation=build_report(REPORT)
    validation.update(written_at=datetime.datetime.now().astimezone().isoformat(),
        revision_date='2026-10-22',balanced_run=str(RUN.relative_to(ROOT)),
        extension_run=str(EXT.relative_to(ROOT)),balanced_jobs=200,seeds=[43,44,45,46,47],
        source_figure_manifest='report_figure_sources_20261022.json',
        class_values='report_class_values_20261022.json',
        source_pdf_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in EVID.glob('*.pdf')})
    validation['source_pdf_hashes']['lotbonc.pdf']=hashlib.sha256((BASE/'evidence/20261015/lotbonc.pdf').read_bytes()).hexdigest()
    (EVID/'review_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    log=BASE/'log.md';lines=log.read_text().splitlines();prefix='- 2026-10-21 · **문헌·데이터 검토**'
    line=prefix+' · 10/22 개정: EXP86–87 학습곡선·PFN F1/차이 heatmap·비용을 앞에 배치하고 상세표를 부록으로 이동; 서베이 Figure 3, 모델별 원문 그림, TabArena 실제 CSV, LotBoNC·Streams·ICD 방법/수식·출처 링크 정리. [LaTeX](reports/20261021_references.tex)'
    hit=next((i for i,x in enumerate(lines) if x.startswith(prefix)),None)
    if hit is None:lines.append(line)
    else:lines[hit]=line
    log.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(validation=validation,structure=check()),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
