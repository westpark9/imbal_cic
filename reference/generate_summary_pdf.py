"""
Generate a Korean-readable PDF summary of imbalanced classification papers
and their applicability to the MoE intrusion-detection project.
Font: UnDotum (sans-serif Korean TTF, always available on this system).
"""

from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Font registration ──────────────────────────────────────────────────────────
FONT_REG  = "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/unfonts-core/UnDotumBold.ttf"
pdfmetrics.registerFont(TTFont("KOR",     FONT_REG))
pdfmetrics.registerFont(TTFont("KOR-B",   FONT_BOLD))

F   = "KOR"
FB  = "KOR-B"

# ── Color palette ──────────────────────────────────────────────────────────────
HDR_BG   = colors.Color(0.13, 0.22, 0.42)
HDR_FG   = colors.white
C_VH     = colors.Color(0.72, 0.93, 0.72)   # 매우 높음 – green
C_H      = colors.Color(0.85, 0.96, 0.85)   # 높음 – light green
C_M      = colors.Color(1.00, 0.96, 0.78)   # 중 – yellow
C_L      = colors.Color(0.97, 0.88, 0.88)   # 낮음 – pink
ROW_ODD  = colors.Color(0.96, 0.97, 1.00)
ROW_EVEN = colors.white
GRID_C   = colors.Color(0.68, 0.68, 0.68)

PRIORITY_COLOR = {"매우 높음": C_VH, "높음": C_H, "중": C_M, "낮음": C_L}

# ── Paragraph styles ───────────────────────────────────────────────────────────
def ps(name, font=F, size=7.5, leading=10.5, align=TA_LEFT, color=colors.black):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                          alignment=align, textColor=color, wordWrap="CJK",
                          spaceAfter=0, spaceBefore=0)

S_HDR   = ps("H",  FB, 8,  11, TA_CENTER, HDR_FG)
S_BODY  = ps("B",  F,  7.5,10.5, TA_LEFT)
S_CENT  = ps("C",  F,  7.5,10.5, TA_CENTER)
S_TITLE = ps("T",  FB, 14, 18, TA_CENTER)
S_SUB   = ps("S",  F,  8.5,12, TA_CENTER, colors.Color(0.3,0.3,0.3))
S_NOTE  = ps("N",  F,  7.5,11, TA_LEFT,   colors.Color(0.4,0.4,0.4))
S_REC   = ps("R",  F,  8,  11.5, TA_LEFT)
S_RECC  = ps("RC", F,  8,  11.5, TA_CENTER)

def P(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)

# ── Table data ─────────────────────────────────────────────────────────────────
HEADERS = ["No.", "논문 / Method", "학회\n연도", "분류",
           "핵심 방법론", "MoE 차용 포인트", "프로젝트 적용 방안", "우선\n순위"]

ROWS = [
    ["01",
     "IMMAX\n(Balancing the Scales)",
     "ICML\n2025",
     "이론\n손실함수",
     "클래스별 신뢰 마진(ρ+, ρ−)을 도입한\n"
     "클래스 불균형 마진 손실 함수.\n"
     "Class-Sensitive Rademacher Complexity로\n"
     "일반화 경계를 이론적으로 도출.",
     "각 Expert마다 클래스 민감 마진 적용.\n"
     "Tail 클래스에 더 큰 ρ+ 부여해\n"
     "결정 경계를 넓힘.",
     "XGBoost에 class_weight 외에\n"
     "LDAM 변형 손실(마진 기반)을 적용.\n"
     "Tail Expert 손실 함수에 클래스별\n"
     "신뢰 마진 항 추가.",
     "중"],
    ["02",
     "Latent Score\nReweighting",
     "ICML\n2025",
     "샘플\n가중치",
     "VAE + 확산(Diffusion) 모델로\n"
     "잠재 공간의 P(X,Y) 추정.\n"
     "저밀도 영역(소수 클래스) 샘플을\n"
     "업가중해 균형 분포 달성.",
     "Expert 학습 시 tail 샘플의 가중치를\n"
     "밀도 추정 기반으로 자동 조정.\n"
     "라우터 학습에도 저밀도 샘플 선별.",
     "전처리 단계에서 KDE 또는 스코어 모델로\n"
     "클래스별 샘플 밀도 추정.\n"
     "저밀도 tail 샘플에 sample_weight 부여.",
     "중"],
    ["03",
     "PRIME\n(Proxy-based Repr.)",
     "ICML\n2025",
     "프로토\n타입",
     "클래스별 합성 프록시(Proxy) 점을\n"
     "균등 배치하고 샘플을 가장 가까운\n"
     "프록시에 정렬하는 표현 학습.\n"
     "PRW·CB·LDAM 손실과 통합 가능.",
     "Expert 라우팅 기준으로 클래스\n"
     "프로토타입(중심 벡터) 활용.\n"
     "샘플이 프록시에 가까울수록\n"
     "해당 Expert로 라우팅.",
     "각 공격 클래스의 Centroid + 공분산으로\n"
     "Mahalanobis 거리 계산.\n"
     "기존 OOD z-score 대신\n"
     "프로토타입 거리를 라우터 특징으로 사용.",
     "높음"],
    ["04",
     "DP + Class\nImbalance",
     "ICML\n2025",
     "이론\n프라이버시",
     "차분 프라이버시(DP) 환경에서\n"
     "오버샘플링·SMOTE의 감도 폭발 분석.\n"
     "Private 합성 데이터 + Weighted ERM이\n"
     "가장 효과적임을 실증.",
     "SMOTE의 구조적 한계 재확인.\n"
     "Weighted ERM = class_weight 기반\n"
     "XGBoost가 강력한 베이스라인임을 지지.",
     "SMOTE 대신 class_weight 또는\n"
     "scale_pos_weight 조정을 사용.\n"
     "이미 검토한 방향과 일치—\n"
     "추가 실험 불필요.",
     "낮음"],
    ["05",
     "DIRECT\n(Active Learning)",
     "ICML\n2025",
     "능동\n학습",
     "클래스 분리 최적 임계값(Optimal\n"
     "Separation Threshold)을 찾아\n"
     "경계 근방의 균형 잡힌 샘플을\n"
     "선택해 어노테이션.",
     "Expert별 결정 경계 근처의\n"
     "모호한 tail 샘플을 선별해\n"
     "추가 학습에 활용(Hard Example Mining).",
     "Expert 예측에서 confidence가 낮은\n"
     "tail 샘플(OvR margin 기반)을 식별.\n"
     "해당 Expert 학습 집합에 오버샘플링.",
     "낮음"],
    ["06",
     "DCE\n(Dual-Balance\nCollab. Experts)",
     "ICML\n2025",
     "MoE\n전문가 설계",
     "빈도-인식 Expert 3개\n"
     "(Many-shot·Balanced·Few-shot) + 전용 손실.\n"
     "클래스 중심·공분산으로 균형 가우시안\n"
     "의사특징 생성 후 동적 Expert 선택기 학습.",
     "MoE 아키텍처와 직접 대응.\n"
     "Head/Balanced/Tail Expert 분리 +\n"
     "Gaussian 샘플링 기반 라우터가\n"
     "routing error 문제를 구조적으로 우회.",
     "세 Expert(Many-shot·Balanced·Tail)를\n"
     "각각 다른 XGBoost로 구현.\n"
     "라우터는 클래스 중심+공분산 통계로\n"
     "소프트 가중치 계산.",
     "매우 높음"],
    ["07",
     "IP2SL\n(BNS + IP-DPP)",
     "NeurIPS\n2025",
     "샘플링\n표현학습",
     "Stage1: Balanced Negative Sampling(BNS)\n"
     "— 상호정보량 최대화 기반 표현.\n"
     "Stage2: IP-DPP — 행렬식 포인트\n"
     "프로세스로 정보적 균형 서브셋 선택.",
     "Expert 학습을 위한 균형 서브셋\n"
     "구성에 DPP 적용.\n"
     "정보 다양성 최대화 + 클래스 균형 달성.",
     "각 Expert 학습 데이터 구성 시\n"
     "DPP 기반 서브샘플링 적용.\n"
     "Majority 클래스 언더샘플링을\n"
     "랜덤 대신 DPP로 수행.",
     "중"],
    ["08",
     "MORE\n(Model Rebalancing)",
     "NeurIPS\n2025",
     "모델 파라미터\n설계",
     "모델 가중치를 W=W_g+W_t\n"
     "(일반 + tail 전용 저랭크)로 분해.\n"
     "Discrepancy 손실 + Sinusoidal\n"
     "재가중 스케줄로 tail 전용 파라미터 확보.",
     "각 Expert 내부에서\n"
     "tail 클래스 전용 서브모델(저랭크) 유지.\n"
     "학습 후반부에 tail Expert 가중치를\n"
     "시누소이달 스케줄로 강화.",
     "XGBoost 앙상블에서 tail 전용 트리 집합을\n"
     "별도 유지하는 방식으로 구현.\n"
     "학습률 스케줄링에 아이디어 차용.",
     "중"],
    ["09",
     "ImOOD\n(Imbalanced OOD Det.)",
     "NeurIPS\n2024",
     "OOD 탐지",
     "불균형 데이터에서 OOD 탐지기의\n"
     "클래스-인식 편향(class-aware bias) 분석.\n"
     "Tail ID 샘플이 OOD로 오탐되는 현상.\n"
     "통합 훈련 시 정규화항으로 편향 완화.",
     "현재 코드의 OOD z-score 게이팅이\n"
     "tail 샘플을 배제하는 핵심 실패 원인.\n"
     "클래스-인식 편향 보정이 필수.",
     "Expert OOD 임계값을 클래스별로 분리.\n"
     "Tail 클래스는 더 완화된 OOD 임계값 적용.\n"
     "클래스 사전확률 P(y)로\n"
     "OOD 스코어를 사후 보정.",
     "매우 높음"],
    ["10",
     "PRL\n(Hypernetwork\nDiverse Experts)",
     "NeurIPS\n2024",
     "MoE\n하이퍼네트워크",
     "하이퍼네트워크로 Pareto 전면\n"
     "(Head-Tail 트레이드오프)을 커버하는\n"
     "다양한 Expert 생성.\n"
     "디리클레 분포 선호도 벡터로 추론 시 제어.",
     "고정 Expert 대신 선호도 파라미터로\n"
     "Head-Tail 균형을 동적 조정.\n"
     "테스트 시 tail recall 요구에 맞춰\n"
     "라우팅 가중치 변경.",
     "class_weight 또는 scale_pos_weight를\n"
     "선호도 벡터로 파라미터화.\n"
     "Pareto 최적 모델 선별 후\n"
     "앙상블 가중치로 트레이드오프 제어.",
     "높음"],
    ["11",
     "NCMC\n(Neural Collapse\nMulti-Center)",
     "NeurIPS\n2024",
     "표현학습\n결정 경계",
     "소수 클래스 특징이 하나의 중심으로\n"
     "붕괴하는 'Minority Collapse' 문제.\n"
     "MSE형 손실로 클래스당 복수 중심\n"
     "유도. 일반화 분류 규칙(GCR) 제안.",
     "Tail 공격 클래스에 복수의\n"
     "클러스터 중심 할당.\n"
     "중심별 Expert로 라우팅해\n"
     "세분화된 전문화 달성.",
     "각 tail 공격 유형을 K-Means로\n"
     "하위 군집화 후 군집별 Expert 배치.\n"
     "라우팅 시 군집 중심 거리 사용.",
     "중"],
    ["12",
     "LLM-AutoDA",
     "NeurIPS\n2024",
     "데이터 증강\n자동화",
     "LLM이 Long-tail 데이터에 맞는\n"
     "증강 전략을 자동 탐색.\n"
     "검증 성능 향상을 보상 신호로\n"
     "증강 생성 모델 업데이트.",
     "네트워크 트래픽은 시각/언어 의미 없어\n"
     "직접 적용 어려움.\n"
     "AutoML 스타일의 하이퍼파라미터\n"
     "탐색 아이디어 참고.",
     "Expert 구성(클래스 배정, feature view)을\n"
     "검증 F1 향상을 목적으로\n"
     "자동 탐색하는 방향으로 간접 차용.",
     "낮음"],
    ["13",
     "EPIC\n(LLM Tabular Synthesis)",
     "NeurIPS\n2024",
     "합성 데이터\n테이블형",
     "LLM in-context learning으로\n"
     "균형 잡힌 테이블형 합성 데이터 생성.\n"
     "CSV 스타일 프롬프트 + 균형 클래스 그룹\n"
     "제시 + 고유 변수 매핑.",
     "테이블형 데이터를 다루는\n"
     "우리 도메인에 직접 적용 가능.\n"
     "LLM으로 tail 공격 트래픽\n"
     "합성 샘플 생성.",
     "GPT에 tail 공격 클래스 샘플을\n"
     "CSV 형태로 제공 후 합성 샘플 생성.\n"
     "Expert 학습 데이터 보강.\n"
     "단, 네트워크 피처 의미 해석 필요.",
     "중"],
    ["14",
     "DiffMatch\n(Semi-sup Segment.)",
     "ICLR\n2025",
     "생성 모델\n반지도",
     "반지도 학습을 조건부 이산 데이터\n"
     "생성 문제로 재구성.\n"
     "확산 모델로 Matthew Effect 완화.\n"
     "비편향 역확률 조정 추가.",
     "생성 모델 관점에서 Expert\n"
     "의사 레이블 품질 개선 아이디어.\n"
     "비편향 조정 개념을 Expert\n"
     "신뢰도 점수에 적용.",
     "세그멘테이션 특화 방법이라\n"
     "직접 적용 어려움.\n"
     "핵심 아이디어(생성 모델로\n"
     "class prior 재보정)만 참고.",
     "낮음"],
    ["15",
     "TaleOfTwoClasses\n(SupCon Binary)",
     "CVPR\n2025",
     "표현학습\n대조학습",
     "이진 불균형에서 SupCon이\n"
     "다수 클래스로 임베딩 붕괴.\n"
     "SAA·CAC 새 지표 +\n"
     "이진 특화 SupCon 전략으로 35% 향상.",
     "Expert 특징 공간의 표현 품질\n"
     "진단에 SAA/CAC 지표 활용.\n"
     "Tail 공격 클래스가 정상 클래스로\n"
     "붕괴하는지 진단.",
     "XGBoost 특징 중요도 대신\n"
     "SAA/CAC 유사 지표로 Expert별\n"
     "클래스 분리도 측정.\n"
     "Expert 재편성 기준으로 사용.",
     "중"],
    ["16",
     "Learning from Neighbors\n(Category Extrapolation)",
     "CVPR\n2025",
     "데이터 증강\n지식 이전",
     "LLM으로 tail 클래스의 의미적으로\n"
     "유사한 보조 카테고리 추가(web crawl).\n"
     "neighbor-silencing loss로 간섭 방지.\n"
     "추론 시 보조 클래스 마스킹.",
     "공격 분류체계에서 세분화된\n"
     "하위 공격 유형을 보조 클래스로 추가.\n"
     "Expert 내 클래스 경계 강화.",
     "DoS-Slowloris를 DoS-HTTP와\n"
     "DoS-TCP로 세분화 후 Expert에\n"
     "보조 서브클래스로 포함.\n"
     "추론 시 합산.",
     "중"],
    ["17",
     "ADR\n(Head-to-Tail\nData Calibration)",
     "CVPR\n2025",
     "데이터\n재조정",
     "분석 단계: 편향 파악.\n"
     "재조정 단계: 중복 Head 데이터 필터링.\n"
     "합성 단계: DDPM으로\n"
     "tail 데이터 합성.",
     "BENIGN 과다 데이터 필터링 +\n"
     "tail 공격 클래스 DDPM 합성을\n"
     "Expert 학습 전처리로 활용.",
     "BENIGN 샘플 중 중심 기반\n"
     "필터링으로 대표 서브셋 선별.\n"
     "tail 공격 유형 합성 데이터(TabDDPM) 생성\n"
     "후 Expert에 투입.",
     "높음"],
    ["18",
     "TAET\n(Two-Stage Adversarial)",
     "CVPR\n2025",
     "적대적 학습\n두 단계",
     "초기 안정화 단계 + 계층적 평준화\n"
     "적대 학습(HARL).\n"
     "Balanced Robustness 새 지표 도입.\n"
     "메모리·계산 효율 최적화.",
     "두 단계 학습 프레임워크\n"
     "(안정화→전문화)를 Expert 학습에 적용.\n"
     "1단계: 전체 데이터로 Anchor Expert 학습.\n"
     "2단계: 클래스별 특화 Expert 미세 조정.",
     "code_mat.py의 Baseline Probe →\n"
     "Expert 특화 구조를\n"
     "안정화(Stage1)+전문화(Stage2)로\n"
     "명시적 분리.",
     "중"],
    ["19",
     "GPA\n(Geometric Prototype\nAlignment)",
     "ICCV\n2025",
     "프로토타입\n증분 학습",
     "클래스 프로토타입을 단위 하이퍼구면에\n"
     "투영해 분류기 가중치 초기화.\n"
     "Dynamic Anchoring으로\n"
     "증분 업데이트 시 기하 일관성 유지.",
     "Expert 분류기 가중치를\n"
     "클래스 중심 프로토타입으로 초기화.\n"
     "Gradient competition 없이\n"
     "tail 클래스 특화 가능.",
     "각 Expert의 초기 분류기를\n"
     "프로토타입 기반으로 초기화 후 fine-tune.\n"
     "XGBoost base_score를 tail 클래스\n"
     "사전 확률로 설정.",
     "낮음"],
    ["20",
     "Multi-Granularity\nSemantics (SKCL)",
     "ICCV\n2025",
     "지식 이전\n대조학습",
     "LLM으로 다중 세분성 의미 기술 생성\n"
     "→ 유사도 행렬 구성.\n"
     "Semantic Knowledge-Driven Contrastive\n"
     "Learning(SKCL)으로 Head→Tail 지식 이전.",
     "공격 분류 계층(DoS > Slowloris 등)\n"
     "기반 의미 그래프로\n"
     "Expert 경계 설계 및\n"
     "tail 공격 유형 간 지식 이전.",
     "CIC-IDS 공격 분류체계를\n"
     "의미 그래프로 구성.\n"
     "유사 공격 유형(DoS 계열)의 Expert가\n"
     "특징 공간을 공유하도록 설계.",
     "중"],
    ["21",
     "NodeImport\n(Node Importance)",
     "KDD\n2025",
     "샘플\n중요도",
     "균형 메타셋으로 개별 노드 중요도를\n"
     "이분 최적화에서 도출.\n"
     "중요도 공식으로 라벨·비라벨·합성 노드를\n"
     "통합 필터링.",
     "Expert 학습 샘플의 중요도 점수 산출.\n"
     "고중요도 tail 샘플을 선별해\n"
     "Expert 학습 집합 구성.",
     "검증 성능을 기준으로 각 Expert의\n"
     "학습 샘플 중요도 점수 계산.\n"
     "상위 k% tail 샘플만\n"
     "Expert 재학습에 사용.",
     "중"],
    ["22",
     "ORD\n(Overlap Region\nDetection)",
     "AAAI\n2025",
     "합성 데이터\n테이블형",
     "k-fold RF 불일치로\n"
     "Majority의 경계 중첩 영역 식별.\n"
     "3-클래스 레이블로 생성 모델 학습.\n"
     "중첩 제거 후 합성 데이터로 학습.",
     "BENIGN과 tail 공격 사이\n"
     "경계 중첩 영역 제거 후 합성 데이터 품질 향상.\n"
     "XGBoost Expert 학습 데이터\n"
     "정제에 직접 활용.",
     "Expert 학습 전 CIC-IDS 데이터에서\n"
     "BENIGN ↔ 공격 경계 중첩 샘플\n"
     "(분류기 불확실성 높은 샘플) 제거.\n"
     "Expert 결정 경계 명확화.",
     "높음"],
    ["23",
     "DBM\n(Difficulty-aware\nMargin Loss)",
     "AAAI\n2025",
     "손실함수",
     "DBM = 클래스별 마진\n"
     "(m_C ∝ ρ_y^{-τ}) +\n"
     "인스턴스별 마진(각도 거리 기반).\n"
     "Hard Positive 샘플에 추가 마진 부여.",
     "Expert 손실 함수에 DBM 적용.\n"
     "Tail Expert 내에서도\n"
     "어려운 샘플에 더 큰 마진 할당.",
     "XGBoost의 scale_pos_weight를\n"
     "클래스 난이도 기반으로 동적 조정.\n"
     "오분류 경향 샘플(val 낮은 확률)에\n"
     "높은 sample_weight 부여.",
     "높음"],
    ["24",
     "BCE3S\n(Tripartite Synergistic\nLearning)",
     "arXiv\n2025",
     "손실함수\n분류기 설계",
     "BCE 기반 3중 시너지 학습:\n"
     "(1) BCE 조인트 학습(특징+분류기)\n"
     "(2) BCE 대조 학습(클래스 내 밀집)\n"
     "(3) BCE 균등 분리 학습(ETF 분류기).\n"
     "Softmax의 불균형 증폭 효과를 BCE로 우회.",
     "Expert 최종 분류기를\n"
     "Softmax → BCE로 전환.\n"
     "다수 클래스 로짓이 소수 클래스\n"
     "예측에 미치는 간섭 차단.",
     "XGBoost objective를\n"
     "binary:logistic(OvR)으로 변경.\n"
     "각 tail Expert를 단일 클래스\n"
     "이진 분류기로 구성.",
     "높음"],
]

# ── Build main table ───────────────────────────────────────────────────────────
COL_W = [0.75*cm, 3.5*cm, 1.4*cm, 2.0*cm,
          7.0*cm,  6.8*cm, 7.0*cm, 1.55*cm]

def build_main_table():
    hdr = [P(h, S_HDR) for h in HEADERS]
    data = [hdr]
    for row in ROWS:
        styled = []
        for i, cell in enumerate(row):
            s = S_CENT if i in (0, 2, 7) else S_BODY
            styled.append(P(cell, s))
        data.append(styled)
    return data

main_data = build_main_table()
main_tbl  = Table(main_data, colWidths=COL_W, repeatRows=1)

ts_main = [
    ("BACKGROUND", (0,0), (-1,0), HDR_BG),
    ("TEXTCOLOR",  (0,0), (-1,0), HDR_FG),
    ("GRID",       (0,0), (-1,-1), 0.4, GRID_C),
    ("LINEBELOW",  (0,0), (-1,0),  1.2, colors.white),
    ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ("TOPPADDING",    (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING",   (0,0), (-1,-1), 4),
    ("RIGHTPADDING",  (0,0), (-1,-1), 4),
]
for i, row in enumerate(ROWS):
    r = i + 1
    # alternating row bg
    bg = ROW_ODD if i % 2 else ROW_EVEN
    ts_main.append(("BACKGROUND", (0,r), (6,r), bg))
    # priority cell color
    pc = PRIORITY_COLOR.get(row[7], ROW_EVEN)
    ts_main.append(("BACKGROUND", (7,r), (7,r), pc))

main_tbl.setStyle(TableStyle(ts_main))

# ── Legend (compact, single row) ──────────────────────────────────────────────
def legend_row():
    items = [
        ("매우 높음", C_VH, "핵심 실패 원인 직접 해결 / MoE 구조와 완전 대응"),
        ("높음",     C_H,  "라우팅 또는 Expert 학습에 즉시 적용 가능"),
        ("중",       C_M,  "전처리·손실함수 보조 개선"),
        ("낮음",     C_L,  "간접 참고 또는 도메인 불일치"),
    ]
    cells = [P("【범례】", ps("LL", FB, 8, 11, TA_CENTER))]
    widths = [1.8*cm]
    for label, col, desc in items:
        cells.append(P(f"{label}: {desc}", ps("LD", F, 7.5, 10)))
        widths.append(8.4*cm)
    tbl = Table([cells], colWidths=widths)
    ts = [
        ("VALIGN",       (0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
        ("BOX",          (0,0),(-1,-1), 0.5, GRID_C),
        ("BACKGROUND",   (0,0),(0,-1), colors.Color(0.92,0.92,0.92)),
    ]
    for idx, (_, col, _) in enumerate(items):
        ts.append(("BACKGROUND", (idx+1,0),(idx+1,0), col))
    tbl.setStyle(TableStyle(ts))
    return tbl

# ── Recommendation table (page 2) ─────────────────────────────────────────────
REC_HEADERS = ["우선순위", "논문", "구체 적용 방안", "예상 효과"]
REC_ROWS = [
    ["매우 높음", "06 DCE\n(Dual-Balance Experts)",
     "Many-shot / Balanced / Tail XGBoost Expert 3개 분리.\n"
     "라우터: 클래스 중심+공분산 통계 기반 Gaussian 소프트 가중치.\n"
     "Tail Expert는 threshold 이하 클래스만 학습.\n"
     "손실: Tail Expert에 LDAM 또는 Focal Loss 적용.",
     "Routing error가 tail 샘플을 잘못된 Expert로 보내는\n"
     "문제를 구조적으로 해소.\n"
     "클래스 빈도 그룹별 특화로\n"
     "expert composition imbalance 완화."],
    ["매우 높음", "09 ImOOD\n(Imbalanced OOD Detection)",
     "기존 OOD z-score 임계값을 클래스별로 분리 설정.\n"
     "Tail 클래스는 더 완화된(높은) OOD 임계값 적용.\n"
     "클래스 사전 확률 P(y)로 Mahalanobis 스코어를 사후 보정.\n"
     "코드 변경: code_mat.py ood_threshold를 dict로 변환.",
     "tail ID 샘플이 OOD 게이트에서 배제되는\n"
     "핵심 실패 원인 제거.\n"
     "val에서 tail recall 대폭 향상 예상.\n"
     "기존 OOD 게이팅 레이어에 최소 수정으로 적용 가능."],
    ["높음", "10 PRL\n(Hypernetwork Diverse Experts)",
     "class_weight 벡터를 선호도 파라미터로 파라미터화.\n"
     "Pareto 전면 탐색: 다양한 (tail 강조, head 강조) Expert 학습.\n"
     "앙상블 가중치로 Head-Tail 트레이드오프를 제어.\n"
     "테스트 데이터 분포 추정 후 최적 선호도 벡터 선택.",
     "고정 Expert 대신 tail recall / precision 간\n"
     "유연한 트레이드오프 달성.\n"
     "데이터셋별(CIC-2017/2018/UNSW) 최적점 적응.\n"
     "chrono/file split에서도 robust한 Expert 배치 가능."],
    ["높음", "03 PRIME\n(Prototype-based Routing)",
     "각 공격 클래스의 Centroid + 공분산으로 프록시 구성.\n"
     "샘플을 가장 가까운 프록시 거리로 Expert에 배정.\n"
     "기존 LogisticRegression 라우터를 프로토타입 거리로 대체.\n"
     "val에서 라우터 정확도를 클래스별로 평가해 재배정.",
     "분류 경계 근처 tail 샘플의 라우팅 정확도 향상.\n"
     "OOD z-score보다 해석 가능하고 안정적인 라우팅.\n"
     "PRIME의 프록시 손실을 Expert 학습에 추가해\n"
     "클래스 중심 주변 특징 압축 효과."],
    ["높음", "17 ADR\n(Adaptive Data Calibration)",
     "BENIGN 과다 샘플 중 중심 기반 필터링으로 대표 서브셋 선별.\n"
     "TabDDPM으로 tail 공격 클래스 합성 샘플 생성.\n"
     "생성된 합성 샘플을 해당 Expert 학습 데이터에 추가.\n"
     "합성 비율은 val F1 기준으로 튜닝.",
     "Expert 학습 데이터 품질 개선.\n"
     "tail 클래스 샘플 수 증가로 Expert training imbalance 완화.\n"
     "BENIGN 편향을 줄이고 공격 클래스 표현 강화."],
    ["높음", "22 ORD\n(Overlap Region Detection)",
     "k-fold Random Forest 불일치 점수로 BENIGN↔공격\n"
     "경계 중첩 샘플(confusion zone) 식별 후 제거.\n"
     "중첩 제거 후 Expert 학습 데이터 구성.\n"
     "코드: 학습 전 전처리 단계에 ORD 필터 삽입.",
     "Expert 결정 경계 명확화.\n"
     "모호 샘플 제거로 confusion matrix 기반\n"
     "전문화 효과 향상.\n"
     "테이블형 데이터에 직접 적용 가능한 검증된 방법."],
    ["높음", "23 DBM\n(Difficulty-aware Margin Loss)",
     "val에서 낮은 예측 확률을 받은 tail 샘플에\n"
     "높은 sample_weight 부여.\n"
     "XGBoost scale_pos_weight를 클래스 빈도 × 난이도로 동적 조정.\n"
     "클래스별 마진 + 인스턴스별 마진 조합.",
     "Hard tail 샘플 집중 학습으로 tail recall 향상.\n"
     "기존 class_weight보다 세밀한 샘플별 난이도 반영.\n"
     "기존 XGBoost 파이프라인에 sample_weight 추가만으로\n"
     "적용 가능 — 구현 난이도 낮음."],
    ["높음", "24 BCE3S\n(OvR Binary Expert)",
     "각 tail Expert를 Softmax 다중 분류 대신\n"
     "OvR(One-vs-Rest) 이진 XGBoost 분류기 집합으로 구성.\n"
     "objective='binary:logistic'으로 변경.\n"
     "Ensemble: 각 이진 분류기 확률을 정규화 후 argmax.",
     "Softmax 분모에서 다수 클래스가 tail 예측을\n"
     "억제하는 구조적 문제 해소.\n"
     "tail 클래스별 독립 최적화 가능.\n"
     "BENIGN이 분모를 지배하는 현상 차단."],
]

REC_COL_W = [2.2*cm, 4.8*cm, 15.5*cm, 13.5*cm]

def build_rec_table():
    hdr = [P(h, S_HDR) for h in REC_HEADERS]
    data = [hdr]
    for row in REC_ROWS:
        styled = []
        for i, cell in enumerate(row):
            s = S_RECC if i == 0 else S_REC
            styled.append(P(cell, s))
        data.append(styled)
    return data

rec_data = build_rec_table()
rec_tbl  = Table(rec_data, colWidths=REC_COL_W, repeatRows=1)

ts_rec = [
    ("BACKGROUND", (0,0), (-1,0), HDR_BG),
    ("TEXTCOLOR",  (0,0), (-1,0), HDR_FG),
    ("GRID",       (0,0), (-1,-1), 0.4, GRID_C),
    ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ("TOPPADDING",    (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING",   (0,0), (-1,-1), 5),
    ("RIGHTPADDING",  (0,0), (-1,-1), 5),
]
for i, row in enumerate(REC_ROWS):
    r  = i + 1
    pc = PRIORITY_COLOR.get(row[0], ROW_EVEN)
    ts_rec.append(("BACKGROUND", (0,r),(0,r), pc))
    bg = ROW_ODD if i % 2 else ROW_EVEN
    ts_rec.append(("BACKGROUND", (1,r),(-1,r), bg))

rec_tbl.setStyle(TableStyle(ts_rec))

# ── Assemble document ──────────────────────────────────────────────────────────
OUTPUT = "/home/user/Desktop/imbal_cic/reference/paper_summary_moe.pdf"
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=landscape(A3),
    leftMargin=1.2*cm, rightMargin=1.2*cm,
    topMargin=1.5*cm,  bottomMargin=1.5*cm,
)

elements = []

# ── Page 1 ────────────────────────────────────────────────────────────────────
elements.append(P(
    "Top Conference 논문 분석: Imbalanced Classification → MoE 프로젝트 차용 전략",
    ps("PT", FB, 13.5, 18, TA_CENTER)
))
elements.append(Spacer(1, 0.15*cm))
elements.append(P(
    "대상 데이터셋: CIC-IDS2017/2018 · UNSW-NB15 | 목표: 극심한 불균형 다중 분류(IR > 1000)에서 XGBoost Baseline을 능가하는 MoE 설계",
    ps("PS", F, 8, 11, TA_CENTER, colors.Color(0.3,0.3,0.3))
))
elements.append(Spacer(1, 0.3*cm))
elements.append(legend_row())
elements.append(Spacer(1, 0.3*cm))
elements.append(main_tbl)

# ── Page 2 ────────────────────────────────────────────────────────────────────
elements.append(PageBreak())
elements.append(P(
    "핵심 차용 전략 상세 — 우선순위 높음 이상 (8개 논문)",
    ps("PT2", FB, 13, 17, TA_CENTER)
))
elements.append(Spacer(1, 0.3*cm))
elements.append(rec_tbl)
elements.append(Spacer(1, 0.5*cm))
elements.append(P(
    "※ 우선순위 '매우 높음' 논문(06 DCE · 09 ImOOD)이 현재 프로젝트의 핵심 실패 원인 "
    "(routing error compounding, OOD gate tail exclusion)을 직접 해결합니다. "
    "DCE의 frequency-aware Expert 분리 구조를 새 코드의 뼈대로 채택하고, "
    "ImOOD의 클래스-인식 OOD 임계값 보정을 게이팅 레이어에 즉시 적용하는 것을 권장합니다. "
    "| 분석 기준일: 2026-04-24",
    S_NOTE
))

doc.build(elements)
print(f"PDF saved: {OUTPUT}")
