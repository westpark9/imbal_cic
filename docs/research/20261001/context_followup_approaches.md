# Context 개선 후속 접근

정리일: 2026-10-01. 사용자와 합의한 네 가지 검토 방향이며 **아직 실행하지 않은 계획**이다. 이번 작업은 한국어 랩미팅 슬라이드 수정과 계획 기록이며 새 모델 실험은 실행하지 않았다.

현재 결과는 **공통 anchor + residual 군집별 특화 block**이라는 한 가지 context 구성 방법의 예비 결과다. CIC2018·ToN, seed 43, K=2/4/6/8 및 S/V 포함 결과는 EXP63에 저장되어 있다. 현재 구성을 최종 방법으로 확정하지 않는다.

연구 문제는 제한된 TabPFN context 예산에 정상·공격을 구분하는 정보를 얼마나 담을 수 있는가다. Global과 expert 양쪽의 context 설계에 적용 가능하다. 현재 성능 저하를 window 라벨링의 불가피한 결과로 단정하지 않는다. 향후 expert의 오류 정답을 수집해야만 context를 만들 수 있는 절차는 이번 후속 접근에서 제외한다.

## 1. Benign을 다양하게 구성

- 프로토콜·통신량·시간 특성 등 서로 다른 정상 트래픽 패턴의 실측 사례를 context에 포함
- 같은 클래스 할당량에서 유사 사례의 반복을 줄이고 여러 정상 패턴을 보존
- 비교: context 크기·클래스별 행 수 고정, 기존 선택과 패턴 다양성을 고려한 선택
- 확인할 효과: 공격별 F1과 Benign 오탐의 변화, context 추출에 따른 결과 변동
- 준비: train feature에서 범주형·연속형 특성을 고려한 군집/대표 표본 선택 기준 정의

## 2. LITO 방식의 tail 합성

- Feature 일부를 가린 뒤 tail 조건으로 재생성하고 생성 결과를 검증
- LITO 원문의 majority-to-minority imputation과 self-authentication을 참고
- IDS 적용에서는 프로토콜과 feature 간 관계, 이산값·최소/최대 관계 등 유효성 조건 반영
- 같은 tail 할당량에서 실측 사례 반복과 합성 사례를 비교하여 클래스 비중과 합성의 효과 분리
- 생성 모델의 자기 검증은 실제 공격 라벨의 정당성을 보장하지 않으며, 실측 validation에서 효과 확인
- Global이 맞힌 합성 표본만 남기는 방식은 Global의 약점을 답습할 수 있으므로 자동 채택하지 않음
- 구체적 생성기, 합성 비율, feature 마스킹 규칙은 아직 미정

참고: June Yong Yang, Geondo Park, Joowon Kim, Hyeongwon Jang, Eunho Yang. **Language-Interfaced Tabular Oversampling via Progressive Imputation and Self Authentication**, ICLR 2024. [논문](https://proceedings.iclr.cc/paper_files/paper/2024/file/5d54d2df6ec8f7b920aa0fec9a6d1b2e-Paper-Conference.pdf)

## 3. Feature 학습으로 embedding 추가

- 여러 훈련 표본과 클래스의 관계를 이용해 분류에 유용한 encoder 학습
- 적용 시에는 flow 한 행을 encoder에 넣어 그 행의 embedding을 산출
- 기존 feature에 embedding을 추가하고 context 행·query 행 모두 같은 변환 적용
- 여러 행을 하나의 embedding으로 합치는 방법과 구분
- 비교: 원래 feature와 원래 feature + 학습 embedding, context 크기와 train 정보 사용 범위 명시
- Prototype-label embedding 논문의 분류기를 그대로 재현하는 것과 TabPFN 입력에 embedding을 추가하는 응용을 구분

참고: Manuel Lopez-Martin, Antonio Sanchez-Esguevillas, Juan Ignacio Arribas, Belen Carro. **Supervised contrastive learning over prototype-label embeddings for network intrusion detection**, Information Fusion 79 (2022), 200–228. [논문](https://doi.org/10.1016/j.inffus.2021.09.014) · [사용자 지정 공개 코드](https://github.com/mlopezm/Supervised-contrastive-learning-over-prototype-label-embeddings)

## 4. Context 자체의 최적화

- TabPFN을 고정하고, 훈련 query를 잘 분류하도록 작은 context의 값을 최적화
- 원본에서 표본을 고르는 방법과 학습된 context를 비교
- 현재 v3에서 context gradient 지원, 전처리 경로, 메모리 비용을 먼저 확인
- 학습된 context는 실제 네트워크 flow와 다를 수 있는 분류용 압축 표현이며 실측 표본과 구분
- 기존 비교 방법 DistPFN의 posterior 보정과는 다른 방법

참고: Junwei Ma, Valentin Thomas, Guangwei Yu, Anthony Caterini. **In-Context Data Distillation with TabPFN**, arXiv:2402.06971 (2024). [논문](https://arxiv.org/abs/2402.06971) · [사용자 지정 저자 연구 페이지](https://valthom.github.io/). ICLR 2024 ME-FoMo Workshop 발표 표기는 저자 페이지에서 확인했다.

## 공통 비교 원칙

- 우선 단일 seed·대표 K=4에서 개별 변경 효과 확인, 여러 접근을 처음부터 한꺼번에 결합하지 않음
- Context 크기와 클래스별 할당량을 고정한 비교를 기본으로 하되, 변경된 학습 정보·총예산은 명시
- 생성·표현 학습·context 최적화에는 훈련 데이터만 사용, validation으로 설정 확인, 평가 데이터로 context를 학습하지 않음
- 유망한 context가 나오면 그 bank에 맞춰 S/V를 다시 학습
- 전체 Macro-F1, 클래스별 Precision/Recall/F1, Benign 오탐과 비용을 함께 확인
- 추가 반복 seed와 독립 평가가 필요한 예비 단계이며, 현 holdout 최고값으로 방법·K를 선정한 최종 성능 주장은 하지 않음

랩미팅 반영: 한국어 슬라이드 9장(1·2번), 10장(3·4번). HTML과 영어 슬라이드는 이번 요청에서 수정하지 않았다.
