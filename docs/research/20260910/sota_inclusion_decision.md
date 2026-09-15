# HINT·LoGIC·Xiaomi-TabLDM 비교군 판단

확인일: 2026-09-10. 아래는 비교 실험의 우선순위 판단이며, CIC2018에서 세 방법의 성능을 측정한 결과가 아니다.

## 포함 판단

| 방법 | 판단 | 근거와 비교 시 역할 |
|---|---|---|
| Xiaomi-TabLDM | **주요 성능 비교군에 포함** | 공식 분류 구현과 classifier checkpoint가 공개되어 있다. 새로운 tabular backbone의 성능을 비교할 수 있어 프리프린트라는 이유로 제외할 필요가 없다. 다만 논문이 강조한 회귀 성과를 IDS 다중분류 SOTA 달성의 증거로 쓰지는 않는다. [공식 코드](https://github.com/xiaomi-research/xiaomi-tabldm), [기술 보고서](https://arxiv.org/abs/2609.03880) |
| HINT | **조건부 호출 방식의 비교군에 포함** | kNN의 신뢰도로 TabPFN 호출 여부를 결정하고 검색한 이웃을 context로 제공한다. 저비용 예측과 선택적 TFM 호출이라는 구조가 우리 라우팅 주장과 직접 연결된다. 다만 원문은 예측 후 라벨을 반영하는 streaming 평가다. 고정 train/test의 우리 실험에 옮긴 버전은 HINT-style adaptation으로 표시한다. [원문 §3–4](https://arxiv.org/html/2609.07956v1) |
| LoGIC | **관련 연구에는 포함, 현재 주 성능표에서는 제외** | GraphLand의 node classification을 위한 graph-aware context 구성이다. 그래프 구조 검색과 unlabeled halo가 입력 프로토콜에 포함되므로 NetFlow 행 분류에 그대로 대입하기 어렵다. Feature·coverage 검색만 가져오면 자체 adaptation이며 원본 LoGIC 전체의 비교 결과로 부를 수 없다. [원문 §III–V](https://arxiv.org/html/2609.05955v1) |

여기서 “SOTA 비교군에 포함”은 **비교할 만한 최근 방법으로 채택**한다는 뜻이다. 우리 IDS 설정의 최고 성능이 이미 입증되었다는 뜻이 아니다. HINT·LoGIC은 원문과 이번 검색에서 저자 공식 저장소를 확인하지 못했으며, 코드가 없다고 확정하는 것은 아니다.

## Xiaomi의 MoE와 우리 expert의 겹침

| 기준 | Xiaomi-TabLDM | 현재 제안 모델 |
|---|---|---|
| Expert의 단위 | Transformer 내부의 FFN/MLP | 서로 다른 context로 예측하는 PFN 실행 |
| Router가 고르는 것 | Row token에 적용할 내부 FFN expert | Query의 global 예측을 바꿀 후보 context expert |
| Shared 부분 | 항상 활성화되는 shared FFN expert | 공통 global 기준선과 expert context의 anchor |
| 학습 | 합성 데이터 사전학습 중 router·expert 파라미터 학습 | PFN 가중치를 고정하고 실제 IDS route 자료로 scorer/verifier 학습 |
| 교정 절차 | 내부 expert의 표현을 통합해 모델 출력 생성 | Global과 선택 expert의 출력 비교 후 채택/거절 |

Xiaomi의 MoE는 일부 ICL layer의 FFN을 대체하며, shared expert와 token별 top-k routed expert를 결합한다. 이는 분명 관련된 조건부 계산 구조다. 하지만 내부 표현을 만드는 expert와 최종 라벨을 교정하는 context expert는 학습 대상과 결정 단위가 다르다. [구조·사전학습 원문 §2.1–2.2](https://arxiv.org/html/2609.03880v2)

따라서 **“MoE를 썼다”는 주장만으로 우리의 독창성을 세우면 안 된다.** 입증할 대상은 residual에서 만든 context bank, 호출 전 교정 가능성 예측, 호출 후 훼손 제어가 같은 기준선 위에서 추가 교정을 만드는가이다. EXP39의 최종 ΔF1=0으로는 그 성능 기여를 아직 입증하지 못했다.

## 실제 비교 프로토콜

1. **Backbone 비교:** 같은 train에서 고정한 C0 100k 행과 같은 test를 Xiaomi classifier·TabPFN global에 제공한다. 동일 split, feature, class mapping과 validation 규칙을 사용한다. Ensemble 수가 같더라도 비용이 같지는 않으므로 실제 시간·최대 메모리도 보고한다. 공식 Xiaomi 기본 설정과 맞춘 예산 설정을 구분한다.
2. **전체 시스템 비교:** Ours는 C0 외에 expert bank와 route의 라벨도 사용한다. 따라서 Xiaomi에 C0만 주고 Ours의 추가 정보 사용을 숨긴 채 공정한 전체 모델 비교라고 하면 안 된다. 접근 가능한 train pool·라벨 예산과 검증 예산을 맞춘 비교를 추가한다.
3. **HINT 방식 비교:** 현재 고정 split에서는 검색 memory를 train으로만 만들고 threshold를 validation에서 선택한다. Test 정답으로 memory나 gate를 갱신하지 않는다. 원문의 online update까지 재현하려면 모든 모델을 동일한 prequential 프로토콜에서 별도로 비교한다.
4. **지표:** Macro-F1, tail 평균 F1, 클래스별 F1, benign FPR, global 대비 교정/훼손, 실제 호출량·시간·메모리를 함께 기록한다. 비용 감소만으로 교정 성능 개선을 대체하지 않는다.

이 판단으로 Xiaomi와 HINT의 비교 우선순위를 올리되, 이 문서 작성 중 새 baseline 학습이나 CIC2018 평가를 실행한 것은 아니다.
