# EXP46 — ToN global context 2×2 비교 사전 고정

승인: 2026-09-15 사용자 지시 “일단 global 모델에 대한 실험은 지금 진행해도 될 것같음.”

## 목적과 범위

기존 C0와 자연 비율 context의 차이에 섞인 **class quota**와 **추첨 시간 구간**을 분리한다. 이번 실행은 global 비교이며 expert·scorer·verifier를 학습하거나 선택하지 않는다. 기존 실험 스크립트는 변경하지 않는다.

## 고정 조건

- 데이터: 원본 uncapped ToN-IoT. 기존 scenario/class별 chronological train/validation/test 유지; 중복 삭제·재라벨 없음.
- Context: 각 100,000행, seed 42, frozen TabPFN-v3, n_estimators=4, fit_with_cache, batch=500,000.
- 자연 quota: 전체 train의 비율로 한 번 결정. 같은 quota를 early/full에 사용.
- 기존 quota: 기존 C0의 앞 절반 pool에서 결정된 실제 클래스 수를 early/full에 동일하게 사용. 전체 pool에서 희소 클래스 수가 더 늘어나지 않도록 고정.
- 각 클래스 추첨 seed: 42+990+class_id. 같은 window에서 quota만 바뀌면 해당 클래스의 추첨 순서는 동일.
- 각 arm은 별도 프로세스. PFN 생성 직전에 Python·NumPy·Torch RNG를 동일하게 초기화. 모든 arm의 모델 및 split 평가 순서도 동일.
- 같은 context의 XGBoost: 300 trees, depth 8, learning rate .05, subsample/colsample .8, seed 42, CPU threads=8.
- Full validation 5,504,048행과 full test 5,504,060행을 모두 평가. 네 조건을 결과를 보기 전에 고정하며 test를 보고 실행 조건을 바꾸지 않는다.
- 초기 비교는 seed 42 한 번이다. 재현성 우위 확정은 다중 seed 확인 전까지 보류한다.

## 실행 순서

| 순서 | Arm | Pool | Quota |
|---|---|---|---|
| 1 | early_natural | 각 class×scenario의 train 앞 50% | 자연 |
| 2 | full_natural | 전체 train | 자연 |
| 3 | early_legacy | 각 class×scenario의 train 앞 50% | 기존 C0의 실제 class 수 |
| 4 | full_legacy | 전체 train | 3과 동일 |

Context 네 개와 evaluation arrays를 먼저 생성한다. 각 worker는 같은 context의 XGBoost validation/test → TabPFN validation/test 순서다. 각 arm 완료 시 루트 summary를 갱신한다.

## 판정

- 주지표: 모든 클래스 macro-F1. 함께 볼 지표: mitm/ransomware 및 전체 클래스 P/R/F1, benign FPR, FP의 목적 클래스, inference 시간·GPU peak.
- Window 효과: full_natural−early_natural, full_legacy−early_legacy.
- Quota 효과: early_natural−early_legacy, full_natural−full_legacy.
- 두 window 효과의 차이로 상호작용을 읽는다. 순위 하나로 원인을 단정하지 않는다.
- EXP45의 0.6740/0.5415는 과거 참고치다. 이번 비교는 동일 seed 초기화·프로세스 순서로 새로 만든 동일 실험 안에서 계산한다.
- 이 실험은 quota 전체 변경의 효과를 본다. benign 비중과 공격 클래스 간 배분의 개별 효과는 분리하지 않는다.

## 산출물과 검증

- [실행 코드](../../../tabpfn/scripts/nfv3_v3_exp46_ton_global_context.py)
- [과학적 불변조건 테스트](../../../tabpfn/tests/test_exp46_ton_global_context.py): window 간 quota 동일, quota 변화 시 클래스 추첨의 중첩, 불가능 quota 거절, sklearn과 지표 일치 — 4개 통과.
- GPU synthetic smoke: 두 모델×validation/test, 여러 batch, float32 posterior·argmax 예측 일치 확인. 성능 결과로 인용하지 않는다.
- 실행 폴더: source 사본·hash, args, context/outer split 인덱스, class·시간 bin 구성, float32 posterior, hard prediction, confusion matrix, per-class/summary CSV.
- 상태: RUNNING.json → COMPLETE.json 또는 ERROR.json. 전체 arm이 끝나기 전 완료로 기록하지 않는다.

예상 시간: EXP45에서 PFN 한 모델의 full test 추론은 약 15분이었다. Full validation까지 포함한 네 조건은 준비·XGB 시간을 더해 **약 2시간 이상**으로 예상하며 실제 진행 속도를 기준으로 갱신한다.
