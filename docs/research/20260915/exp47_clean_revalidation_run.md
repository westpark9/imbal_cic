# EXP47 — 모순 행 제거 후 CIC2018 기존 구조 재검증

사용자 실행 지시: “일단 모순 행 제거하고 새 실험 run해줘 eta 알려줘”.

## 상태

**최종 상태: 2026-09-15 19:23:03 KST 정상 완료, 총 73분 50초.** Global과 최종 시스템 macro-F1 0.810653, 최종 expert 채택 0건. 새 bank 교정 기회와 scorer/verifier 차단 원인은 [EXP47 결과 보고서](../20260916/exp47_results.md)에 정리했다. 아래 ETA는 착수 당시 예상 기록이다.

- 정제 완료: 약 **78초**. 데이터 manifest와 인덱스 검증 통과.
- 실제 seed 42 실행 시작: **2026-09-15 18:09:13 KST**.
- Controller PID: **1090967**. 순서: 기존 EXP31 기준 실행 → 기존 EXP39 `legacy` 정책 → 단계별 진단.
- 최초 ETA: **19:40–20:40 KST**(시작 후 약 1시간 30분–2시간 30분). 기존 EXP31 약 72분과 전체 expert 평가 비용을 참고했다. 새 bank의 K와 context 크기에 따라 달라질 수 있다.
- ToN EXP46은 계속 중단 상태다.

## 정제 결과

| 항목 | 행 수 |
|---|---:|
| 원래 CIC2018 | 20,115,529 |
| 삭제한 모순 그룹의 행 | **5,360,112** |
| 남은 행 | **14,755,417** |
| 정제 train | 8,666,430 |
| 정제 validation | 3,046,514 |
| 정제 test | 3,042,473 |

- 동일 입력·상반 모델 라벨이 있는 **20,609개 벡터 그룹 전체**를 제외했다. 나머지 동일 라벨 중복은 유지했다.
- Hash 후보에 대해 실제 float32·비유한 값 처리 후 벡터 일치를 확인했다. 발견된 hash collision 그룹 0개, 남은 모순 그룹 0개.
- 원본 pickle을 보존한다. 파생 데이터 버전은 원본 row ID와 `keep_idx/drop_idx`, 기존 split과 교차한 `train/val/test_idx`, fingerprint로 정의한다. 격리된 실험 loader가 이 인덱스만 반환하므로 제외 행은 global·expert·route·tune·cal·test에 들어가지 않는다.
- Outer split 간 행 이동은 0이다. 내부 역할은 정제된 각 split 안에서 기존 scenario별 분할 규칙을 적용한다.
- 모든 클래스가 train에 남았다. 예를 들어 infiltration은 train 30,143 / test 8,742행, web_attacks는 train 450 / test 127행이다. 고정 7개 클래스와 support를 함께 보고한다.

자료: [manifest](../../../data/derived/cic2018_conflict_free_fixed_split_20260915_180000/manifest.json), [클래스별 삭제 수](../../../data/derived/cic2018_conflict_free_fixed_split_20260915_180000/class_counts.csv), [scenario별 삭제 수](../../../data/derived/cic2018_conflict_free_fixed_split_20260915_180000/scenario_counts.csv).

## 실행 조건과 진단

기준 설정은 `20260904_010654_nfv3_cic2018_exp31_c0alloc/args.json`에서 가져왔다. 이는 0909 frozen bank의 원래 생성 설정이며, **과거 예측·bank·임계값은 재사용하지 않는다.**

- C0 100,000행, benign share 0.75, attack balanced, 같은 라벨 중복 유지. 정제 pool에서 다시 추첨.
- Frozen TabPFN-v3, n_estimators 4, fit_with_cache, seed 42. 기존 anchor·residual·K 선택·pruning·scorer·normgain verifier 설정 유지.
- 첫 기준 실행은 EXP31의 weighted-NLL 정책 선택을 그대로 사용한다. 추가로 기존 EXP39의 `legacy` arm 하나만 같은 새 bank에서 학습·평가해 macro-F1 정책 선택과 calibration/confirmation을 확인한다. 다른 EXP39 모델 변형은 실행하지 않는다.
- 원래 실행과의 args 차이는 정제 manifest, 산출물 디렉터리, **진단용 `dense_eval=True`**뿐이다. XGB 별도 full/C0 대조군 격자는 이번에 추가하지 않았다.
- 원본 EXP31 파일을 변경하지 않고 실행 사본에 관측 callback 9개를 삽입한다. Callback을 제거하면 원본과 AST가 완전히 일치함을 확인했다. Residual 행별 입력, pruning 전 context, scorer 학습 라벨, cal/test 점수, 모든 expert의 test 확률을 저장한다.
- Cal의 모든 expert 예측은 기존 정책·test 평가 후 같은 fitted model로 계산한다. 선택된 행만 예측하는 기존 호출과 전체 행을 예측하는 진단 호출 간 차이를 기록한다. 합성 smoke에서 cal 후보 11/700행이 달랐으므로 완전 동일한 호출 결과라고 가정하지 않는다. 원래 cal 후보·점수를 별도로 보존해 원래 gate 분석에 사용한다.
- 전체 예측 캐시를 직접 저장하므로 EXP38처럼 global·expert를 다시 fit하는 재구성 단계가 필요 없다.

## 검증과 산출물

- 정제 불변조건 테스트 4개: 그룹 전체 삭제, 비모순 중복 유지, hash collision 분리, 전처리·signed zero와 split 경계.
- 관측 코드·교정 집계 테스트 2개 통과.
- 7개 클래스·46개 feature 합성 데이터로 GPU 전체 smoke 통과: 정제 loader → global/expert → scorer/verifier → dense 캐시 → EXP39 legacy → 진단. 합성 성적은 연구 결과로 인용하지 않는다. Smoke의 원래 최종 정책과 dense 진단 재계산 최종 라벨 차이는 0행이었다.
- [실행 기록](exp47_launch.json), [사전 고정 프로토콜](../../../tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42/protocol.json), [검증 기록](../../../tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42/validation.json).
- [결과 폴더](../../../tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42), [controller 로그](../../../tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42/controller.log), [기준 실행 로그](../../../tabpfn/results/20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42/baseline.log).

상태 파일은 `RUNNING.json`에서 `COMPLETE.json`으로 전환됐다. 기준 모델·기존 legacy 정책·진단 세 단계 모두 완료됐다.
