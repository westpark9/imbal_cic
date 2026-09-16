# EXP48 — ToN 모순 제거 후 기존 구조 재검증

## 실행

- 사용자 요청: CIC2018 정제 후 진단을 시각화하고 ToN에도 같은 절차를 실행. 이후 요청: 다음 날 아침 확인할 수 있도록 실행을 계속하고 결과를 HTML로 작성.
- Run: `tabpfn/results/20260916_021730_nfv3_toniot_exp48_clean_revalidation_s42`.
- 학습 시작: **2026-09-16 02:19:58 KST**. 최초 ETA **03:20–04:00 KST**. 종료 여부는 run의 `COMPLETE.json`/`ERROR.json`으로 확인한다.
- 새 EXP31 baseline → 동일 새 bank의 기존 EXP39 legacy → residual/expert/scorer/verifier 진단을 순서대로 실행한다.
- 종료 후 로컬 개별 HTML, 통합 HTML, 근거 JSON을 자동 갱신한다. 성공적으로 학습이 끝나면 `exp48_results.md`도 생성한다. `REPORT_REFRESH.json`에 갱신 성공 여부를 기록한다.

## 정제 결과

| 항목 | 행 |
|---|---:|
| 원본 ToN | 27,520,260 |
| 모순 그룹에 속해 제외 | 16,712,972 |
| 남은 행 | 10,807,288 |
| Train | 4,669,119 |
| Validation | 3,866,446 |
| Test | 2,271,723 |

같은 float32 모델 입력 벡터에 여러 family 라벨이 있는 그룹의 모든 행을 제외한다. 같은 라벨 중복과 원래 outer split 소속은 유지한다. 원본 파일은 보존한다. 남은 실제 모순 그룹 0, 사라진 train 클래스 0. Tail test support: mitm 1,161, ransomware 709행.

정제 manifest: `data/derived/toniot_conflict_free_fixed_split_20260916_021730/manifest.json`.

## 비교 설정

기존 CIC2018 EXP47과 같은 EXP31 설정을 바탕으로 target dataset만 ToN으로 변경했다. C0 100k, benign share 0.75, 공격 균등, TabPFN n_estimators=4, 기존 residual/anchor/K=2,4,8 선택/scorer/normgain verifier를 유지한다. Seed 42. 전체 expert test 진단과 원래 calibration 후보를 저장한다. 경로와 loader는 새 실행에 격리한다.

별도 EXP39 legacy의 보호 클래스에서 ToN에 없는 brute_force를 제외하여 ddos,dos를 사전 지정했다. Tail은 ToN vocabulary인 mitm,ransomware다. Baseline과 secondary policy의 중간 수치를 섞지 않는다. 예전 EXP46 global context 비교를 재개한 것이 아니다.

## 검증

- 기존 모순 제거 단위 검증 4건, 관찰 hook/교정 집계 검증 2건 통과.
- ToN 10클래스 합성 데이터로 모순 제거→baseline→legacy→audit 전체 smoke 통과. Benign label ID가 1이어도 동작하며, 모순 두 행 제외·같은 라벨 중복 유지·ToN tail/protected vocabulary를 확인했다.
- 합성 smoke는 작은 모델에서 non-degenerate teacher를 얻기 위해 verifier quantile 0.75 및 50 trees를 사용했다. 실제 EXP48은 원래 quantile **0.25**, scorer **300 trees/depth 6**, verifier **400 trees/depth 6**을 유지한다. Smoke 결과는 실제 성능 근거로 사용하지 않는다.
- 최초 smoke의 작은 normgain teacher는 scorer 양성이 단일 클래스라서 중단됐으며 실제 데이터 학습의 실패가 아니다. 그 로그를 보존하고, 구조 확인용 합성 설정을 위와 같이 조정해 전체 경로를 검증했다.
- 원본 EXP47 cleaner/trace/runner와 기존 EXP31/39 소스를 변경하지 않았다. 새 ToN loader/runner/trace wrapper를 별도로 두고 실행 소스의 SHA-256을 고정했다.

## 완료 후 결과 위치

`docs/research/20260916/exp48_results.md`, `clean_revalidation_visual_data.json`, `lablog/html_report/clean_revalidation_0916.html`, `lablog/html_report/post_tabpfn.html`.

연결된 온라인 Claude 아티팩트를 편집할 도구가 없으므로 온라인 갱신은 수행하지 못한다. 로컬 결과·동기화 검증·Git push·온라인 게시 상태를 구분한다.
