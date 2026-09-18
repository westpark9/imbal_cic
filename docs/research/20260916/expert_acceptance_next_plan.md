# 모순 제거 후 유용한 expert 채택까지 — 다음 실험 계획

## 현재 실행과 범위

- **완료:** CIC2018 EXP47. 정제 후 기존 C0·global·expert bank·scorer·verifier를 새로 학습했다. 실제 호출·채택 0건이다.
- **완료:** ToN EXP48, seed 42. 09-16 03:08 KST 완료. 모순 제거 후 같은 구조를 재검증한다. 기존 EXP46 global context 비교는 중단 상태를 유지한다.
- 아래 변경 실험은 **다음 계획**이다. 09-17에는 기존 코드·저장 예측을 확인하고 보고서를 보완했다. 새 모델 학습이나 변경 정책의 test 평가는 하지 않았다.

## 완료 결과에 따른 우선순위

ToN에서 실제 채택과 macro-F1 개선이 확인됐다. EXP31은 0.683734→0.738956, 별도 legacy는 0.754327이다. **후속 개발은 ToN을 우선하고, CIC2018 저장 캐시는 verifier 병목의 대조로 유지**한다. 먼저 같은 ToN bank에서 legacy의 낮은 훼손이 어느 학습·정책 요소에서 나오는지 분리한다. Mitm/ransomware의 test tail F1은 legacy 0.103410이고 후행 confirm에서도 tail F1이 소폭 감소했으므로, tail 비감소 조건을 통과한 최종 방식으로 확정하지는 않는다. Tail의 train/context/후행 구간 피복과 bank 교정 기회를 별도 진단한다.

아래 S/V 대조 설계는 두 데이터셋에 공통 적용한다. ToN의 실제 채택 정책을 더 유용하게 만드는 실험과, CIC2018의 거의 전부 거절되는 경로를 비교한다.

## 0. 중립 호출을 허용한 정책 조건 대조

둘 다 정답인 호출을 잘못된 호출로 간주하지 않는다. 이번 EXP47도 비용 가중치 `lambda_cost=0`이며 중립 승인을 허용하고, 정답 확률 개선은 NLL 목적함수에 기여한다. Calibration은 수행됐지만 별도 조건인 `H+D>=30`에서 모든 조합이 탈락했다.

저장된 77개 selection-calibration 조합에서 **H+D 최소 조건만 제외하면 7개가 통과**한다. 나머지 호출률·승인 수·훼손 비율·benign FPR 조건은 그대로다. 그중 기존 목적함수가 가장 큰 `(0.827007, 0)`은 승인 36,156, H=6, D=0, 둘 다 정답 36,146, 둘 다 오답 4다. 이는 조건 재집계이며 새 정책을 test에 적용한 결과가 아니다.

첫 비교는 같은 bank·scorer·verifier·점수·후보 격자에서 기존 최소 30건 / 최소 조건 완화를 대조한다. 30을 필수 원칙으로 고정하지 않는다. NLL 개선과 분류 교정은 따로 보고, 중립 승인이 많다는 이유로 실패로 처리하지 않는다. 중립을 H로 재분류하거나 훼손 비율의 분모에 넣어 위험을 희석하지 않는다. 호출 비용을 더 허용하는 정책은 별도의 호출률 상한 대조로 분리한다.

정답 확률이 낮아져도 최종 라벨이 맞으면 허용한다는 목표까지 포함하면, 양의 NLL 개선을 요구하는 현재 verifier와도 차이가 있다. 아래 H/D/중립 target 대조로 이를 검증한다. 최소 사례 수 조건 완화와 학습 목표 변경을 같은 효과로 해석하지 않는다.

현재 호출 후보는 scorer 최대 점수의 사전 지정 백분위 11개, 승인 후보는 normgain 단위의 고정값 7개다. 두 격자 모두 정제 데이터에서 최적이라고 검증된 것은 아니다. 조건 대조 후, 관측된 승인 점수의 분포·0 근처를 반영한 후보 생성 규칙을 validation에서 비교한다. 이를 test에 맞춰 조정하지 않는다.

## 1. Verifier target 분리 대조

정제 후 calibration에서 top-1 교정 2,816건이 원시 verifier 양수 6건으로 줄고, 보정 후에도 6건이다. 같은 bank·scorer를 고정해 승인 학습의 효과를 먼저 분리할 수 있다. 추가 TabPFN 학습은 필요하지 않다.

| Arm | Scorer | 최종 verifier target | 목적 |
|---|---|---|---|
| S0V0 | 기존 NLL·OOF 기반 양성 | 기존 normgain | 같은 확인 프로토콜의 대조군 |
| S0V1 | S0와 동일 | 교정 / 훼손 / 중립 | Verifier target만 바꾼 효과 |
| S1V0 | 교정 가능성 − 훼손 위험 | V0와 동일 | 후보 선택 목표의 효과 |
| S1V1 | S1와 동일 | V1와 동일 | 두 변경의 상호작용 |

S0V0·S0V1부터 실행하고, 결과를 확인한 뒤 S1V0·S1V1을 추가한다. S0의 OOF teacher와 만들어진 양성 라벨을 고정한다. 최종 verifier 변경이 scorer 학습 라벨 변경까지 유발하지 않도록 한다.

### 학습 라벨과 점수

- **H(교정):** global 오답, expert 정답.
- **D(훼손):** global 정답, expert 오답.
- **N(중립):** 둘 다 정답 또는 둘 다 오답. 둘 다 오답인데 예측 라벨이 달라진 경우도 별도 집계한다.
- 첫 V1 후보는 H/D/N 확률을 추정하고 `P(H) − λ·P(D)`를 승인 점수로 사용하는 방식이다. λ는 사전에 정한 작은 격자에서 validation으로 선택한다. 클래스 가중치와 중립 표본 정책은 명시·고정하고, 보정 시 실제 검증 모집단의 분포로 되돌린다.
- 최초 비교는 기존 post feature·모델 용량을 유지한다. Raw feature 또는 embedding 추가는 target 대조 후 별도 arm으로 비교한다.
- Scorer는 **expert를 호출하기 전** 이용 가능한 feature로만 후보를 고른다. 실제 expert 예측이나 정답을 pre gate 입력으로 넣지 않는다. H/D 라벨은 학습 때만 사용한다.
- Target을 바꾸면 점수 단위도 달라질 수 있으므로, 서로 다른 점수에 동일한 숫자 임계값을 기계적으로 적용하지 않는다. 두 대조군에 같은 후보 수의 사전 정의된 분위수 탐색 규칙을 적용하고, 원래 EXP47 정책 결과도 별도 기준으로 보존한다.

## 2. 호출·채택·유효 결정을 분리한 검증

1. **Route:** scorer/verifier 학습과 cross-fitting. Context·mining과의 분리 및 같은 벡터 겹침을 기존 mask 기준으로 감사한다.
2. **Cal select:** 호출/승인 임계값 선택. Test를 사용하지 않는다.
3. **시간상 뒤의 cal confirm:** 선택된 정책을 그대로 적용해 승인 수·H/D·클래스별 F1·benign FPR를 확인한다. 여기서 실패한 후보를 같은 confirm에 맞춰 반복 조정하지 않는다.
4. **정책 고정 후 test:** 최종 성능을 기록한다. 기존 development holdout이라는 한계를 명시하고 seed 43·44에서 재현성을 확인한다.

공통 표에는 호출, 승인, 예측 라벨 변경, H, D, H+D, 양쪽 정답, 양쪽 오답, 클래스별 bank 기회와 회수율을 넣는다. Query 수와 query–expert 쌍 수를 섞지 않는다.

분류 교정의 목표는 **확인 구간에서 실제 채택 > 0, Δmacro-F1 > 0**이면서 benign FPR 증가 ≤ 0.0005, tail/보호 클래스 F1 조건을 충족하는 정책이다. 중립 위주 정책도 호출·확률 개선 결과로 보고하되 분류 교정 성과와 구분한다. 최소 H+D는 0절의 대조 설정이며 30건을 계속 고정하지 않는다. 처음에는 최소 승인 200건·호출률·훼손 비율 조건을 고정해 변경 효과를 분리한다. 작은 tail에서 고유 벡터·시나리오별 지지 사례도 함께 기록한다.

## 3. Expert context 대조

학습 목표를 바꿔도 적절히 구분되는 유용한 답이 부족하면 bank를 개선한다.

- Global과 residual 군집, 각 expert의 총 context 행 수를 고정한다.
- **기존 공통 anchor + residual block**을 대조군으로 둔다.
- **동일 예산의 benign 대조 사례 보강**을 비교한다. 특히 global 또는 expert가 침투로 잘못 보는 benign과 실제 infiltration을 함께 보여준다.
- 대조 사례는 train 내 별도 mining fold에서 얻는다. Route/cal/test의 정답이나 오류를 context 선택에 사용하지 않는다. Context 후보 간 중복·고유 벡터 수와 class×scenario 피복도 기록한다.
- 먼저 infiltration recall 유지, precision 증가, benign 훼손 감소를 평가한다. 그 다음 같은 라우팅 학습 프로토콜로 채택 증가가 실제 최종 성능으로 이어지는지 본다.
- 각 block의 raw residual 중앙값·95백분위·최댓값·상한 초과 수와 정답→global 예측 오류 방향을 기록한다. 같은 정답 클래스의 test H/D와 병치하되 같은 행의 전후 변화로 해석하지 않는다. 추론에서 관측할 수 있는 호출 영역을 먼저 정의하고 그 안의 H/D로 전문화를 검증한다.
- E1·E5의 benign 4·1행은 전체 signature의 군집 배정 결과이며, 작은 군집을 전부 쓴 것이다. 의도된 hard-negative quota나 bot/brute 혼동 교정 사례가 아니다. Anchor 포함 benign은 2,004·2,001행이다. 이를 전문가 역할 설계의 근거로 사용하지 않는다.
- Residual clipping **없음 / 기존 전역 99.5% / 클래스별 규칙**을 context 비율 변경과 분리해 비교한다. 0.601645는 expert용 200만 residual의 99.5백분위 계산값이며 0.995 설정의 최적성은 미검증이다. 전체 9,815행만 잘리지만 web 103/103, infiltration 5,856/6,956행이 해당한다. 실제 tail 오분류 행은 모두 block에 들어갔으므로 clipping만을 원인으로 전제하지 않고, block 피복과 실제 라우팅 성능으로 평가한다.

호출 범위를 global top-1/top-2나 관측 feature로 제한하는 방식도 별도 대조가 가능하다. 예를 들어 benign–mitm expert에 두 클래스 관련 후보만 허용할 수 있지만, 이 제한이 다른 global 예측에서 넘어오는 교정 기회를 얼마나 놓치는지 함께 평가해야 한다. 실제 정답 클래스를 호출 조건으로 쓰지 않는다.

### K는 최종 라우팅 결과로 다시 선택

현재 K=8은 tune NLL-oracle 기준으로 고른 **임시 기준선**이다. 실제 scorer/verifier까지 포함한 최적 K라는 근거는 없다.

1. K=2·4·8 각각 bank뿐 아니라 scorer·verifier·임계값 선택까지 수행한다.
2. Global·정제·split·context 선택 규칙과 예산 정책을 고정한다. K별 총 context 행 수·호출 비용은 함께 보고한다. 고정된 expert당 예산으로 K가 늘어나 총예산도 증가하면 그 효과를 분리한다.
3. 임계값 선택용 cal과 별도의 validation에서 최종 정책 macro-F1·클래스별 H/D·회수율을 비교해 K를 선택한다. Oracle은 잠재 교정 기회 진단으로만 둔다.
4. K와 정책을 고정한 뒤 선택에 사용하지 않은 후행 confirm, 그 다음 test에서 평가한다. 같은 confirm으로 K를 고른 뒤 독립 확인이라고 보고하지 않는다.

## 4. CIC2018와 ToN 중 어디를 이어갈 것인가

**현재 우선 대상은 ToN이며, CIC2018은 고정 캐시 대조로 유지한다.** 아래 기준으로 다음 반복의 범위와 최종 기준선 채택 여부를 판단한다.

| 관찰 | 이어갈 실험 |
|---|---|
| ToN에 tail·관련 benign의 고유 사례가 남고, bank 교정과 후행 확인 support가 충분 | ToN에도 같은 S/V 대조를 적용하고 확인 구간 개선이 재현되는 쪽을 주 실험으로 선택 |
| ToN bank는 유용하지만 scorer/verifier에서 대부분 소실 | Context를 동시에 바꾸지 않고 먼저 고정 bank의 목표 대조 |
| ToN C0가 정제 후에도 부적합해 global·residual부터 불안정 | 같은 100k 예산의 기존 C0 / 무작위 / train에서 관측한 특성 구성 비교 후 새 기준선 고정 |
| ToN의 교정 기회 또는 확인 support가 부족 | CIC2018 infiltration·benign 교정을 우선, ToN은 context·피복 진단으로 유지 |

ToN 정제 후 10개 클래스는 모두 남았다. 하지만 train/val/test의 benign 유지율 등이 크게 다르므로 과거 C0 비교 결과를 그대로 새 데이터에 적용하지 않는다. 이번 EXP48이 새 기준선이다. ToN test macro-F1이나 bank oracle이 더 높다는 이유만으로 데이터셋을 고르지 않는다.

## 보고서

- 개별 HTML: `lablog/html_report/clean_revalidation_0916.html`
- 통합본: `lablog/html_report/post_tabpfn.html#clean_revalidation_0916`
- ToN 완료 후 `build_clean_revalidation_report.py`가 실제 최종 결과·calibration·클래스별 기회·expert 단독 행렬을 생성하고 `sync_html_reports.py`로 통합본에 반영한다.
- 온라인 Claude 아티팩트 편집 도구가 없어 원격 갱신은 불가하다. 로컬 동기화와 온라인 상태를 구분해 기록한다.
