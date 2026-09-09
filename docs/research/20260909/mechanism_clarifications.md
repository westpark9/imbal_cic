**RACE-PFN 추가 설명 — residual 유지, 학습/추론 순서, 선택 지표**

2026-09-09 사용자 추가 질의에 대한 보완이다. 제안 단계이며 모델 코드를 변경하거나 성능을 측정한 결과가 아니다.

1. **하위패턴 표본 확보:** 현재 모든 세부 공격의 표본이 부족하다고 단정하지 않는다. 기존 scenario 피복 보완 실험도 있다. 추가 제안은 최종 C0/Ck에 대해 class×scenario 및 관측 가능한 feature mode별 포함 수·고유 벡터 수를 점검하고, 부족한 구간에 최소 지원을 두는 것이다. 알려진 세부 공격명과 feature에서 발견한 mode는 같은 개념이 아니다. 대상은 큰 클래스뿐 아니라 모든 클래스다. 표본 수만 늘려 해결되는 문제인지 같은 예산의 대조 실험으로 확인한다.

2. **Global 고정:** 초기에는 C0 행·backbone 설정·global 출력·expert bank까지 고정해 scorer/verifier 변경의 효과를 비교한다. context 실험은 그 뒤 별도 요인으로 수행한다.

3. **Residual과 중간 클래스:** 앞서 쓴 `anchor + tail + 사례`는 범위를 지나치게 좁혔다. 정확한 제안은 `anchor + global residual로 찾은 취약 구간의 지원 사례 + 그 구간과 혼동되는 대조 사례`다. Global의 약점 정보는 support를 찾는 주된 신호로 유지된다. 모든 클래스의 class-conditional error/residual과 내부 mode를 조사하므로 중간 빈도 클래스의 특정 공격 유형에서도 expert를 만들 수 있다. residual 후보 안에서 대표성이 높은 지원 사례를 고르고, 관련된 반대 라벨 사례를 추가한다. residual이 큰 순으로만 뽑지 않는다는 뜻이다. 같은 global 위의 실제 held-out 교정으로 bank의 유용성을 판단한다. 초기에는 class별로 정규화해 비교하여 행 수가 많은 클래스가 후보 예산을 독점하지 않게 하고, tail만을 expert의 소유 대상으로 정하지 않는다.

4. **현재 학습 중 의존성:** 최종 추론 순서는 scorer→선택 expert→verifier다. 그러나 학습은 다음 순서다.

```text
D_route에서 global 및 모든 expert 예측
→ NLL 이득 G 계산
→ 임시 verifier를 fold별로 학습
→ 다른 fold의 수락 여부 b_oof 생성
→ scorer 정답 = 1{b_oof × G − 계산비용 > 0}
→ scorer 학습
→ 전체 route pair로 최종 verifier 학습
```

코드 근거: [전체 expert의 route 예측](../../../tabpfn/scripts/nfv3_v3_exp31_c0alloc.py:1355), [OOF verifier와 scorer 정답 생성](../../../tabpfn/scripts/nfv3_v3_exp31_c0alloc.py:1440). “이전 verifier”는 여기서는 이전 실험의 모델이 아니라 **같은 훈련 과정에서 먼저 만드는 임시 OOF verifier**다. 학습 완료 후 온라인으로 scorer의 파라미터를 다시 바꾸는 과정은 이 코드에 없다.

계산비용이 0일 때 어떤 pair의 G가 양수여도 b_oof=0이면 scorer 라벨은 0이다. 이 경로를 분리하려는 것이다. 최초 비교에서는 임시 verifier의 b로 가리지 않은 결정 손익을 두 컴포넌트의 감독 신호로 사용한다. 검증된 최종 정책을 OOF로 증류하는 것은 이후의 선택적 확장이고 최초 변경에 포함하지 않는다.

5. **호출 전 예상과 호출 후 관측:** 학습 데이터에서는 expert를 실행해 결과를 알아야 감독 신호를 만들 수 있다. 이 과정은 현재도 존재한다. 실제 추론에서 scorer는 query의 입력 표현·global 출력·expert 설명자만으로 예상 이득을 계산한다. 후보별 점수 계산은 후보 expert의 TabPFN 실행과 다르다. 선택한 expert만 실행하고 verifier가 그때 얻은 출력을 본다.

```text
추론: query → global
             → scorer가 호출 전 정보로 expert별 점수 예측
             → 선택된 expert만 실행
             → verifier가 global/expert의 실제 출력을 비교
             → expert 채택 또는 global 유지
```

[온라인 호출 코드](../../../tabpfn/scripts/nfv3_v3_exp31_c0alloc.py:1630). Scorer는 실제 정답을 보지 않으므로 예상이 틀릴 수 있다. 따라서 같은 bank에서 “교정 가능한 pair를 얼마나 선택했는가”를 별도 평가한다. scorer 입력에 pk를 넣는 변경은 제안하지 않는다.

6. **학습 목표는 실제 결정 교정:** 단순 0–1 loss 기준으로 pair의 정답은 아래와 같다.

| Global | Expert | Δk |
|---|---|---:|
| 오답 | 정답 | +1 |
| 정답 | 오답 | −1 |
| 정답 | 정답 | 0 |
| 오답 | 오답 | 0 |

이 값 또는 class-balanced 가중값을 감독 신호로 사용한다. reward와 같은 해석은 가능하지만 새로운 강화학습을 도입하는 제안은 아니다. 모든 pair 결과가 있으므로 supervised regression 또는 correctness-state 분류로 학습한다. verifier는 post-call 정보, scorer는 pre-call 정보로 기대 이득을 예측한다. NLL은 보조 진단으로 남길 수 있다. 두 예측이 모두 오답이면 위 Δ는 0이지만 어느 클래스의 FP를 만드는지는 달라질 수 있어, 이 감독 신호만으로 macro-F1을 정확히 최적화한다고 주장하지 않는다.

7. **구체적인 임계값 선택 지표:** 앞선 “실제 교정 손익”을 다음처럼 학습 감독 신호와 최종 정책 선택으로 구분한다. 이 표는 제안을 구체화한 것으로 현재 코드에 이미 구현되었다는 뜻이 아니다.

| 용도 | 지표 | 적용 |
|---|---|---|
| Scorer/verifier 학습 | pair별 Δk 또는 class-balanced 가중 Δk | 전체 클래스의 교정/훼손 사례를 학습. tail에만 +1을 주지 않음. |
| 임계값 선택의 주지표 | `ΔMacroF1_all = MacroF1(system) − MacroF1(global)` | pre/post 임계값 grid의 실제 최종 예측에서 직접 계산해 최대화. 클래스별 F1의 단순 평균으로 큰 클래스의 행 수 지배를 줄임. |
| Tail 유지·개선 확인 | `ΔMacroF1_tail` | 선택 시 tail의 악화를 제한하고, tail 개선이라는 연구 목표의 성공은 양의 개선을 별도 요구. 전체 평균 향상만으로 tail 개선을 주장하지 않음. |
| 중간 클래스 보호 | 각 중간 클래스의 `ΔF1_c` | 평균으로 숨겨지는 하락을 막기 위해 클래스별 허용 감소 εc를 사전 고정. 평균 하나로 합치지 않음. |
| Benign 오경보 제한 | `ΔFPR_B = FPR_B(system) − FPR_B(global)` | 기존 실험의 허용 상한 εB를 초기 비교에서 고정하고, absolute FPR도 보고. |
| 기전 진단 | `NetCorrection_c = (H_c − D_c)/N_c` | Hc는 global 오답→system 정답, Dc는 global 정답→system 오답, Nc는 실제 class c의 전체 행 수. 정확히 class c의 recall 변화량. |

즉 임계값 쌍 θ에 대한 기본 선택은 다음과 같다.

\[
\max_\theta\ \Delta\operatorname{MacroF1}_{all}(\theta)
\quad\text{s.t.}\quad
\Delta\operatorname{FPR}_{B}\le\epsilon_B,\quad
\Delta\operatorname{MacroF1}_{tail}\ge0,\quad
\Delta\operatorname{F1}_{c}\ge-\epsilon_c\ (c\in\mathcal M).
\]

M은 train 통계로 미리 정의한 중간 빈도 클래스 집합이며 test 결과로 고르지 않는다. 별도로 합의된 손실 허용치가 없으면 최초 진단의 εc=0을 비교 기준으로 제시할 수 있다. 이는 성능 보장이나 운영상 정답인 허용치가 아니다. 표본이 적으면 불확실성이 크므로 실제 선택 안정성과 다음 시간 구간 결과도 보고한다. feasible한 양의 개선이 없으면 호출 0을 모델 개선 성공으로 처리하지 않는다.

Benign FPR의 분모는 **전체 benign 행 수**다. `FPR_B = #{y=benign, prediction≠benign}/#{y=benign}`. 예를 들어 benign 10,000개에서 기존 오경보 20건을 고치고 새 오경보 5건을 만들었다면 ΔFPRB는 `(5−20)/10,000 = −0.0015`, 즉 −0.15%p다. 기존 threshold 함수도 신규 오경보와 교정된 오경보를 함께 계산한다. [현재 선택 함수](../../../tabpfn/scripts/nfv3_v3_exp31_c0alloc.py:556)

`ΣΔk` 또는 클래스 정규화한 순교정률은 정확도/recall 계열이다. 이것이 F1과 같지는 않으므로 threshold grid에서는 F1을 직접 계산한다. 미분이 필요 없는 유한 후보 선택이므로 supervised 학습의 대리 목표와 정책 선택 지표를 구분할 수 있다.

표본추출로 calibration 비중이 달라졌다면 confusion matrix를 추출확률로 가중하거나 class-conditional confusion을 과거 목표 비중으로 재구성한 뒤 F1을 계산한다. **클래스별 F1을 나중에 prior로 가중하는 것만으로는 precision의 비중 의존성이 보정되지 않는다.** class 안의 편향·시간 이동까지 해결되는 것은 아니므로 독립된 다음 시간 구간에서 검증한다. threshold 선택에 test 정답은 쓰지 않는다.

이 지표 설계의 목적은 모든 클래스의 global 약점을 교정하면서 tail 개선과 benign 유지가 실제로 함께 나타나는지 판별하는 것이다. 비용 감소만으로 성능 목표를 대체하지 않는다.
