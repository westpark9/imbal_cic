# RACE-PFN 진행 상태 (2026-08-31 기준)

TAILGUARD(에너지 게이트 + family expert) 트랙을 접고, frozen TabPFN 위에서
글로벌 모델의 실패로부터 expert context를 자동 발견·조건부 호출하는
**RACE-PFN**으로 전환한 이후의 기록 요약. 상세 근거는 랩 로그
(`manuscript/report/0825~0828.md`)와 보고서(`docs/racepfn_main_figure.html`,
아티팩트 배포본)에 있다.

## 한 줄 상태

- expert **구성층은 검증 통과**: regime 서로소·실패 유형별 분리, dos/inf/web
  순수 전문가 생존 가능, bank oracle 상한 0.83–0.87로 XGB(0.7548) 상회.
- **안전층 통과**: 최고 구성에서 유해 교정 4행/4.02M (정밀도 0.9997),
  benign FPR 불변, ECE 0.008, 평균 호출 1.16회.
- **병목은 회수층(제안 관문)으로 특정**: 이득-실재 행의 74~95%가
  scorer·τ_pre에서 걸러짐 (recall 0.09→0.26까지 개선, utility 실현 ~0.20).
- **IID macro-F1은 XGB 미돌파**: 계열 최고 0.7428(exp22c), guide-정합
  계열 0.65~0.70.

## 실험 계보 (모든 판정은 같은 run·같은 split; XGB 0.7548 인용)

| run | knob | system | vs global | 비고 |
|---|---|---|---|---|
| exp22c `0826_130310` | 실패-서명 + hard 배정 + diversity 블록 | 0.7428 | +0.063 | 계열 최고. 관문·pruning 없던 개방 구성 (harmful 8,524 · 1.60회) |
| exp23 `0826_183417` | guide 정합 패키지(6-way split·K선택·scorer·quantile verifier·보정) | 0.7001 | +0.032 | single-pass pruning이 8→1 expert로 붕괴 |
| exp23b `0827_011706` | greedy backward pruning | 0.6937 | +0.025 | 생존 2, 쌍둥이 중 "쉬운 쪽" 생존 근시안 잔존 |
| exp23c `0827_023651` | anchor 2000→200 | 0.6745 | +0.003 | oracle 최고 0.8651이나 회수 붕괴·web 재손상 |
| exp23d ×3 `0827_035659/051400/063039` | anchor 구성 natural/inverse/natural-2000 | 0.651~0.669 | −0.005~−0.017 | 순수 전문가 생존(가설 적중)했으나 비전문 posterior 붕괴로 전패 |
| exp24a `0827_144730` | regime-조건부 marginal pruning | 0.6721 | +0.026 | marginal 1.0~10.2로 재분화, 생존 5 |
| exp24b `0827_161322` | scorer를 sign 분류 P(U>0)로 | 0.6824 | +0.013 | recall 0.18, 채택 10×, web 손상 |
| **exp24b-off** `0827_174858` | + pruning off 대조군 | **0.6978** | **+0.028** | **현 추천 구성**: harmful 4 · recall 0.261 · 1.158회 |
| 축-3 분석 `8a_prior_shift` | test-prior shift (§20.4, 사후 재가중) | — | — | 가설 기각: known-prior 보정 시 XGB 압승. RACE의 +0.03은 4개 mix 전부에서 유지(prior-강건) |

## 핵심 발견 (시간순)

1. **anchor의 양면성**: 공유 균형 anchor(2k/класс)는 leave-one-out marginal을
   눌러 pruning을 붕괴시키지만(exp23), 제거하면 expert의 비전문 posterior가
   무너져 회수층 전체가 실패(exp23c/d). natural anchor에서는 residual bank의
   random 대조군 대비 oracle 우위(+0.075)도 +0.009로 소멸 — Gate-1 우위의
   상당분이 [균형 anchor × 블록] 결합 효과.
2. **pruning 측정은 regime-조건부여야 함**: expert의 marginal을 자기 regime
   행에서 재면 anchor 기여가 상쇄되어 전문가가 생존(exp24a).
3. **scorer는 절대-이득 회귀가 아니라 부호 문제**: 6자릿수 w_bal 스케일
   회귀가 판별력을 죽였음(상관 0.03). sign 분류로 recall 0.09→0.26,
   τ_pre가 확률 단위로 안정화(exp24b).
4. **temporal drift의 명명된 증거**: hard-brute regime = **ssh_bruteforce
   (2/14 18:01 개시)**, easy = ftp_bruteforce(오후). frozen global은 test의
   SSH 37,696행을 전량 오답(brute F1 0.725 동결의 원인)이고, brute의 tune
   창(15:31–15:43)은 SSH 이전에 끝나 SSH 전문가가 측정 불능 → 탈락.
   temporal refresh(축 1)의 자체 동기 증거.
5. **축-3(prior shift)은 차별화 축이 아님**: XGB posterior가 재가중을 견딜
   만큼 날카롭고 TabPFN posterior는 sharpness 부족(보정은 raw posterior에
   적용해야 하며 tempered에 곱하면 붕괴). 차별화 서사는 축 1로 집중.

## 다음 단계 (우선순위)

1. **축 1 — temporal refresh 실험** (guide §17, exp25 예정): window별 라벨
   예산 B∈{0,50,200,1000}로 context 갱신 → 다음-window 회복 측정. FTP→SSH
   전환(18:01)이 자연 실험 경계. 코딩 1일 + GPU 하룻밤.
2. 선행 knob: tune-부재 regime의 marginal을 D_cal에서 fallback 측정
   (SSH 전문가 구제 — D_cal에 SSH 27,301행 존재 확인).
3. γ<1 tempering (w_web≈1500이 관문·oracle을 web 방향으로 편향).

## 파일 맵

- 실험 스크립트: `tabpfn/nfv3_v3_exp22c…exp24b_*.py` (one script = one frozen
  experiment), 분석: `scripts/exp24_prior_shift_eval.py`
- 설계 문서: `tabpfn/RACE-PFN_RESEARCH_IMPLEMENTATION_GUIDE.md`
- 랩 로그: `manuscript/report/0825.md ~ 0828.md` (사전 등록·결과·보강)
- 보고서: `docs/racepfn_main_figure.html` / 발표: `docs/racepfn_slides.html`,
  `docs/racepfn_briefing.pptx` (12장, Arial)
- run 기록: `tabpfn/results/<ts>_*` — args.json이 config 원본, npz(예측·row
  ID·gain 행렬)는 로컬 보관(git 제외), 7a/8a는 사후 분석 산출물
