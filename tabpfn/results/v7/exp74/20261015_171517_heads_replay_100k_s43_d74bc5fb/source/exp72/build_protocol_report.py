#!/usr/bin/env python3
"""Publish protocol only; never reads metrics or predictions from an EXP72 run."""
from replay_protocol import ROOT
import base64
import json
import re
import sys
sys.path.insert(0,str(ROOT/'scripts/common'))
from report_order import apply_rules  # 보고서 표 규칙(숫자 우측 정렬·클래스 표 순서) — publish checklist가 요구

REPORT_ID='sequential_replay_1014'
TITLE='새 공격 반영: 갱신 지연을 포함한 순차 재생 평가'

HTML='''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>새 공격 반영: 갱신 지연을 포함한 순차 재생 평가</title>
<style>
:root{--ink:#172634;--muted:#576776;--paper:#fff;--bg:#f5f6f4;--line:#d9e0e2;--blue:#185c90;--orange:#bf6217;--soft:#eef4f7}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.8 "Noto Sans CJK KR","Noto Sans KR",system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:38px 30px 70px}h1{font-size:31px;line-height:1.45;letter-spacing:-.03em;margin:12px 0 18px}h2{font-size:23px;line-height:1.5;margin:38px 0 14px}h3{font-size:17px;margin:6px 0 12px}p{margin:10px 0}p,li,h1,h2,h3{word-break:keep-all;overflow-wrap:anywhere}a{color:var(--blue)}.kicker{font-size:12px;color:var(--blue);letter-spacing:.06em}.lead{font-size:18px;max-width:1050px}.pill{display:inline-block;border:1px solid var(--line);border-radius:18px;padding:2px 12px;margin:3px 5px 3px 0;background:white;font-size:12px}.note,small{font-size:12px;color:var(--muted)}nav{display:flex;gap:18px;flex-wrap:wrap;padding:20px 0;border-bottom:1px solid var(--line)}section{scroll-margin-top:20px}.callout{background:var(--soft);border-left:4px solid var(--blue);padding:15px 20px;margin:18px 0}.callout.orange{border-left-color:var(--orange);background:#fff5e9}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:18px 21px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:20px 0}.step{position:relative;background:white;border:1px solid var(--line);border-top:4px solid var(--blue);padding:14px 17px;border-radius:6px}.step:nth-child(3){border-top-color:var(--orange)}.step .num{font-size:12px;font-weight:700;color:var(--blue)}.step strong{display:block;font-size:17px;margin:6px 0}.step p{font-size:13px}.step:not(:last-child)::after{content:'→';position:absolute;right:-14px;top:45%;z-index:1;color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);background:#fff;border-radius:8px;margin:14px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 13px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--soft);white-space:nowrap}tr:last-child td{border-bottom:0}.numcell{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.table-wrap.stages td{min-width:150px}.table-wrap.stages td:first-child{min-width:65px}details{margin:18px 0;border:1px solid var(--line);border-radius:6px;padding:12px 17px;background:white}summary{color:var(--blue);cursor:pointer;font-weight:600}ul,ol{padding-left:22px}li{margin:8px 0}.timeline{display:grid;grid-template-columns:140px 1fr;gap:8px 12px;align-items:center;padding:18px;background:white;border:1px solid var(--line);border-radius:8px}.timeline .bar{display:flex;min-height:42px}.timeline .segment{padding:7px 12px;background:#e6edf6;border:1px solid #b6cae1}.timeline .old{background:#fff0da;border-color:#e7c796}.timeline .new{background:#e6edf6;border-color:#b6cae1}.timeline .dashed{border-style:dashed;background:#fafafa;color:var(--muted)}.refs li{margin-bottom:18px}.refname{font-weight:600}footer{border-top:1px solid var(--line);margin-top:32px;padding-top:18px;font-size:12px;color:var(--muted)}code{font-size:12px;overflow-wrap:anywhere}@media(max-width:720px){main{padding:25px 16px}h1{font-size:25px}.lead{font-size:16px}.grid,.flow{grid-template-columns:1fr}.step:not(:last-child)::after{content:'↓';top:auto;bottom:-20px;left:50%;right:auto}.timeline{grid-template-columns:1fr}.timeline .bar{font-size:12px}th,td{padding:8px}nav{gap:12px}}
</style></head><body><main>
<header><div class="kicker">2026-10-14 · EXP72 · 연구 버전 v7 유지 · 평가 설계</div>
<h1>새 공격의 사례를 반영하는 동안<br>탐지기는 무엇을 놓치는가?</h1>
<p class="lead">새 공격의 확인된 라벨 사례가 확보됐을 때, 누적 사례로 갱신한 모델의 <strong>희소 공격 분류 능력</strong>과 그 능력을 제공하기까지의 <strong>갱신·추론 지연</strong>을 함께 평가한다.</p>
<div><span class="pill">이번 실행: XGBoost · TabPFN-v3</span><span class="pill">CIC2018 · ToN</span><span class="pill">Seed 43</span><span class="pill">자연비율 · 균등</span><span class="pill">설명만 수록 · 이번 실험 결과 제외</span></div>
<p class="note">BoostPFN·LoCalPFN·DistPFN은 사용자 요청에 따라 이번 실행에서 제외했다. Expert·S/V도 이번 두 모델 비교에는 포함하지 않는다.</p>
<nav aria-label="문서 목차"><a href="#question">1. 실험의 질문</a><a href="#scenario">2. 데이터와 공개 순서</a><a href="#clock">3. 시뮬레이션 시계</a><a href="#evaluation">4. 평가·비용</a><a href="#models">5. 비교 모델</a><a href="#references">6. 근거 논문</a></nav></header>

<section id="question"><h2>1. 갱신 후 성능과 실제 반영 시점을 구분한다</h2>
<div class="grid"><article class="card"><h3>기존 갱신비용 사전 실험</h3><p>같은 누적 사례로 두 모델을 갱신한 뒤, 준비가 완료된 모델의 성능과 갱신·추론 비용을 비교했다.</p><p class="note">새 모델을 기다리는 동안의 트래픽은 평가하지 않았다.</p></article>
<article class="card"><h3>이번 순차 재생 평가</h3><p>갱신 중에도 입력이 계속 도착한다. <strong>새 버전이 준비될 때까지 이전 버전으로 탐지</strong>하고, 준비된 시점부터 새 버전을 사용한다.</p><p class="note">각 시점에 쓸 수 있었던 모델과 그 배치의 실측 추론시간으로 결과와 응답 지연을 계산한다. <a href="#ref-insomnia">[1]</a></p></article></div>
<div class="callout"><strong>확인할 것:</strong> 새로운 공격 사례를 반영한 뒤 희소 공격을 잘 구분하는가? 그 분류 능력을 사용할 수 있기 전까지 어떤 공격을 놓치는가? 갱신과 추론을 함께 고려했을 때 유리한 조건은 무엇인가?</div>
<p>이 실험은 <strong>실측 시간에 기반한 시뮬레이션</strong>이다. 원본 클래스의 최초 출현 순서는 유지하지만 실제 날짜별 트래픽·네트워크 환경 변화·사람의 라벨링 과정을 재현하지 않는다. TabPFN의 가중치 재학습이 없다는 사실만으로 갱신비용 우위를 가정하지 않는다.</p></section>

<section id="scenario"><h2>2. 같은 사례를 같은 시점에 공개한다</h2>
<p>두 모델은 같은 학습 사례와 같은 test 행을 받는다. 신규 클래스의 사례는 아래 단계에서 처음 공개되며, 이후에는 기존 사례와 함께 <strong>전부 누적</strong>한다. 정상 사례도 매 단계 공급한다. 라벨 공개·데이터 누적·갱신·후속 평가라는 반복 구조를 참고했다. <a href="#ref-continuous">[2]</a></p>
<div class="table-wrap stages"><table><thead><tr><th>단계</th><th>CIC2018 · 처음 도입되는 클래스</th><th>ToN · 처음 도입되는 클래스</th></tr></thead><tbody>
<tr><td>초기</td><td>Benign · Brute force · DoS</td><td>Benign · Scanning · DoS</td></tr>
<tr><td>1</td><td>DDoS</td><td>Injection · DDoS</td></tr>
<tr><td>2</td><td>Web attack</td><td>Password</td></tr>
<tr><td>3</td><td>Infiltration</td><td>XSS</td></tr>
<tr><td>4</td><td>Bot</td><td>Ransomware · Backdoor</td></tr>
<tr><td>5</td><td>—</td><td>MitM</td></tr>
</tbody></table></div><p class="note">이 표는 클래스 도입 순서를 나타낸다. 기존 공격과 Benign은 후속 단계 평가에도 계속 포함한다. 새 클래스가 반드시 tail인 것은 아니다.</p>
<div class="table-wrap"><table><thead><tr><th>데이터 구성</th><th class="numcell">CIC2018</th><th class="numcell">ToN</th></tr></thead><tbody>
<tr><td>최종 누적 학습 사례</td><td class="numcell">100,000행</td><td class="numcell">100,000행</td></tr>
<tr><td>그중 Benign / 공격</td><td class="numcell">75,000 / 25,000행</td><td class="numcell">75,000 / 25,000행</td></tr>
<tr><td>단계당 추가 Benign</td><td class="numcell">15,000행</td><td class="numcell">12,500행</td></tr>
<tr><td>재생할 정제 test 전체</td><td class="numcell">3,042,473행</td><td class="numcell">2,271,723행</td></tr>
</tbody></table></div>
<ul><li><strong>자연비율:</strong> 공격 25,000행을 정제 train의 공격 내 비율로 배정한다. 전체 트래픽의 자연분포라는 의미는 아니다.</li><li><strong>균등:</strong> 같은 공격 총량을 클래스별로 균등 배정하되 가용량 상한을 적용한다.</li><li><strong>Test:</strong> 각 클래스의 기존 test 행을 해당 클래스 도입 단계부터 마지막 단계까지 나누고 단계 안에서 섞는다. 재생 정책별로 각 행을 한 번만 평가하며, 전체 클래스별 표본 수는 보존한다.</li><li><strong>정보 공개:</strong> 미래 단계의 훈련·검증 라벨은 사용하지 않는다. 이번 XGB·TabPFN은 고정 설정으로 실행하며 추가 검증 라벨을 소비하지 않는다.</li></ul>
<p class="note">TESSERACT의 미래 정보 제한과 구간별 평가 원칙을 참고한다. 그러나 기존 정제 split을 유지한 클래스 순서 재생이므로, 원 논문의 실제 시간순 분할 제약 C1–C3를 모두 충족했다고 주장하지 않는다. <a href="#ref-tesseract">[3]</a></p></section>

<section id="clock"><h2>3. 갱신이 끝나기 전에 다음 입력이 도착한다</h2>
<div class="flow"><article class="step"><span class="num">01</span><strong>확인된 사례 공개</strong><p>신규 공격과 추가 정상 사례를 두 모델에 동시에 제공한다.</p></article><article class="step"><span class="num">02</span><strong>새 버전 준비</strong><p>XGB는 누적 재학습, TabPFN은 누적 context와 캐시를 준비한다.</p></article><article class="step"><span class="num">03</span><strong>이전 버전으로 탐지</strong><p>갱신 중에도 도착한 배치를 처리한다. 새로운 버전은 아직 사용할 수 없다.</p></article><article class="step"><span class="num">04</span><strong>준비된 버전으로 전환</strong><p>이후 도착한 배치부터 최신 완료 버전으로 분류한다.</p></article></div>
<div class="timeline" aria-label="갱신 준비와 서비스 버전의 관계를 보여주는 개념도"><strong>갱신 작업</strong><div class="bar"><span class="segment dashed" style="flex:2">새 라벨 공개 → 새 버전 준비 중</span><span class="segment new" style="flex:1">준비 완료</span></div><strong>도착 배치의 배정</strong><div class="bar"><span class="segment old" style="flex:2">이전 버전으로 분류</span><span class="segment new" style="flex:1">새 버전으로 분류</span></div></div><p class="note">개념도이며 길이는 측정 결과를 나타내지 않는다. 갱신 지연이 없는 비교에서는 새 버전이 단계 시작에 즉시 준비됐다고 가정한다. <a href="#ref-insomnia">[1]</a></p>
<div class="grid"><article class="card"><h3>입력과 시계</h3><p>입력은 <strong>1,024행 배치</strong>로 수집한다. 유입률은 <strong>100행/초 · 1,000행/초</strong> 두 값으로 둔다. 단계 끝의 잔여 행도 처리한다.</p><p class="note">배치 크기와 유입률은 민감도 분석을 위한 설계값이다. 논문에서 그대로 가져온 값이나 실제 IDS 트래픽률이 아니다.</p></article><article class="card"><h3>대기열과 버전</h3><p>갱신이 밀리면 다음 갱신은 FIFO에서 기다린다. 입력 배치는 <strong>수집 완료 시점의 최신 완료 버전</strong>에 배정한다. 추론 큐에서 기다리는 동안에는 배정 버전을 바꾸지 않는다.</p><p class="note">추론은 별도 FIFO로 처리한다. 마지막 입력 이후 남은 처리시간도 보고한다.</p></article></div>
<div class="callout orange"><strong>자원 가정:</strong> 실제 GPU에서는 측정 간섭을 피하려고 한 worker씩 실행한다. 이 시간을 이용한 재생에서는 갱신 worker와 서비스 worker가 독립된 자원을 가진다고 가정한다. 따라서 갱신과 추론의 GPU 경합·단일 GPU에서 여러 버전 동시 적재의 영향은 포함되지 않는다.</div></section>

<section id="evaluation"><h2>4. 분류 능력과 응답 지연을 함께 읽는다</h2>
<div class="table-wrap"><table><thead><tr><th>관점</th><th>비교할 내용</th><th>해석</th></tr></thead><tbody>
<tr><td>갱신 지연 없는 비교</td><td>전체·단계별 클래스 P/R/F1, 정상 오탐률</td><td>각 단계에 맞는 모델이 즉시 준비됐을 때의 분류 능력</td></tr>
<tr><td>갱신 지연 포함</td><td>실제 배정 버전의 P/R/F1, 구버전 처리량, 공격→Benign 누락</td><td>새 사례가 반영되기까지의 탐지 영향</td></tr>
<tr><td>공격 종류 식별</td><td>공격으로 탐지했지만 다른 공격 유형으로 분류한 행</td><td>공격 자체를 놓치는 것과 구분</td></tr>
<tr><td>응답과 처리량</td><td>수집·추론 큐·연산을 포함한 행별 평균/p50/p95 응답시간, 입력 종료 후 잔여 처리시간</td><td>빠른 갱신이 느린 추론을 상쇄하는지, 처리 지연이 쌓이는지</td></tr>
</tbody></table></div>
<p>Precision의 FP는 정상과 다른 공격을 포함한 평가 구간 전체에서 계산한다. Tail은 기존 test의 희소성에 연결해 클래스별로 확인한다. 구버전은 새 클래스 이름을 출력할 수 없으므로, 공격→Benign 누락과 공격 종류 오분류를 따로 보고한다. 탐지·유형 식별·학습 및 추론 비용을 함께 보는 근거는 Pragmatic Assessment의 평가 기준이다. <a href="#ref-sok">[4]</a></p>
<details><summary>갱신비용 사전 실험의 측정 보완</summary><ul><li>누적 전체·10만 행·자연/균등 조건을 유지한다. 제한 보존은 도입하지 않는다.</li><li>모델 생성·학습/캐시 준비 및 train-only 첫 1,024행 응답을 단계당 3회 측정한다. 재생에는 첫 번째 갱신 측정값을 사용하고 중앙값과 범위는 비용 변동 확인용으로 별도 보존한다.</li><li>초기 구축과 이후 갱신을 구분한다. TabPFN은 매 단계 누적 전체의 캐시를 다시 만든다는 점을 기록한다.</li><li>실제 test 추론은 배치별로 측정한다. Context 전처리, 캐시, 첫 응답으로 계산이 어떻게 배분되는지 구분하여 해석한다.</li><li>실험용 checkpoint 저장·복구, 결과 기록, 프로세스 시작과 사전 데이터 준비는 서비스 연산시간과 구분한다.</li></ul></details>
<div class="callout"><strong>이 페이지에는 EXP72의 성능·비용 결과를 싣지 않는다.</strong> 실행과 검증이 끝나면 별도의 요청에 따라 결과를 반영한다. 기존 사전 실험 결과는 10/13 탭에 보존한다.</div></section>

<section id="models"><h2>5. 이번 비교와 후속 SOTA 적용</h2>
<div class="table-wrap"><table><thead><tr><th>모델</th><th>단계별 반영 방식</th><th>이번 실행</th></tr></thead><tbody>
<tr><td>XGBoost</td><td>누적 전체로 CUDA hist 재학습 · 300 trees · depth 8</td><td>포함</td></tr>
<tr><td>TabPFN-v3</td><td>누적 전체 context · ensemble 4 · frozen weights · CPU KV cache</td><td>포함</td></tr>
<tr><td>BoostPFN <a href="#ref-boost">[5]</a></td><td>약한 학습기 context와 결합 가중치 재구성</td><td>후속 비교 · 이번 큐에서 제외</td></tr>
<tr><td>LoCalPFN <a href="#ref-local">[6]</a></td><td>검색 준비·미세조정·검증 기반 모델 선택까지 갱신 비용에 포함</td><td>후속 비교 · 이번 큐에서 제외</td></tr>
<tr><td>DistPFN <a href="#ref-dist">[7]</a></td><td>기반 TabPFN 갱신 + 현재 도착 배치의 posterior 분포로 보정</td><td>후속 비교 · 이번 큐에서 제외</td></tr>
</tbody></table></div>
<p class="note">세 SOTA의 단계별 실행은 원 방법을 반복 적용하는 방식이며, 원 논문이 신규 클래스 증분 갱신 알고리즘을 제안했다고 해석하지 않는다. DistPFN에 전체 미래 test 분포를 제공하지 않는다. 기존 BoostPFN·LoCalPFN 포트는 v1 기반이므로 추후 비교 시 backbone 차이를 명시한다.</p></section>

<section id="references"><h2>6. 근거 논문과 이번 설계에 가져온 부분</h2><ol class="refs">
<li id="ref-insomnia"><a class="refname" href="https://fabio.pierazzi.com/assets/pdf/insomnia.pdf" target="_blank" rel="noopener">INSOMNIA: Towards Concept-Drift Robustness in Network Intrusion Detection</a> · AISec 2021 · §4.2<p>새 버전이 준비될 때까지 이전 버전을 사용하는 지연 반영 평가를 참고한다. 본 실험은 논문의 binary DNN·pseudo-labeling을 채택하지 않고 확인된 다중 클래스 사례 공급을 가정한다.</p></li>
<li id="ref-continuous"><a class="refname" href="https://www.usenix.org/system/files/usenixsecurity23-chen-yizheng.pdf" target="_blank" rel="noopener">Continuous Learning for Android Malware Detection</a> · USENIX Security 2023 · §4.1–4.3<p>라벨 확보·학습 데이터 누적·모델 갱신·후속 구간 평가를 참고한다. 본 실험은 active learning이나 사람의 라벨링 지연을 측정하지 않는다.</p></li>
<li id="ref-tesseract"><a class="refname" href="https://www.usenix.org/system/files/sec19-pendlebury.pdf" target="_blank" rel="noopener">TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time</a> · USENIX Security 2019 · §4.1–4.2<p>미래 정보 접근을 제한하고 구간별 성능을 보는 근거다. 본 실험의 클래스 공개 순서 제어는 실제 timestamp 기반 분할을 대신하지 않는다.</p></li>
<li id="ref-sok"><a class="refname" href="https://www.giovanniapruzzese.com/files/papers/eurosp23/eurosp23.pdf" target="_blank" rel="noopener">SoK: Pragmatic Assessment of Machine Learning for Network Intrusion Detection</a> · EuroS&amp;P 2023 · Appendix E.3<p>정상 오탐·공격 탐지·공격 종류 식별과 학습·추론 비용을 함께 평가한다. 이번에는 그 구분을 지연 포함 재생 결과에 적용한다.</p></li>
<li id="ref-boost"><a class="refname" href="https://proceedings.mlr.press/v258/wang25d.html" target="_blank" rel="noopener">Prior-Fitted Networks Scale to Larger Datasets When Treated as Weak Learners</a> · AISTATS 2025<p>BoostPFN의 약한 학습기 결합 구조를 유지해 단계별 모델을 재구성할 수 있다.</p></li>
<li id="ref-local"><a class="refname" href="https://arxiv.org/abs/2406.05207" target="_blank" rel="noopener">Retrieval &amp; Fine-Tuning for In-Context Tabular Models</a> · NeurIPS 2024<p>LoCalPFN의 검색과 미세조정을 모두 반영해야 하므로 context 교체만으로 갱신 비용을 계산하지 않는다.</p></li>
<li id="ref-dist"><a class="refname" href="https://arxiv.org/abs/2605.04363" target="_blank" rel="noopener">Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment</a> · ICML 2026<p>DistPFN의 posterior 보정은 현재 배치에만 적용한다. <a href="https://github.com/seunghan96/DistPFN" target="_blank" rel="noopener">공식 구현</a>의 test posterior 평균 계산 범위를 미래까지 넓히지 않는다.</p></li>
</ol></section>
<footer>설계 기록: docs/research/20261014/exp72_protocol.md · 코드: scripts/v7/exp72/ · 승인된 연구 버전 v7 유지 · 결과 자동 게시 비활성화</footer>
</main></body></html>'''


def publish(html, title=TITLE, verdict='설계 · XGB / TabPFN · 결과 미수록',
            title_en='Sequential replay · Protocol and references', card=None, flow=None):
    html=apply_rules(html)
    dest=ROOT/'lablog/html_report'/f'{REPORT_ID}.html'
    dest.write_text(html,encoding='utf-8')
    index=ROOT/'lablog/html_report/post_tabpfn.html';text=index.read_text()
    m=re.search(r'const REPORTS = (.*?);\n',text);reports=json.loads(m.group(1))
    entry=dict(id=REPORT_ID,date='2026-10-14',titleKr=title,
               titleEn=title_en,verdict=verdict,
               verdictClass='neutral',group='report',type='iframe',b64=base64.b64encode(html.encode()).decode())
    reports=[r for r in reports if r['id']!=REPORT_ID]
    reports.insert(1,entry)
    text=text[:m.start(1)]+json.dumps(reports,ensure_ascii=False)+text[m.end(1):]
    card=card or '<div class="status-cell"><p class="label">10/14 · 순차 재생 평가 설계</p><p class="value">XGB·TabPFN 두 모델로 갱신 지연을 포함한 재생을 진행한다. SOTA 3종은 이번 실행에서 제외. <button class="jump" data-jump="'+REPORT_ID+'">시나리오·측정 기준·근거 논문</button>만 수록했다.</p></div>'
    flow=flow or '<div class="flow-step"><time>10-14</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div><div class="body"><button class="jump" data-jump="'+REPORT_ID+'"><h3>'+title+'</h3></button><p>갱신 중 이전 버전으로 탐지하고, 새 버전의 준비 시점과 추론 대기열을 실측 시간으로 재생한다. 현재 페이지는 설명과 인용만 포함한다.</p></div></div>'
    for name,body,anchor in [('CARD',card,'<div class="status-grid">'),('FLOW',flow,'<div class="flow-rail">')]:
        wrapped=f'<!-- EXP72_{name}_START -->'+body+f'<!-- EXP72_{name}_END -->'
        pattern=f'<!-- EXP72_{name}_START -->.*?<!-- EXP72_{name}_END -->'
        if re.search(pattern,text,re.S):text=re.sub(pattern,lambda _:wrapped,text,flags=re.S)
        else:text=text.replace(anchor,anchor+'\n'+wrapped,1)
    text=text.replace('2026-08-18 &ndash; 2026-10-13','2026-08-18 &ndash; 2026-10-14')
    index.write_text(text,encoding='utf-8')
    print(dest)


def main():
    dest=ROOT/'lablog/html_report'/f'{REPORT_ID}.html'
    if dest.exists() and '<!-- EXP72_RESULTS_START -->' in dest.read_text():
        raise SystemExit('Results already published. Use build_results_report.py --run RUN to preserve them.')
    publish(HTML)

if __name__=='__main__':main()
