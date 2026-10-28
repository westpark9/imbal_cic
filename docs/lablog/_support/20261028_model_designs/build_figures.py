#!/usr/bin/env python3
"""Build the four proposal diagrams; keep only figure PDFs in figures/."""
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
LABLOG = HERE.parent.parent
PREFIX = r"""\documentclass[tikz,border=7pt]{standalone}
\usepackage{fontspec,kotex,amsmath}
\setmainfont{DejaVu Sans}
\setmainhangulfont{Noto Sans CJK KR}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds}
\definecolor{blue}{HTML}{245B8A}
\definecolor{orange}{HTML}{AD5A17}
\definecolor{green}{HTML}{287255}
\tikzset{
  box/.style={draw=blue,fill=blue!6,rounded corners=3pt,align=center,
    text width=4.1cm,minimum height=1.1cm,inner sep=6pt,font=\small},
  train/.style={box,draw=orange,fill=orange!8},
  deploy/.style={box,draw=green,fill=green!8},
  note/.style={align=center,font=\footnotesize,text=black!80},
  arr/.style={-{Stealth[length=2.4mm]},line width=.8pt,draw=black!70},
  upd/.style={arr,dashed,draw=orange},
  label/.style={font=\scriptsize,fill=white,inner sep=2pt}}
\begin{document}
\begin{tikzpicture}
"""
SUFFIX = "\n\\end{tikzpicture}\n\\end{document}\n"
FIGURES = {
"a_architecture": r"""
\node[box] (l) at (0,0) {참라벨 집합 $L$\\공격당 $n$개 + 정상};
\node[box] (u) at (7.1,0) {비라벨 학습 집합 $U$\\정답은 모델에 제공하지 않음};
\node[box] (t) at (0,-2) {TabPFN 교사\\가중치 고정 / context는 $L$\\$p^T(y\mid x,L)$};
\node[train] (e) at (7.1,-2) {수치형 flow encoder $f_\theta$\\마스킹 복원 + 참라벨 준비\\표현 $z$와 클래스 대표점 $\mu$};
\node[train,text width=7.2cm] (q) at (3.55,-4.1) {교사 확률 + 대표점 거리로 soft label $Q$ 배정\\비균등 클래스 비율은 약한 제약으로 사용\\불확실한 배정은 학습 기여를 낮춤};
\node[train,text width=7.2cm] (s) at (3.55,-6.1) {학생 encoder 학습 + 대표점 갱신\\참라벨 손실 + soft label 손실 + 복원 손실\\참라벨은 유지 / 교사로 역전파하지 않음};
\node[deploy,text width=10cm] (d) at (3.55,-8.1) {배포: 새 flow $x$ $\longrightarrow$ 학습한 $f_\theta(x)$
  $\longrightarrow$ 대표점 분류 $p^S(y\mid x)$\\TabPFN과 복원 decoder는 배포 경로에서 제외};
\draw[arr] (l)--(t);
\draw[arr] (u)--(e);
\draw[arr] (u.west)--(3.6,0)--(3.6,-1.2)--(t.north east);
\draw[arr] (l.north)--(0,.9)--node[label,above]{학생의 참라벨 학습}(9.8,.9)--(9.8,-1.2)--(e.north east);
\draw[arr] (t.south)--(t.south|-q.north);
\draw[arr] (e.south)--(e.south|-q.north);
\draw[arr] (q)--(s);
\draw[upd] (s.east)--(10.1,-6.1)--(10.1,-2)--(e.east);
\node[note,rotate=90] at (10.4,-4.1) {표현·대표점 반복 갱신};
\draw[arr] (s)--(d);
\node[note] at (3.55,-9.35) {파랑: 고정 교사·입력 \quad 주황: 학습/갱신 \quad 초록: 최종 추론\\실선: 정보 전달 \quad 점선: 학생 상태 갱신};
""",
"a_flow": r"""
\node[box,text width=10.7cm] (a) at (0,0) {① 분할을 먼저 고정: $L$ / 비라벨 $U$ / 평가 $T$\\$U$의 정답과 $T$ 전체를 학습·선택에서 차단};
\node[train,text width=10.7cm] (b) at (0,-1.8) {② 학생 준비: $L\cup U$의 가려진 특징 복원 + $L$의 분류\\대표점은 참라벨로 초기화 / TabPFN은 $L$로 $U$를 예측};
\node[train,text width=10.7cm] (c) at (0,-3.6) {③ 고정 교사 확률 + 현재 학생 거리로 $Q$ 배정\\행별 확률 합 1 / 클래스 균등 할당을 강제하지 않음};
\node[train,text width=10.7cm] (d) at (0,-5.4) {④ $Q$를 고정하고 학생 가중치 학습, 대표점 갱신\\추정 비율은 교사 초기값에 묶어서 천천히 갱신};
\node[box,text width=10.7cm] (e) at (0,-7.2) {⑤ 사전에 고정한 반복 수로 종료 / 독립 개발자료가 있으면 조기 종료\\$U$의 숨긴 정답이나 test 점수로 최적 epoch를 고르지 않음};
\node[deploy,text width=10.7cm] (f) at (0,-9) {⑥ 학생을 고정한 뒤 $T$ 전체 평가\\공격 F1 + 정상 오탐 + 적응 비용 + 추론 비용 기록};
\foreach \a/\b in {a/b,b/c,c/d,d/e,e/f}{\draw[arr] (\a)--(\b);}
\draw[upd] (d.east)--(6.4,-5.4)--(6.4,-3.6)--(c.east);
\node[note,rotate=90] at (6.7,-4.5) {고정 횟수 반복};
""",
"b_architecture": r"""
\node[box,text width=3.5cm] (l) at (0,0) {참라벨 전체 $L$\\Global이 모두 사용};
\node[box,text width=3.5cm] (g) at (4.9,0) {Global TabPFN\\가중치 고정};
\node[box,text width=3.5cm] (s0) at (9.8,0) {전체 클래스 점수\\$s^{(0)}=\log p_G$};
\node[box,text width=3.5cm] (x) at (0,-2.1) {새 query $x$\\정답 라벨 사용 없음};
\node[box,text width=3.5cm] (e1) at (4.9,-2.1) {쌍 전문가 1\\예: 정상 / Infiltration\\$L$의 두 클래스만\\이진 context로 사용};
\node[train,text width=3.5cm] (c1) at (9.8,-2.1) {쌍 1의 점수차만 보정\\$s_a\!\leftarrow s_a+\delta/2$\\$s_b\!\leftarrow s_b-\delta/2$};
\node[box,text width=3.5cm] (e2) at (4.9,-4.4) {쌍 전문가 2\\이전 보정 뒤 남은 혼동\\$L$의 두 클래스만\\이진 context로 사용};
\node[train,text width=3.5cm] (c2) at (9.8,-4.4) {쌍 2의 점수차 보정\\순서·계수·상한은\\평가 전에 고정};
\node[deploy,text width=8.4cm] (out) at (7.35,-6.5) {최종 softmax $\longrightarrow$ 최종 클래스\\다른 logit 직접 수정 없음\\정규화 확률과 다른 클래스 예측은 바뀔 수 있음};
\draw[arr] (l)--(g);
\draw[arr] (g)--(s0);
\draw[arr] (x.north)|-(g.south west);
\draw[arr] (x)--(e1);
\draw[arr] (x.south)--(0,-4.4)--(e2.west);
\draw[arr] (e1)--node[label,above]{$t_{ab}$}(c1);
\draw[arr] (e2)--(c2);
\draw[arr] (s0)--(c1);
\draw[arr] (c1)--(c2);
\draw[arr] (c2.south)--(c2.south|-out.north);
\node[note,text width=4.0cm] at (.2,-6.3) {Global·전문가의\\TabPFN 가중치는 모두 고정\\학습/선택 대상: 쌍, 순서, 보정 계수};
\node[note,text width=12.5cm] at (5,-8) {그림의 클래스 쌍은 설명 예시이며 test에서 미리 선택하지 않음.\\초기안은 선택된 전문가를 모두 호출하므로 호출 비용 절감은 별도 검증 대상.};
""",
"b_flow": r"""
\node[box,text width=10.4cm] (a) at (0,0) {① $L$을 동일 특징 벡터 group 단위로 fold 분리\\한 행의 검증 예측에는 그 행/동일 벡터의 라벨을 쓰지 않음};
\node[box,text width=10.4cm] (b) at (0,-1.8) {② Global의 out-of-fold 예측으로 혼동 쌍 후보 생성\\클래스당 오류를 정규화 / 정상 오탐도 따로 집계};
\node[box,text width=10.4cm] (c) at (0,-3.6) {③ fold 학습부의 해당 두 클래스만 전문가 context로 사용\\같은 fold 검증부에서 이진 확률과 보정 후 다중분류를 계산};
\node[train,text width=10.4cm] (d) at (0,-5.4) {④ 제한된 보정 계수 후보 비교\\공격 손실 감소와 정상 오탐 제약을 함께 확인};
\node[train,text width=10.4cm] (e) at (0,-7.2) {⑤ 안정된 개선이면 채택 후 남은 혼동을 다시 계산\\표본 부족·클래스 누락·기준 미충족이면 보정 0 / 종료};
\node[deploy,text width=10.4cm] (f) at (0,-9) {⑥ 전체 $L$로 Global·채택 전문가 context를 재구성\\고정된 순서와 계수로 $T$ 평가 / 최대 전문가 수를 제한};
\foreach \a/\b in {a/b,b/c,c/d,d/e,e/f}{\draw[arr] (\a)--(\b);}
\draw[upd] (e.east)--(6.3,-7.2)--(6.3,-1.8)--(b.east);
\node[note,rotate=90] at (6.6,-4.5) {채택된 보정 이후 다시 계산};
\node[note,text width=11.5cm] at (0,-10.2) {OOF는 선택용이다. 같은 OOF에서 고른 성능을 최종 일반화 성능으로 보고하지 않는다.};
""",
}

def main():
    destination = LABLOG / "figures"
    destination.mkdir(exist_ok=True)
    for name, body in FIGURES.items():
        source = HERE / f"{name}.tex"
        source.write_text(PREFIX + body + SUFFIX)
        with tempfile.TemporaryDirectory(prefix="model-design-figure-") as temporary:
            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                 f"-output-directory={temporary}", str(source)],
                cwd=HERE, capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(proc.stdout[-4000:])
            log = (Path(temporary) / f"{name}.log").read_text()
            if "Missing character:" in log:
                raise RuntimeError(f"Missing glyph in {name}")
            target = destination / f"20261028_design_{name}.pdf"
            shutil.copy2(Path(temporary) / f"{name}.pdf", target)
            print(target.relative_to(LABLOG))

if __name__ == "__main__":
    main()
