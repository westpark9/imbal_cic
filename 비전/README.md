# Computer Vision Homework 1

`Homework1_CV.pdf`의 A, B, C를 직접 구현한 코드와 국문·영문 보고서입니다. 영상처리 연산은 NumPy 배열 연산과 FFT/IFFT를 이용해 작성했으며, PIL은 이미지 로딩에, Matplotlib은 시각화에 사용합니다.

## 제출 및 검토 파일

- `Homework1_Report.docx`: 영문 보고서
- `Homework1_Report_KO.docx`: 내용 검토용 국문 보고서
- `partA_point_processing.py`, `partB_filtering.py`, `partC_wiener.py`: 파트별 실행 코드
- `Homework1_CV_solution.ipynb`: 전체 실행을 완료한 검토용 노트북
- `images/`: 코드 실행에 필요한 입력 이미지
- `output/`, `results_A.json`, `results_B.json`, `results_C.json`: 재현 가능한 그림과 정량 결과

노트북 Part B는 같은 폴더의 `partB_filtering.py`를 가져와 실행하므로 함께 보관합니다. 실제 제출 파일의 선택은 강의 제출 방식에 따릅니다.

## 실행

Python 3.13에서 검증했습니다. 아래 명령은 이 폴더에서 실행합니다.

```bash
pip install -r requirements.txt
python partA_point_processing.py
python partB_filtering.py
python partC_wiener.py
```

각 스크립트는 `output/`에 그림을, `results_*.json`에 수치를 저장합니다. 저장 경로는 스크립트 위치를 기준으로 하며 결과 JSON의 그림 경로는 상대 경로입니다.

두 보고서는 저장된 결과를 읽으므로 코드 실행 후 생성합니다.

```bash
python build_report.py
python build_report_ko.py
```

`report_updates.py`는 두 언어의 공통 수정 사항과 표를 반영하는 보고서 생성 보조 모듈입니다.

Jupyter 또는 VS Code에서 노트북을 열어 모든 셀을 실행할 수 있습니다. 노트북의 초기 실행 폴더는 이 폴더로 설정합니다.

```bash
# 현재 출력이 없는 노트북을 다시 생성할 때만 실행합니다.
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace Homework1_CV_solution.ipynb
```

노트북과 스크립트는 일부 그림 파일명을 공유합니다. 보고서를 다시 만들 때는 파트별 스크립트를 실행한 후 보고서 생성기를 실행합니다. `build_explore.py`와 `clahe_clip_explore.ipynb`는 별도의 탐색 자료이며 제출 보고서의 생성 과정에 사용하지 않습니다.

## 검토에서 수정한 내용

- A: 히스토그램 CDF로 2·98 백분위수 선택, CLAHE 두 clip limit의 출력 히스토그램, 향상 전후 휘도 통계, 단일 재분배 방식의 clip limit 해석, 감마·색차 보존 설명을 보완했습니다.
- B: 실제 선형 합성곱의 패딩 격자에서 F·H·G를 표시하고, Gaussian sharpening의 σ=1·3 및 k=1·2 모든 조합을 공간·주파수 영역에서 비교했습니다. 샤프닝의 DC 이득, σ의 영향, 경계조건과 차이맵 표시도 설명했습니다.
- C: Wiener 필터의 목적과 K의 의미를 설명하고, [0,1] 클리핑 후 수치와 클리핑 전 원시 복원 수치를 구분했습니다. 열화와 복원에는 같은 순환 합성곱 경계조건을 사용합니다.

Gaussian sharpening의 네 조합은 MSE 약 10⁻²⁷~10⁻²⁶, SSIM은 표시 정밀도에서 1입니다. C의 기본 영상에서는 실험한 다섯 K 중 클리핑 후 PSNR은 K=0.01에서 25.04 dB, SSIM은 K=0.1에서 0.6987로 가장 높았습니다. 이 결과는 사용한 영상·잡음 시드·후보 집합에 대한 비교입니다.
