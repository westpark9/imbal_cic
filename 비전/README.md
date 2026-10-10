# Computer Vision Homework 1

`Homework1_CV.pdf`의 A, B, C 실행 코드를 `homework1_cv.py` 한 파일로 통합했습니다. 핵심 영상처리 연산은 NumPy 배열 연산과 FFT/IFFT로 직접 구현했고, Pillow와 Matplotlib은 이미지 입출력과 시각화에 사용합니다.

## 파일

- `homework1_cv.py`: A/B/C 전체 제출용 실행 코드
- `Homework1_Report.docx`: 영문 보고서
- `Homework1_Report_KO.docx`: 내용 검토용 국문 보고서
- `Homework1_CV_solution.ipynb`: 실행 결과가 포함된 검토용 노트북
- `images/`: 실행에 필요한 입력 이미지
- `output/`: 실행 결과 그림
- `results_A.json`, `results_B.json`, `results_C.json`: 정량 결과

`build_report.py`, `build_report_ko.py`, `report_updates.py`, `report_presentation.py`, `build_notebook.py`는 문서·노트북 작성 도구입니다. 제출용 영상처리 코드는 `homework1_cv.py`만 사용하며 작성 도구를 가져오지 않습니다.

## 실행

Python 3.13에서 검증했습니다. 입력 이미지 폴더를 코드 옆에 둡니다.

```bash
pip install -r requirements.txt
python homework1_cv.py
```

일부 파트만 실행할 수도 있습니다.

```bash
python homework1_cv.py --part A
python homework1_cv.py --part B C
```

`run_part_a()`, `run_part_b()`, `run_part_c()`가 각 파트를 실행합니다. 내부 함수는 `a_`, `b_`, `c_` 접두사로 구분합니다. 저장 경로는 스크립트 위치를 기준으로 하며, JSON의 그림 경로는 상대 경로입니다.

## 보고서와 노트북 재생성

보고서는 저장된 결과를 읽으므로 전체 코드를 실행한 뒤 생성합니다.

```bash
python build_report.py
python build_report_ko.py
```

노트북 Part B는 같은 폴더의 `homework1_cv.py`를 불러옵니다. 노트북을 실행할 때 작업 폴더를 이 폴더로 설정합니다. 출력 없는 노트북을 재생성할 때만 다음 명령을 사용합니다.

```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace Homework1_CV_solution.ipynb
```

노트북과 스크립트는 일부 그림 파일명을 공유합니다. 보고서 재생성 시에는 노트북 실행 후 `python homework1_cv.py`를 실행해 보고서용 그림을 저장합니다.

## 결과 해석

A-9는 결과 영상·히스토그램과 지시문의 다섯 비교 항목을 가로 페이지의 표로 비교합니다. C파트에는 K별 MSE·PSNR·SSIM 표를 유지합니다. 질문별 답변에만 하이픈 불렛과 질문 번호를 붙이며, 관찰·설명 문단은 일반 문단입니다. 영문 글꼴은 Times New Roman입니다. 앞부분은 코드와 결과 폴더 안내로 줄였으며 전체 요약은 삭제했습니다.

B파트는 클리핑 전 공간영역·주파수영역 결과를 비교합니다. C파트 표는 복원값을 [0,1]로 클리핑한 뒤 원본과 비교한 값입니다. 클리핑 전 지표와 범위 밖 화소 비율은 `results_C.json`의 `raw_table`에 보관합니다. 열화와 복원은 동일한 순환 합성곱 경계조건을 사용합니다.
