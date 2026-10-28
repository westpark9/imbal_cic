#!/usr/bin/env python3
"""Preserve source diagrams as vector PDF crops and build the comparison table."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'scripts/common/experiment_paths.py').exists())
LAB = ROOT/'docs/lablog'
OUT = LAB/'figures/20261028_design_sota'
EVIDENCE = LAB/'evidence/references/20261028'

CROPS = [
    ('beta', 'beta_2502.02527.pdf', 6, (54, 62, 544, 320), 'Figure 3', 'https://arxiv.org/pdf/2502.02527'),
    ('cure', 'cure_2606.18677.pdf', 3, (57, 68, 543, 196), 'Figure 1', 'https://arxiv.org/pdf/2606.18677'),
    ('tabdpt', 'tabdpt_2410.18164.pdf', 5, (106, 70, 520, 171), 'Figure 2', 'https://arxiv.org/pdf/2410.18164'),
]

ROWS = [
 ('LotBoNC', 'AsiaCCS 2026', 'lotbonc', 'Figure 4 · PDF 5쪽', 'https://doi.org/10.1145/3779208.3785276',
  '표현·대표점 학습', '바이트 encoder와 대표점을 갱신했다. 비균등 soft 배정이 핵심이다. 1안은 수치 flow와 TabPFN 교사를 추가하며 학생을 배포한다.'),
 ('TabDPT', '원문 v3, 2026', 'tabdpt', 'Figure 2 · PDF 5쪽', 'https://arxiv.org/html/2410.18164v3',
  '실제 표 사전학습', '실제 표의 열 예측 과제로 기반 모델을 학습했다. 1안은 범용 기반 모델 재학습 대신 별도 학생을 적응시킨다. 비라벨 접근 조건을 맞춘다.'),
 ('LoCalPFN', 'NeurIPS 2024', 'local', 'Figure 3 · PDF 5쪽', 'https://arxiv.org/abs/2406.05207',
  '검색 + 가중치 적응', '근접 context 검색과 PFN fine-tuning을 결합했다. 2안의 이진 쌍 분리는 query별 검색이 아니며 PFN 가중치를 고정한다.'),
 ('BoostPFN', 'AISTATS 2025', 'boost', 'Figure 1 · PDF 2쪽', 'https://proceedings.mlr.press/v258/wang25d.html',
  '순차 앙상블', '잔차로 다음 context를 선택하고 예측기를 더했다. 2안은 전체 클래스 출력을 더하는 대신 선택한 두 logit의 차이를 제한 수정한다.'),
 ('DistPFN', 'ICML 2026', 'dist', 'Figure 4 · PDF 4쪽', 'https://arxiv.org/abs/2605.04363',
  'posterior 보정', '가중치 학습 없이 context prior와 예측 posterior로 보정했다. 쌍 경계 학습과 별도 요소이며 query 집합 통계 접근을 공개한다.'),
 ('BETA', 'ICML 2025', 'beta', 'Figure 3 · PDF 6쪽', 'https://proceedings.mlr.press/v267/liu25cn.html',
  'encoder + 앙상블', 'PFN은 고정하고 입력 encoder를 학습했다. 여러 표현과 bootstrap context의 PFN 예측을 평균했다. 1안의 학생 단독 배포·2안의 쌍 수정과 구별한다.'),
 ('CURE', 'FMSD@ICML 2026 워크숍', 'cure', 'Figure 1 · PDF 3쪽', 'https://arxiv.org/abs/2606.18677',
  '고정 용량 버퍼', '최근 short bank와 불확실성·중복 기준 long bank를 갱신했다. 고정한 것은 용량이며 내용은 바뀐다. 기존 EXP84 버퍼 정책과 직접 비교할 대상이다.'),
]


def main():
    OUT.mkdir(exist_ok=True)
    manifest = []
    old = json.loads((LAB/'evidence/references/20261021/report_figure_sources_20261022.json').read_text())
    for key in ['lotbonc', 'local', 'boost', 'dist']:
        src = LAB/'figures/references_20261022'/f'{key}.pdf'
        dst = OUT/f'{key}.pdf'
        shutil.copy2(src, dst)
        record = next(r for r in old if r['asset'].endswith('/'+key+'.pdf'))
        manifest.append(dict(record, asset=str(dst.relative_to(ROOT)), reused_from=str(src.relative_to(ROOT))))
    for key, name, page, (left, top, right, bottom), figure, url in CROPS:
        src = EVIDENCE/name
        # Original media size was checked with pdfinfo: all three are 612x792 pt.
        tex = (r'\documentclass[border=0pt]{standalone}' '\n'
               r'\usepackage{graphicx}' '\n' r'\begin{document}' '\n'
               +f'\\includegraphics[page={page},trim={left}bp {792-bottom}bp {612-right}bp {top}bp,clip]'
               +'{'+str(src)+'}'+'\n'+r'\end{document}'+'\n')
        source = HERE/f'sota_{key}.tex'
        source.write_text(tex)
        with tempfile.TemporaryDirectory(prefix='design-sota-crop-') as tmp:
            result = subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',
                f'-output-directory={tmp}',str(source)],cwd=HERE,capture_output=True,text=True)
            if result.returncode: raise RuntimeError(result.stdout[-2500:])
            shutil.copy2(Path(tmp)/(source.stem+'.pdf'),OUT/f'{key}.pdf')
        manifest.append(dict(asset=str((OUT/f'{key}.pdf').relative_to(ROOT)), source=str(src.relative_to(ROOT)),
            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), pdf_page=page, figure=figure,
            crop_top_left_pdf_points=[left,top,right,bottom],url=url))
    for row in manifest:
        row['asset_sha256'] = hashlib.sha256((ROOT/row['asset']).read_bytes()).hexdigest()
    (EVIDENCE/'design_sota_figure_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    body = [r'\begingroup\small',
        r'\begin{longtable}{>{\raggedright\arraybackslash}p{.16\linewidth}>{\raggedright\arraybackslash}p{.52\linewidth}>{\raggedright\arraybackslash}p{.27\linewidth}}',
        r'\caption{대표 그림으로 비교한 SOTA 후보군. 순위표가 아니라 입력·갱신·최종 예측기의 구조 비교다. 원문 그림의 출처와 페이지를 각 행에 표시한다.}\\',
        r'\toprule 방법·계열 & 원문 대표 그림 & 학습·배포 방식과 우리 설계의 차이\\\midrule\endfirsthead',
        r'\toprule 방법·계열 & 원문 대표 그림 & 학습·배포 방식과 우리 설계의 차이\\\midrule\endhead',
        r'\bottomrule\endfoot']
    for i,(name,venue,key,figure,url,kind,description) in enumerate(ROWS):
        if i%2 == 0: body.append(r'\rowcolor{blue!4}')
        body += [r'\textbf{'+name+r'}\par\footnotesize '+venue+r'\par\vspace{3pt}'+kind+' &',
            r'\begin{minipage}[t]{\linewidth}\vspace{0pt}\centering',
            r'\includegraphics[width=\linewidth,height=4.1cm,keepaspectratio]{figures/20261028_design_sota/'+key+'.pdf}',
            r'\par\scriptsize\href{'+url+'}{'+figure+r' / 원문}\end{minipage} &',
            description+r'\\[7pt]\midrule']
    body += [r'\end{longtable}',r'\endgroup']
    (HERE/'sota_table.tex').write_text('\n'.join(body)+'\n')
    print('Built seven source figures and SOTA comparison table.')


if __name__ == '__main__': main()
