#!/usr/bin/env python3
"""한국어 상세 LaTeX와 단일 흐름 로그 작성, 임시 PDF 컴파일 검증."""
from __future__ import annotations
import argparse,hashlib,json,re,shutil,subprocess,tempfile
from pathlib import Path
from experiment_paths import repo_root
from lablog_format import convert_md,esc
ROOT=repo_root(); OUT=ROOT/'docs/lablog'

def write_record(destination,text,*,summary=None,compile_pdf=True):
    """메모리상 Markdown을 변환한다. Markdown 상세 파일은 생성하지 않는다.
    날짜·실험 ID·해석은 호출자가 제공하며 현재 시각이나 성공 여부를 추정하지 않는다.
    """
    destination=Path(destination)
    date=next((p for p in destination.parts if re.fullmatch(r'20\d{6}',p)),None)
    experiment=re.search(r'exp\d+',destination.stem)
    if not date or not experiment: raise ValueError('기록 경로에 실제 실험 날짜 YYYYMMDD와 expNN을 지정하세요')
    eid=experiment.group(); target=OUT/'reports'/f'{date}_{eid}.tex'
    title=next((s.lstrip('# ').strip() for s in text.splitlines() if s.startswith('#')),eid.upper())
    display=f'{date[:4]}-{date[4:6]}-{date[6:]}'
    body='\n'.join([f'% !TeX root = {target.name}','% !LW recipe=lablog XeLaTeX',r'\input{_support/style.tex}',r'\begin{document}',r'\reporttitle{'+display+'}{'+esc(title)+'}',convert_md(text,destination),r'\end{document}',''])
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(body)
    summary=summary or f'{title} — 결과·설정과 해석 기록 갱신'
    entry=f'- {display} · **{eid.upper()}** · {summary}. [LaTeX](reports/{target.name})'
    log=OUT/'log.md';lines=log.read_text().splitlines()
    pattern=re.compile(r'^- '+display+r' · \*\*(?:v\d+/)?'+eid+r'\*\*',re.I)
    hits=[i for i,line in enumerate(lines) if pattern.search(line)]
    if hits:
        lines[hits[0]]=entry
        for i in reversed(hits[1:]):lines.pop(i)
    else:
        position=next((i for i,line in enumerate(lines) if line.startswith('- 20') and line[2:12]>display),len(lines))
        lines.insert(position,entry)
    log.write_text('\n'.join(lines)+'\n')
    if compile_pdf:
        if shutil.which('xelatex'):
            build_report(target)
        else:
            # Reporting dependencies must not turn a completed model run into a
            # failed experiment. The missing PDF remains visible to `check`.
            pending=' (컴파일 검증 대기: 실행 환경에 XeLaTeX 없음)'
            log.write_text(log.read_text().replace(entry, entry+pending))
            print(f'LaTeX 기록 완료; 임시 컴파일 검증 대기: {target}', flush=True)
    return target

def build_report(path):
    path=Path(path).resolve()
    if not shutil.which('xelatex'):raise RuntimeError(f'LaTeX 기록 완료, 컴파일 검증 대기: XeLaTeX 필요 ({path})')
    with tempfile.TemporaryDirectory(prefix='imbalcic-report-') as directory:
        command=['xelatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',f'-output-directory={directory}',str(path)]
        for _ in range(2):
            result=subprocess.run(command,cwd=OUT,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(f'{path.name}:\n{result.stdout[-4500:]}')
        log=(Path(directory)/(path.stem+'.log')).read_text(errors='replace')
        missing=[s for s in log.splitlines() if 'Missing character:' in s]
        if missing:raise RuntimeError(f'{path.name}: 누락된 글자: '+'\n'.join(missing[:20]))
        pdf=Path(directory)/path.with_suffix('.pdf').name
        return dict(report=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                    pdf_persisted=False,compiled_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),missing_glyphs=0,
                    overfull_boxes=log.count('Overfull \\hbox'))

def check():
    issues=[]
    allowed_files={
        ROOT/'docs/lablog': {'log.md', '.gitignore'},
        ROOT/'docs/latex': {'main.tex', 'ref.bib', '.gitignore'},
        ROOT/'docs/slides': {'.gitignore'},
    }
    for directory, allowed in allowed_files.items():
        for p in directory.iterdir():
            if p.is_file() and p.name not in allowed:
                issues.append('보조 파일을 _support로 옮겨야 함: '+str(p.relative_to(ROOT)))
    for lang in ('eng', 'kor'):
        for p in (ROOT/'docs/slides'/lang).iterdir():
            if p.name!='.gitignore' and (not p.is_file() or p.suffix not in ('.pptx', '.pdf')):
                issues.append('발표자료 열람 폴더의 보조 자원: '+str(p.relative_to(ROOT)))
    for path in ['docs/research','lablog','docs/lablog/main.tex','docs/lablog/main.pdf','docs/latex/experiment_record.tex']:
        if (ROOT/path).exists():issues.append('폐기 경로가 다시 생성됨: '+path)
    if sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob('*.md'))!=['log.md']:issues.append('lablog에는 log.md 하나만 허용')
    for ref in re.findall(r'\]\(([^)]+)\)',(OUT/'log.md').read_text()):
        if '://' not in ref and not (OUT/ref).exists():issues.append('로그 연결 없음: '+ref)
    reports=sorted((OUT/'reports').glob('*.tex'))
    for p in reports:
        s=p.read_text()
        if r'\begin{document}' not in s or r'\end{document}' not in s:issues.append('독립 문서 아님: '+p.name)
        if p.with_suffix('.pdf').exists():issues.append('상시 보관하지 않는 보고서 PDF: '+p.with_suffix('.pdf').name)
        for ref in re.findall(r'\\(?:input|includegraphics(?:\[[^]]*\])?|reportfigure)\{([^}]+)\}',s):
            if not (OUT/ref).exists():issues.append(p.name+'의 연결 없음: '+ref)
    return dict(passed=not issues,reports=len(reports),issues=issues)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['build','check']);parser.add_argument('--report',type=Path);args=parser.parse_args()
    if args.command=='check':
        value=check();print(json.dumps(value,ensure_ascii=False,indent=2));raise SystemExit(not value['passed'])
    reports=[args.report] if args.report else sorted((OUT/'reports').glob('*.tex'));records=[]
    for p in reports:
        records.append(build_report(p));print('compiled '+p.name,flush=True)
    validation_path=OUT/'_support/validation.json'
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
    previous={r['report']:r for r in validation.get('reports',[])} if args.report else {}
    previous.update({r['report']:r for r in records})
    validation.update(reports=[previous[k] for k in sorted(previous)],structure=check())
    validation_path.write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
