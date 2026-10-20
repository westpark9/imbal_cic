#!/usr/bin/env python3
"""Publish cumulative-only EXP71 results, optionally with a pending larger run."""

# HTML reporting was retired by the 2026-10-20 documentation reorganization.
if __name__ == '__main__':
    raise SystemExit('이 HTML 발행 명령은 폐기되었습니다. 결과를 검토한 뒤 scripts/common/lablog.py의 write_record로 한국어 LaTeX·PDF와 log.md를 갱신하세요. 기존 결과 계산 함수는 이력 참조용입니다.')

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from build_exp71_report import ROOT, PAGE, REPORT_ID, build_data, apply_rules
from exp71_cumulative_report import TITLE, make_page, notes


def integrate(data):
    path = ROOT/'lablog/html_report/post_tabpfn.html'
    text = path.read_text()
    match = re.search(r'const REPORTS = (.*?);\n', text)
    reports = json.loads(match.group(1))
    for report in reports:
        if report['id'] == REPORT_ID:
            report.update(titleKr=TITLE, titleEn='Cumulative updates · Cost and detection quality',
                          verdict='누적 전체 · 균등/자연 배정 · 비용·성능', verdictClass='neutral')
    text = text[:match.start(1)] + json.dumps(reports, ensure_ascii=False) + text[match.end(1):]
    pending = '10만 행 확대 실험은 실행 중이다.' if data.get('pending') else '완료된 누적 입력의 비용과 탐지 품질을 비교한다.'
    desc = ('원본 클래스의 출현 순서대로 라벨 사례를 추가하고 과거 사례를 모두 누적하여 단일 TabPFN과 XGBoost를 비교했다. '
            '공격 균등·공격 내 자연비율에서 갱신 비용, 추론 비용, 탐지 성능을 구분한다. '+pending)
    text = re.sub(r'<p class="lede">.*?</p>',
                  '<p class="lede">'+desc+' <button class="jump" data-jump="'+REPORT_ID+'">10/13 누적 갱신 결과</button>. 기존 expert·S/V 결과는 09/30 탭에 보존했다.</p>', text, count=1)
    replacements = {
        'FLOW': '<div class="flow-step"><time>10-13</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div><div class="body"><button class="jump" data-jump="'+REPORT_ID+'"><h3>'+TITLE+'</h3></button><p>'+desc+'</p></div></div>',
        'CARD': '<div class="status-cell"><p class="label">10/13 · 누적 전체 갱신</p><p class="value">'+pending+' 단계별 클래스 성능은 상세 토글에 보존했다. <button class="jump" data-jump="'+REPORT_ID+'">결과 보기</button>.</p></div>'}
    for key, body in replacements.items():
        text, count = re.subn('<!-- EXP71_'+key+'_START -->.*?<!-- EXP71_'+key+'_END -->',
                         '<!-- EXP71_'+key+'_START -->'+body+'<!-- EXP71_'+key+'_END -->', text, flags=re.S)
        assert count == 1
    path.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--pending-run', type=Path)
    p.add_argument('--previous-run', type=Path)
    a = p.parse_args(); run = a.run.resolve()
    data = build_data(run, 'cumulative')
    data['environment_note'] = ('RTX 4090, 동일 입력으로 모델 worker를 순차 실행했다. '
        'TabPFN ensemble 4, fit_with_cache, CPU KV cache; XGB CUDA hist 300 rounds, depth 8. '
        '단계별 1회 timing이며 프로세스 시작·데이터 준비·배포·직렬화는 비용에 포함하지 않는다.')
    if (run/'recovery.json').exists():
        data['environment_note'] += (' 이전 완료 단계는 보존했고 ToN 자연비율의 두 모델은 복구 후 전체 단계를 다시 실행했다. '
            '따라서 ToN 자연비율의 모델 간 비교는 같은 복구 환경이며, 다른 구성과 driver/kernel이 다를 수 있다.')
    if a.pending_run:
        pending = a.pending_run.resolve()
        assert json.loads((pending/'scale_request.json').read_text())['total'] == 100000
        status = json.loads((pending/'status.json').read_text())
        if status['state'] not in ['preparing', 'running']:
            raise ValueError('Pending run is not running: '+str(status))
        data['pending'] = dict(run=str(pending.relative_to(ROOT)), status=status,
                              checked_at=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%m/%d %H:%M KST'))
    if a.previous_run:
        old = build_data(a.previous_run.resolve(), 'cumulative')
        for key in ['seed', 'batch', 'threads', 'checkpoint_sha256']:
            assert data['request'][key] == old['request'][key], key
        for ds in data['datasets']:
            x, y = data['datasets'][ds], old['datasets'][ds]
            assert x['stages'] == y['stages'] and x['class_names'] == y['class_names']
            for name in ['test_X.npy','test_y.npy','test_ids.npy','test_hash.npy']:
                assert x['files'][name] == y['files'][name]
        assert json.loads((run/'scale_validation.json').read_text())['status'] == 'passed'
        data['previous'] = {key: old[key] for key in ['run','summary','datasets']}
    page = make_page(data, PAGE)
    report = ROOT/'lablog/html_report'/f'{REPORT_ID}.html'
    report.write_text(apply_rules(page.replace('__DATA__', json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</','<\\/'))))
    record = ROOT/'docs/lablog/reports/20261014.tex'
    record.parent.mkdir(parents=True, exist_ok=True); record.write_text(notes(data))
    integrate(data)
    snapshot = (a.pending_run.resolve() if a.pending_run else run)/'report_revisions/cumulative_only'
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot/'report_snapshot.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (snapshot/'report_provenance.json').write_text(json.dumps(dict(
        results_run=data['run'], pending=data.get('pending'),
        html_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        updated=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()), ensure_ascii=False, indent=2)+'\n')
    print('Built', report.relative_to(ROOT), 'and', record.relative_to(ROOT))


if __name__ == '__main__':
    main()
