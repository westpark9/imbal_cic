#!/usr/bin/env python3
"""Publish the requested EXP74/75 report from verified, immutable result files."""

# HTML reporting was retired by the 2026-10-20 documentation reorganization.
if __name__ == '__main__':
    raise SystemExit('이 HTML 발행 명령은 폐기되었습니다. 결과를 검토한 뒤 scripts/common/lablog.py의 write_record로 한국어 LaTeX·PDF와 log.md를 갱신하세요. 기존 결과 계산 함수는 이력 참조용입니다.')

from pathlib import Path
import base64
import hashlib
import json
import re
import sys

_root = next(p for p in Path(__file__).resolve().parents
             if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import repo_root

ROOT = repo_root(__file__)
REPORT_ID = 'architecture_adaptation_1015'
RUNS = {
    'exp72': '20261014_134735_replay_100k_xgb_tabpfn_s43_de8b4a78',
    'exp73': '20261014_154250_replay_current_sota_100k_s43_b0f8a9dc',
    'exp74': '20261015_171517_heads_replay_100k_s43_d74bc5fb',
    'exp75': '20261015_181113_icd_replay_100k_s43_6558120d',
}
METHODS = {'xgb': 'exp72', 'tabpfn': 'exp72', 'ours': 'exp73',
           'heads': 'exp74', 'icd_init': 'exp75', 'icd': 'exp75'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified():
    roots = {e: ROOT / 'tabpfn/results/v7' / e / r for e, r in RUNS.items()}
    hashes, jobs, status = {}, {}, {}

    def source(path):
        hashes[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    for exp, expected in [('exp72', 24), ('exp73', 48), ('exp74', 12), ('exp75', 24)]:
        root = roots[exp]
        st, v = source(root / 'status.json'), source(root / 'verification.json')
        assert st['state'] == 'complete', (exp, st)
        assert v['state'] == 'passed' and v['verified_policies'] == expected, (exp, v)
        status[exp] = dict(run=str(root.relative_to(ROOT)), completed=st['completed'],
                           verified_policies=v['verified_policies'])
    for ds in ['cic2018', 'toniot']:
        for case in ['natural', 'uniform']:
            for method, exp in METHODS.items():
                key = f'{ds}_{case}_{method}'
                path = roots[exp] / 'jobs' / key
                result = source(path / 'results.json')
                assert read(path / 'COMPLETE.json')['results_sha256'] == sha(path / 'results.json'), key
                fits = []
                for stage in sorted(path.glob('stage_*')):
                    fit = source(stage / 'fit_metrics.json')
                    fit.pop('phases', None)
                    for name in ['head_training', 'context_training']:
                        p = stage / (name + '.json')
                        if p.exists():
                            fit[name] = source(p)
                            if name == 'head_training':
                                for h in fit[name]['heads']:
                                    h.pop('epochs', None)
                    fits.append(fit)
                jobs[key] = dict(results=result['results'], fits=fits,
                                 updates=source(path / 'schedule.json')['updates_seconds'])
                if method == 'heads':
                    jobs[key]['heads'] = result['head_diagnostics_ideal']
                for policy, r in result['results'].items():
                    expected = 3042473 if ds == 'cic2018' else 2271723
                    assert r['rows'] == r['evaluated_rows'] == expected
                    assert r['each_test_row_once']
                    assert sum(sum(row) for row in r['confusion']) == expected
                    assert abs(sum(c['f1'] for c in r['classes']) / len(r['classes']) - r['macro_f1']) < 1e-12
                if method in ['heads', 'icd']:
                    assert fits[-1]['train_rows'] == 100000
    # Protect the prose's key comparisons from accidentally using another run.
    def metric(ds, case, method):
        return jobs[f'{ds}_{case}_{method}']['results']['ideal']
    for ds in ['cic2018', 'toniot']:
        for case in ['natural', 'uniform']:
            for method in ['heads', 'icd']:
                assert metric(ds, case, method)['macro_f1'] < metric(ds, case, 'tabpfn')['macro_f1']
        assert jobs[f'{ds}_natural_heads']['heads']['head_natural']['macro_f1'] > metric(ds, 'natural', 'tabpfn')['macro_f1']
    chosen = [f['context_training']['best_step'] for k, j in jobs.items()
              if k.endswith('_icd') for f in j['fits']]
    assert len(chosen) == 22 and chosen.count(0) == 4
    return dict(jobs=jobs, methods=list(METHODS), runs=status, source_hashes=hashes,
                seed=43, research_version='v7', report_date='2026-10-15')


def publish(page):
    dest = ROOT / 'lablog/html_report' / (REPORT_ID + '.html')
    dest.write_text(page, encoding='utf-8')
    index = ROOT / 'lablog/html_report/post_tabpfn.html'
    text = index.read_text()
    match = re.search(r'const REPORTS = (.*?);\n', text)
    reports = json.loads(match.group(1))
    reports = [r for r in reports if r['id'] != REPORT_ID]
    reports.insert(1, dict(id=REPORT_ID, date='2026-10-15',
        titleKr='구조 개선: 분포별 head와 context 최적화',
        titleEn='Distribution heads and in-context distillation',
        verdict='2·3안 완료 · 단독 head 가능성 · 결합·오탐 개선 필요',
        verdictClass='neutral', group='report', type='iframe',
        b64=base64.b64encode(page.encode()).decode()))
    text = text[:match.start(1)] + json.dumps(reports, ensure_ascii=False) + text[match.end(1):]
    text = re.sub(r'<p class="lede">.*?</p>',
        '<p class="lede">새 공격 도입 시나리오에서 기존 S/V 모델을 비교한 뒤, 공통 TabPFN 표현의 분류 head와 '
        'context 최적화를 시험했다. 단독 head의 개선 가능성, 결합 후 정상 오탐, 최적화 효과와 갱신 비용을 구분한다. '
        '<button class="jump" data-jump="'+REPORT_ID+'">10/15 구조 개선 결과</button>. '
        '기존 여섯 방법의 성능·비용은 10/14 탭에 보존했다.</p>', text, count=1, flags=re.S)
    card = '<div class="status-cell"><p class="label">10/15 · 구조 대안 2·3 완료</p><p class="value">자연분포 head 단독은 자연 배분 두 데이터에서 단일 PFN보다 높았다. 균형 결합은 오탐이 늘고, context 최적화는 균등 배분의 초기 상태 대비 개선에 그쳤다. <button class="jump" data-jump="'+REPORT_ID+'">성능·클래스 혼동·비용·후속 판단</button></p></div>'
    flow = '<div class="flow-step"><time>10-15</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div><div class="body"><button class="jump" data-jump="'+REPORT_ID+'"><h3>구조 개선 · 분포별 head와 context 최적화</h3></button><p>EXP74·75의 새 36개 정책 검증 완료. 단독 head와 결합, 초기 context와 최적화 context를 각각 비교해 다음 개선 대상을 좁힌다.</p></div></div>'
    for name, body, anchor in [('CARD', card, '<div class="status-grid">'), ('FLOW', flow, '<div class="flow-rail">')]:
        wrapped = f'<!-- EXP74_75_{name}_START -->{body}<!-- EXP74_75_{name}_END -->'
        pattern = f'<!-- EXP74_75_{name}_START -->.*?<!-- EXP74_75_{name}_END -->'
        if re.search(pattern, text, re.S):
            text = re.sub(pattern, lambda _: wrapped, text, flags=re.S)
        else:
            text = text.replace(anchor, anchor + '\n' + wrapped, 1)
    index.write_text(text, encoding='utf-8')
    return dest


def main():
    data = load_verified()
    record = ROOT / 'docs/lablog/evidence/20261015/exp74_75_report_data.json'
    record.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    page = Path(__file__).with_name('architecture_report.template.html').read_text()
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    assert page.count('__REPORT_DATA__') == 1
    page = page.replace('__REPORT_DATA__', payload)
    print(publish(page))
    print(f'Checked {len(data["jobs"])} jobs, {len(data["source_hashes"])} source files.')


if __name__ == '__main__':
    main()
