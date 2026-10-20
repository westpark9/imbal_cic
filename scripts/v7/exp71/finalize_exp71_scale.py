#!/usr/bin/env python3
"""After a managed scale run, validate predictions and publish its local report."""

# HTML reporting was retired by the 2026-10-20 documentation reorganization.
if __name__ == '__main__':
    raise SystemExit('이 HTML 발행 명령은 폐기되었습니다. 결과를 검토한 뒤 scripts/common/lablog.py의 write_record로 한국어 LaTeX·PDF와 log.md를 갱신하세요. 기존 결과 계산 함수는 이력 참조용입니다.')

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from exp71_bounded_updates import ROOT, sha, utc, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--previous-run', type=Path, required=True)
    p.add_argument('--expected-html-sha256', required=True)
    args = p.parse_args(); run = args.run.resolve(); previous = args.previous_run.resolve()
    state_file = run/'postprocess_status.json'
    write(state_file, dict(state='waiting_for_experiment', pid=os.getpid(), started=utc()))
    started = time.monotonic()
    while True:
        meta = json.loads((run/'RUN_META.json').read_text())
        status = json.loads((run/'status.json').read_text())
        if meta['status'] in ['failed', 'interrupted', 'invalid'] or status['state'] == 'failed':
            write(state_file, dict(state='experiment_failed', experiment_status=status, updated=utc()))
            return 1
        if (run/'COMPLETE.json').exists() and meta['status'] in ['finished','complete']:
            break
        if time.monotonic() - started > 8 * 3600:
            write(state_file, dict(state='waiting_timeout', updated=utc()))
            return 1
        time.sleep(20)

    def execute(*argv):
        print('Executing', *argv, flush=True)
        subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, check=True)

    try:
        write(state_file, dict(state='validating_saved_predictions', updated=utc()))
        execute('scripts/v7/exp71/verify_exp71_results.py', '--run', run)
        report = ROOT/'docs/lablog/reports/20261014.tex'
        # Respect any report edits made while this long experiment was running.
        if sha(report) != args.expected_html_sha256:
            write(state_file, dict(state='validated_report_changed_requires_merge', updated=utc(),
                reason='The report was edited after this finalizer was scheduled; results remain validated.'))
            execute('scripts/results_versions.py', 'mark', run, '--evidence', 'exploratory',
                    '--status', 'complete', '--note', '100k predictions validated; later HTML edits preserved, publication requires merge.')
            return 0
        write(state_file, dict(state='publishing_local_report', updated=utc()))
        execute('scripts/v7/exp71/build_exp71_cumulative_report.py', '--run', run, '--previous-run', previous)
        execute('scripts/sync_html_reports.py')
        execute('scripts/sync_html_reports.py', '--check')
        execute('scripts/v7/exp71/verify_exp71_cumulative_html.py', '--run', run,
                '--qa-dir', run/'report_revisions/cumulative_only/qa_complete')
        dest = run/'report_revisions/cumulative_only/after_complete'
        for source in ['docs/lablog/reports/20261014.tex', 'lablog/html_report/post_tabpfn.html',
                       'docs/lablog/reports/20261014.tex']:
            out = dest/source; out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/source, out)
        execute('scripts/results_versions.py', 'mark', run, '--evidence', 'exploratory', '--status', 'complete',
                '--note', '100k cumulative uniform/natural seed43; full prediction and HTML validation passed. Single-seed exploratory evidence.')
        write(state_file, dict(state='complete', updated=utc(), local_report=str(report),
                              online_artifact_updated=False, git_pushed=False))
        return 0
    except Exception as error:
        write(state_file, dict(state='postprocess_failed', error=repr(error), updated=utc()))
        raise


if __name__ == '__main__':
    sys.exit(main())
