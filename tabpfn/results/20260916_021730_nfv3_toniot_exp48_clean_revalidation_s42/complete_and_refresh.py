import json, subprocess, sys, os
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path('/home/user/Desktop/imbalcic')
command=[sys.executable,'-u',str(repo/'tabpfn/scripts/nfv3_v3_exp48_ton_clean_revalidation.py'),'--run-dir',str(root)]
code=subprocess.call(command)
builder=repo/'docs/research/20260916/build_clean_revalidation_report.py'
if builder.exists():
    report_code=subprocess.call([sys.executable,str(builder)])
    if report_code == 0:
        report_code=subprocess.call([sys.executable,str(repo/'scripts/sync_html_reports.py')])
    if report_code == 0:
        report_code=subprocess.call([sys.executable,str(repo/'scripts/sync_html_reports.py'),'--check'])
    (root/'REPORT_REFRESH.json').write_text(json.dumps({'returncode':report_code,'online_artifact_updated':False}))
sys.exit(code)
