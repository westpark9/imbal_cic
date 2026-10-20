#!/usr/bin/env python3
"""Finish verification and slide rendering after the registered GPU runs."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import json,os,subprocess,time,traceback
from mixed_query_eval import LARGE,EXPANDED

MIXED=_root/'tabpfn/results/v7/exp81/20261016_025421_mixed_query_final_s43_b6ba1d34'
STATUS=MIXED/'slide_finalization.json'


def status(phase,**kw):
    d=dict(phase=phase,updated_epoch=time.time(),**kw)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(STATUS)
    print(json.dumps(d,ensure_ascii=False),flush=True)


def wait(run):
    status('waiting',run=str(run))
    while True:
        m=json.loads((run/'RUN_META.json').read_text())
        if m['status']=='finished':
            if m['returncode']!=0:raise RuntimeError(f'Experiment failed: {run}')
            return
        time.sleep(10)


def call(args,env=None):
    status('executing',command=[str(x) for x in args])
    subprocess.run([str(x) for x in args],cwd=_root,env=env,check=True)


def main():
    wait(LARGE)
    wait(EXPANDED)
    # Keep independent CPU verification out of the timed EXP80 update phase.
    call([sys.executable,'scripts/v7/exp79/verify_final.py',LARGE])
    call([sys.executable,'scripts/v7/exp80/finalize_metadata.py',EXPANDED])
    call([sys.executable,'scripts/v7/exp80/verify_extended.py',EXPANDED])
    wait(MIXED)
    call([sys.executable,'scripts/v7/exp81/verify_mixed.py',MIXED])
    call([sys.executable,'scripts/v7/exp81/build_research_record.py'])
    call([sys.executable,'docs/slides/_support/scripts/build_1015_compact.py'])
    office='/tmp/imbalcic-slide-office/root/usr/lib/libreoffice/program'
    env=dict(os.environ,LD_LIBRARY_PATH=office+':/tmp/imbalcic-slide-office/root/usr/lib/x86_64-linux-gnu')
    call([office+'/soffice','-env:UserInstallation=file:///tmp/imbalcic-slide-office/profile','--headless','--convert-to','pdf','--outdir','docs/slides/kor','docs/slides/kor/1015_labmeeting_ko.pptx'],env)
    call([sys.executable,'docs/slides/_support/scripts/validate_1015_compact.py'])
    call(['pdftoppm','-f','6','-l','11','-scale-to','1600','-png','docs/slides/kor/1015_labmeeting_ko.pdf','docs/slides/_support/qa/kor/1015_labmeeting_qa/compact/page'])
    status('rendered_pending_visual_review',slides=11,automated_validation=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=repr(e));traceback.print_exc();raise
