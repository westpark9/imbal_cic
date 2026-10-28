"""Finish independent checks and the report once the managed model run ends."""
from pathlib import Path
import argparse,json,subprocess,sys,time
from datetime import datetime

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args()
    run=args.run.resolve();directory=Path(__file__).resolve().parent
    state=run/'report_pipeline.json'
    def save(phase,**values):
        state.write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),phase=phase,**values),indent=2)+'\n')
    save('waiting_for_model_completion')
    while not (run/'COMPLETE.json').exists():
        if (run/'FAILURE.json').exists():
            save('model_failure_requires_review');raise RuntimeError('Model run failed; see FAILURE.json')
        time.sleep(5)
    for name in ['analyze_selection.py','report_selection.py']:
        save('running',script=name)
        result=subprocess.run([sys.executable,str(directory/name),'--run',str(run)],capture_output=True,text=True)
        (run/(name+'.log')).write_text(result.stdout+'\n'+result.stderr)
        print(result.stdout,flush=True)
        if result.returncode:
            save('failed',script=name,returncode=result.returncode)
            print(result.stderr,flush=True);raise SystemExit(result.returncode)
    save('complete',review_pending=True)

if __name__=='__main__':main()
