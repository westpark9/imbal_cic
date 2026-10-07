import subprocess,sys,json,datetime
from pathlib import Path
out=Path('/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp70_class_arrival_100k_s43')
commands=[['/home/user/miniconda3/bin/python', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp70_class_arrival_100k_s43/source/scripts/exp69_class_arrival.py', 'prepare', '--out', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp70_class_arrival_100k_s43', '--data', '/home/user/Desktop/imbalcic/data/nfv3_energy_suite_uncapped_scenarios.pkl', '--clean-root', '/home/user/Desktop/imbalcic/data/derived', '--budgets', '100000', '--allocation', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp70_class_arrival_100k_s43/allocation.json', '--extend-from', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp69_class_arrival_s43'], ['/home/user/miniconda3/bin/python', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp70_class_arrival_100k_s43/source/scripts/exp69_class_arrival.py', 'run', '--out', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp70_class_arrival_100k_s43', '--checkpoint', '/home/user/Desktop/imbalcic/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt', '--budgets', '100000', '--seed', '43']]
for cmd in commands:
 rc=subprocess.call(cmd)
 if rc:
  status=json.loads((out/"status.json").read_text()) if (out/"status.json").exists() else {}
  status.update(state="failed" if rc!=75 else "paused",returncode=rc,failed_command=cmd,updated=datetime.datetime.now(datetime.timezone.utc).isoformat())
  (out/"status.json").write_text(json.dumps(status,indent=2)+"\n")
  sys.exit(rc)
