import subprocess,sys
commands=[['/home/user/miniconda3/bin/python', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp69_class_arrival_s43/source/scripts/exp69_class_arrival.py', 'prepare', '--out', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp69_class_arrival_s43', '--data', '/home/user/Desktop/imbalcic/data/nfv3_energy_suite_uncapped_scenarios.pkl', '--clean-root', '/home/user/Desktop/imbalcic/data/derived'], ['/home/user/miniconda3/bin/python', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp69_class_arrival_s43/source/scripts/exp69_class_arrival.py', 'run', '--out', '/home/user/Desktop/imbalcic/tabpfn/results/20261002_exp69_class_arrival_s43', '--checkpoint', '/home/user/Desktop/imbalcic/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt']]
for cmd in commands:
 rc=subprocess.call(cmd)
 if rc: sys.exit(rc)
