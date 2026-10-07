"""I/O and measured resource costs for the cleaned-split SOTA comparison."""
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def context_ids(root, dataset):
    import numpy as np
    config=Path(root)/'tabpfn/configs/exp61'
    ref=json.loads((config/'reference.json').read_text())['datasets'][dataset]
    path=config/ref['file']
    if sha(path)!=ref['gzip_sha256']:raise ValueError('Context archive changed')
    content=gzip.decompress(path.read_bytes())
    if hashlib.sha256(content).hexdigest()!=ref['npy_sha256']:raise ValueError('Context IDs changed')
    ids=np.load(io.BytesIO(content),allow_pickle=False)
    if len(ids)!=ref['rows'] or len(np.unique(ids))!=len(ids):raise ValueError('Invalid context IDs')
    return ids


class Cost:
    """Inclusive phase wall times, process-tree RSS, and sampled per-process VRAM.

    Validation is nested inside fit; never add it to fit a second time. GPU peak
    sampled by nvidia-smi includes XGBoost allocations outside PyTorch.
    """
    def __init__(self, out):
        import psutil
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.proc=psutil.Process();self.started=time.perf_counter();self.stage='starting'
        self.phases=[];self.rss_peak=0;self.gpu_peak=0;self.gpu_samples=0
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()

    def monitor(self):
        import psutil
        with (self.out/'resource_samples.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['elapsed_seconds','phase','rss_bytes','gpu_used_mib'])
            while not self.stop.is_set():
                try:
                    processes=[self.proc,*self.proc.children(recursive=True)]
                    pids={p.pid for p in processes};rss=sum(p.memory_info().rss for p in processes if p.is_running())
                    self.rss_peak=max(self.rss_peak,rss);gpu=None
                    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                    if r.returncode==0:
                        values=[]
                        for line in r.stdout.splitlines():
                            fields=[v.strip() for v in line.split(',')]
                            if len(fields)==2 and fields[0].isdigit() and int(fields[0]) in pids and fields[1].isdigit():values.append(int(fields[1]))
                        gpu=sum(values);self.gpu_peak=max(self.gpu_peak,gpu);self.gpu_samples+=1
                    writer.writerow([round(time.perf_counter()-self.started,3),self.stage,rss,gpu]);f.flush()
                except (OSError,psutil.Error,subprocess.TimeoutExpired):pass
                self.stop.wait(1)

    @contextmanager
    def phase(self, name):
        import torch
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        previous=self.stage;self.stage=name;t=time.perf_counter()
        write(self.out/'progress.json',dict(state='running',phase=name,pid=os.getpid(),updated_epoch=time.time()))
        print(f'[{name}] start',flush=True)
        ok=False
        try:
            yield
            if torch.cuda.is_initialized():torch.cuda.synchronize()
            ok=True
        finally:
            elapsed=time.perf_counter()-t
            self.phases.append(dict(phase=name,seconds=elapsed,complete=ok))
            self.stage=previous
            print(f'[{name}] {elapsed:.2f}s complete={ok}',flush=True)

    def seconds(self, name):return sum(x['seconds'] for x in self.phases if x['phase']==name)

    def finish(self):
        import psutil,torch
        self.stop.set();self.thread.join(timeout=5)
        self.rss_peak=max(self.rss_peak,psutil.Process().memory_info().rss)
        # Linux ru_maxrss includes short-lived peaks between samples.
        import resource
        self.rss_peak=max(self.rss_peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        value=dict(wall_seconds=time.perf_counter()-self.started,phases=self.phases,
            peak_rss_gib=self.rss_peak/2**30,peak_gpu_sampled_gib=self.gpu_peak/1024 if self.gpu_samples else None,
            peak_torch_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_initialized() else 0,
            peak_torch_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_initialized() else 0,
            gpu_sampling_seconds=1,gpu_samples=self.gpu_samples)
        write(self.out/'cost.json',value)
        return value
