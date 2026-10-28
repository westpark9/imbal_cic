from pathlib import Path
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

ROOT = Path("/home/user/Desktop/imbalcic")
RUN = ROOT / "tabpfn/results/v7/exp84/20261020_160101_small_labels_5_100_buffer_retention_s43_3204a372"
sys.path.insert(0, str(ROOT / "scripts/common"))
from lablog import build_report


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temp = path.with_suffix(path.suffix + ".resume.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def note_resume():
    report = ROOT / "docs/lablog/reports/20261020_exp84.tex"
    text = report.read_text()
    note = (
        r"\section{실행 중단과 재개 기록}" "\n"
        "2026-10-20 16:16:54 이후 진행 기록이 멈췄고, 16:30:12에 서버가 재시작되었다. "
        "재개 시 실행 프로세스가 없음을 확인했다. 정확한 종료 원인은 로그에 남지 않았다. "
        "이미 완료한 12조건은 예측 파일 해시와 완료 표시를 유지하고, 중간에 멈춘 조건부터 다시 계산한다. "
        "재개 전 소스·checkpoint 해시와 라이브러리 버전이 원 실행과 같음을 확인했다. "
        "이 실험의 비용은 동일 서버의 재시작 전후 세션에서 측정되므로 작은 시간 차이는 신중히 해석한다. "
        r"재개 근거는 실행 폴더의 \path{resumptions/20261020_restart/}에 보존한다." "\n"
    )
    assert r"\section{실행 중단과 재개 기록}" not in text
    report.write_text(text.replace(r"\end{document}", note + r"\end{document}"))
    info = build_report(report)
    vp = ROOT / "docs/lablog/_support/validation.json"
    val = read(vp)
    val["reports"] = [info if x["report"] == info["report"] else x for x in val["reports"]]
    write(vp, val)
    sp = ROOT / "docs/lablog/_support/sources.json"
    sources = read(sp)
    for value in sources.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("path") == str(report.relative_to(ROOT)):
                    item["sha256"] = digest(report)
                    item["last_compiled_pdf_sha256"] = info["compiled_pdf_sha256"]
    write(sp, sources)


with (RUN / ".resume.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert all(digest(Path(k)) == v for k, v in read(RUN / "source_hashes.json").items())
    assert all(importlib.metadata.version(k) == v for k, v in read(RUN / "environment.json").items())
    completed = list(RUN.glob("new_*/jobs/*/COMPLETE.json"))
    assert len(completed) == 12
    completed_hashes = {str(p.parent / "predictions.npy"): digest(p.parent / "predictions.npy") for p in completed}
    evidence = RUN / "resumptions/20261020_restart"
    evidence.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, evidence / "resume_manager.py")
    for name in ["RUN_META.json", "results.json", "progress.json", "config.json", "environment.json", "warmup.json", "source_hashes.json"]:
        shutil.copy2(RUN / name, evidence / name)
    for job in RUN.glob("new_*/jobs/*"):
        if job.is_dir() and not (job / "COMPLETE.json").exists():
            target = evidence / "interrupted_jobs" / job.relative_to(RUN)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(job), target)
    resume = dict(resumed_at=stamp(),manager_pid=os.getpid(),server_boot="2026-10-20T16:30:12+09:00",
                  completed_jobs_preserved=12,source_hashes_unchanged=True,environment_versions_unchanged=True,
                  completed_prediction_sha256=completed_hashes,status="running")
    write(evidence / "resume.json", resume)
    meta = read(RUN / "RUN_META.json")
    meta["resumptions"] = [str((evidence / "resume.json").relative_to(ROOT))]
    meta["status"] = "running"
    write(RUN / "RUN_META.json", meta)
    rc = 1
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts/v7/exp84/build_small_label_record.py"), "--run", str(RUN)], cwd=ROOT, check=True)
        note_resume()
        subprocess.run([sys.executable, str(ROOT / "scripts/results_versions.py"), "index"], cwd=ROOT, check=True)
        with (RUN / "stdout.log").open("a") as output:
            output.write("\nRESUME " + stamp() + ": preserving 12 completed conditions after server restart\n")
            output.flush()
            proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/v7/exp84/small_label_updates.py"), "--out", str(RUN)],
                                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, EXPERIMENT_RUN_DIR=str(RUN)))
            resume["worker_pid"] = proc.pid
            write(evidence / "resume.json", resume)
            rc = proc.wait()
        assert all(digest(Path(k)) == v for k, v in completed_hashes.items())
        note_resume()
    except BaseException as exc:
        rc = rc or 1
        resume["error"] = repr(exc)
        raise
    finally:
        resume.update(finished_at=stamp(), returncode=rc, status="finished" if rc == 0 else "failed")
        write(evidence / "resume.json", resume)
        meta.update(finished=stamp(), returncode=rc, status=resume["status"])
        write(RUN / "RUN_META.json", meta)
        subprocess.run([sys.executable, str(ROOT / "scripts/results_versions.py"), "index"], cwd=ROOT)
