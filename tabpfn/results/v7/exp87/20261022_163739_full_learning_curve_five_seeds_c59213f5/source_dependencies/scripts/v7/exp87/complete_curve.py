"""Complete the EXP87 learning curve at the same five seeds at every budget."""
from pathlib import Path
import argparse, copy, json, shutil, subprocess, sys, traceback
from attack_saturation import ROOT, SEEDS, METHODS, read, stamp, progress, run_job
from exp61_utils import write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    args = parser.parse_args()
    out, source = args.out.resolve(), args.source.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = read(source / 'results.json')
    assert original['state'] == 'complete'
    cfg = copy.deepcopy(read(source / 'config.json'))
    cfg.update(protocol='exp87.2', started_at=stamp(), source_run=str(source),
               purpose='Complete all curve points at seeds43–47; same data, models and attack-only queries')
    cfg['design']['validation'] = 'All sample budgets at the identical five seeds, following the completed pilot'
    if (out / 'config.json').exists():
        cfg = read(out / 'config.json')
    write(out / 'config.json', cfg)
    if not (out / 'data').exists():
        (out / 'data').symlink_to(source / 'data', target_is_directory=True)
    for name in ['checkpoint.json', 'environment.json', 'prepared.json']:
        shutil.copy2(source / name, out / name)
    for file in [Path(__file__), Path(__file__).with_name('attack_saturation.py'),
                 Path(__file__).with_name('report_saturation.py')]:
        dest = out / 'source_dependencies' / file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
    write(out / 'source_run.json', dict(path=str(source), results_sha256=sha(source / 'results.json'),
                                      reused_jobs=len(original['jobs']), read_only=True))
    if (out / 'results.json').exists():
        saved = read(out / 'results.json')
        jobs, events = saved['jobs'], saved['events']
    else:
        jobs, events = copy.deepcopy(original['jobs']), []
    decisions = copy.deepcopy(original['decisions'])
    expected = 2 * len(SEEDS) * sum(map(len, cfg['design']['budgets'].values()))
    state = 'running'

    def save(stage):
        write(out / 'results.json', dict(state=state, stage=stage, jobs=jobs, decisions=decisions,
              expected_jobs=expected, events=events, updated_at=stamp(), source_run=str(source)))

    def report():
        p = subprocess.run([sys.executable, str(Path(__file__).with_name('report_saturation.py')),
                            '--run', str(out)], capture_output=True, text=True)
        if p.returncode:
            write(out / 'REPORT_FAILURE.json', dict(at=stamp(), stdout=p.stdout[-3000:], stderr=p.stderr[-7000:]))
            print(p.stderr[-7000:], flush=True)
            return False
        (out / 'REPORT_FAILURE.json').unlink(missing_ok=True)
        print(p.stdout, flush=True)
        return True

    try:
        import torch
        torch.set_num_threads(cfg['threads'])
        save('completing_five_seed_curve')
        report()
        for level in range(max(map(len, cfg['design']['budgets'].values()))):
            for ds in cfg['design']['datasets']:
                grid = cfg['design']['budgets'][ds]
                if level >= len(grid):
                    continue
                n = grid[level]
                changed = False
                for seed in SEEDS:
                    for method in METHODS:
                        if any((j['dataset'], j['seed'], j['n'], j['method']) == (ds, seed, n, method) for j in jobs):
                            continue
                        jobs.append(run_job(out, cfg, ds, seed, n, method))
                        changed = True
                        save('completing_five_seed_curve')
                if changed:
                    events.append(dict(at=stamp(), dataset=ds, n=n, complete_seeds=SEEDS))
                    save('completing_five_seed_curve')
                    report()
        assert len(jobs) == expected
        state = 'complete'
        save('complete')
        progress(out, 'complete', completed_jobs=len(jobs), new_jobs=len(jobs)-len(original['jobs']))
    except BaseException as exc:
        state = 'failed'
        write(out / 'FAILURE.json', dict(at=stamp(), error=str(exc), traceback=traceback.format_exc()))
        save('failed')
        raise
    finally:
        if not report() and state == 'complete':
            raise RuntimeError('Experiments completed; report verification failed, see REPORT_FAILURE.json')


if __name__ == '__main__':
    main()
