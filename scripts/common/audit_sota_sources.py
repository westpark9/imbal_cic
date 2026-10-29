#!/usr/bin/env python3
"""Record pinned source evidence and small CPU diagnostics; never run a model."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

from experiment_paths import repo_root
from sota_sources import inspect


SOURCES = {
    "TabDPT": {
        "pyproject.toml": ["version =", "requires-python", "faiss-cpu", "torch>="],
        "src/tabdpt/estimator.py": ["_VERSION =", "_MODEL_NAME", "context_reduction:", "self.max_features"],
        "src/tabdpt/classifier.py": ["context_size:", "batch_size:", "128*1024", "def _predict_large_cls"],
    },
    "LoCalPFN": {
        "config.py": ["--model_path", "epoch_42", "--filter_feature", "--filter_class", "--context_length", "--num_epochs", "--num_steps"],
        "methods/ftknn.py": ["num_retrievals =", "10 * np.sqrt", "num_classes =", "args.train_query_length"],
        "utils.py": ["assert num_classes <=", "def pad_x", "class SingleclassFaiss"],
    },
    "BoostPFN": {
        "scripts/transformer_prediction_interface.py": ["from tabpfn", "url =", "max_num_features =", "X.shape[0] > 1024"],
        "gradient_boost_tabpfn.py": ["_gb_losses", "max_samples=", "def fit(", "sample_num=", "version=1"],
        "main_10times.py": ["gboost_tabpfnV2", "version=2"],
        "ReadMe.md": ["5000", "maxsample", "ensemble_num"],
    },
    "BETA": {
        "configs/deep_configs.json": ["max_epoch", "batch_size", "evaluate_option"],
        "configs/default/Beta.json": ["arch_type", '"k"', "d_block", "weight_decay"],
        "model/models/Beta.py": ["load_model_workflow(0, 42", "requires_grad = False", "candidate_indices =", "min(1024", "min(1000", "max_num_classes"],
        "model/methods/Beta.py": ["best_epoch'] ==-1", "self.PFN_predict", "batch_size = 4096", "test_logit.mean", "pre_validate"],
        "model/lib/tabpfn/utils.py": ["url ="],
    },
    "DistPFN": {
        "README.md": ["P_test_avg", "y_prior_train", "tau =", "pip install tabpfn"],
        "src/evals/eval_TABPFN_shift_O.py": ["def cross_entropy", "axis=1", "max_samples=4000", "n_estimators=", "P_test_avg", "cross_entropy = log_loss", "max(values"],
        "src/__init__.py": ["BASE_DIR ="],
        "requirements.txt": ["python", "torch", "scikit-learn", "numpy"],
    },
    "CURE": {
        "requirements.txt": ["torch", "tabicl", "numpy", "scikit-learn"],
        "src/stream/model/tabicl_stream_wrapper.py": ["checkpoint_version:", "n_estimators", "batch_size:", "TabICLClassifier"],
        "src/stream/experiment/tabicl_cure_runner.py": ["budget_B:", "short_ratio:", "warmup_steps:", "model.fit(", "model.predict(", "append_then_evict"],
        "scripts/run_tabicl_cure.py": ["poker", "--tabicl-model-path"],
    },
}


def main() -> None:
    root = repo_root()
    lock_path = root / "tabpfn/third_party/upstream_sources.lock.json"
    lock = json.loads(lock_path.read_text())
    result = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Source inspection and isolated CPU diagnostic; no model fitting or GPU benchmark",
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "repositories": [],
    }
    for entry in lock["repositories"]:
        status = inspect(root, entry)
        if status["status"] != "ok":
            raise RuntimeError(status)
        base = root / entry["path"]
        evidence = []
        for relative, markers in SOURCES[entry["name"]].items():
            p = base / relative
            lines = p.read_text().splitlines()
            evidence.append({
                "path": relative,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "url": entry["url"].removesuffix(".git") + "/blob/" + entry["commit"] + "/" + relative,
                "matching_lines": [{"line": i, "text": line} for i, line in enumerate(lines, 1)
                                   if any(marker in line for marker in markers)],
            })
        filenames = subprocess.check_output(["git", "-C", str(base), "ls-files", "-z"], text=True).split("\0")
        parsed, failures, large_files = 0, [], []
        for rel in filter(None, filenames):
            p = base / rel
            if p.suffix == ".py":
                try:
                    ast.parse(p.read_bytes(), filename=rel)
                    parsed += 1
                except SyntaxError as exc:
                    failures.append({"path": rel, "line": exc.lineno, "message": exc.msg})
            if p.is_file() and p.stat().st_size > 100_000_000:
                large_files.append({"path": rel, "bytes": p.stat().st_size,
                                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        result["repositories"].append({**status, "evidence": evidence,
                                        "python_syntax_files_passed": parsed,
                                        "python_syntax_failures": failures,
                                        "files_above_100MB": large_files})

    # Execute only the self-contained array function extracted from the source AST.
    # The benchmark module itself has import-time data loading and must not run here.
    import numpy as np
    dist = root / "tabpfn/third_party/upstream/DistPFN/src/evals/eval_TABPFN_shift_O.py"
    tree = ast.parse(dist.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cross_entropy")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(dist), "exec"), namespace)
    try:
        namespace["cross_entropy"](np.array([.7, .3]), np.array([.9, .1]))
        result["distpfn_t_vector_probe"] = {"status": "unexpectedly passed"}
    except (ValueError, IndexError) as exc:
        result["distpfn_t_vector_probe"] = {"status": "upstream shape error reproduced",
                                             "error": type(exc).__name__, "message": str(exc)}
    try:
        np.bincount(np.array([0., 1., 0.], dtype=np.float32))
    except TypeError as exc:
        result["distpfn_prior_float_probe"] = {"status": "integer-label requirement confirmed",
                                                "error": type(exc).__name__, "message": str(exc)}
    result["parent_tracked_clone_files"] = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "tabpfn/third_party/upstream"], text=True).splitlines()
    result["existing_vendored_diff"] = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", "--", "tabpfn/third_party/BoostPFN",
         "tabpfn/third_party/LoCalPFN", "tabpfn/third_party/DistPFN", "tabpfn/third_party/boostpfn_port"],
        text=True).splitlines()
    output = root / "docs/lablog/evidence/20261029/sota_source_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output)
    for r in result["repositories"]:
        print(r["name"], r["status"], "syntax", r["python_syntax_files_passed"], "failures", len(r["python_syntax_failures"]))
    print(json.dumps(result["distpfn_t_vector_probe"], ensure_ascii=False))


if __name__ == "__main__":
    main()
