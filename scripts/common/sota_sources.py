#!/usr/bin/env python3
"""Restore or check pinned, independent SOTA source repositories (stdlib only).

Existing clones are never pulled, reset, cleaned or overwritten. A changed
checkout must be reviewed explicitly before updating upstream_sources.lock.json.
This does not install packages, download runtime model weights, or run models.
Upstream Git-tracked data/weights are part of a clone and remain parent-ignored.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from experiment_paths import repo_root


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def inspect(root: Path, entry: dict) -> dict:
    path = root / entry["path"]
    result = {"name": entry["name"], "path": entry["path"], "expected": entry["commit"]}
    if not path.exists():
        return {**result, "status": "missing"}
    if not (path / ".git").is_dir():
        return {**result, "status": "refused: not an independent Git clone"}
    if Path(git(path, "rev-parse", "--show-toplevel")).resolve() != path.resolve():
        return {**result, "status": "refused: different Git root"}
    result["actual"] = git(path, "rev-parse", "HEAD")
    result["origin"] = git(path, "remote", "get-url", "origin")
    dirty = git(path, "status", "--porcelain", "--untracked-files=normal")
    if result["origin"] != entry["url"]:
        result["status"] = "refused: origin differs"
    elif dirty:
        result["status"] = "refused: local changes (preserved)"
        result["changes"] = dirty
    elif result["actual"] != entry["commit"]:
        result["status"] = "refused: commit differs (preserved)"
    else:
        result["status"] = "ok"
    return result


def restore(root: Path, entry: dict) -> dict:
    current = inspect(root, entry)
    if current["status"] != "missing":
        return current
    path = root / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Download to an ignored staging directory; publish only a complete clone.
    staging = Path(tempfile.mkdtemp(prefix=f".{entry['name']}-", dir=path.parent))
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--no-checkout", "--depth", "1", "--filter=blob:none",
             entry["url"], str(staging)], check=True, env=env,
        )
        has_commit = subprocess.run(
            ["git", "-C", str(staging), "cat-file", "-e", entry["commit"] + "^{commit}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if not has_commit:
            subprocess.run(
                ["git", "-C", str(staging), "fetch", "--quiet", "--depth", "1", "origin", entry["commit"]],
                check=True, env=env,
            )
        subprocess.run(
            ["git", "-C", str(staging), "checkout", "--quiet", "--detach", entry["commit"]],
            check=True, env=env,
        )
        staging.rename(path)
    except Exception:
        # Preserve an interrupted download for inspection; never remove user files.
        print(f"Incomplete clone retained at {staging}", file=sys.stderr)
        raise
    return inspect(root, entry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "check"])
    parser.add_argument("--models", nargs="+", help="Optional subset, e.g. TabDPT BETA CURE")
    args = parser.parse_args()
    root = repo_root()
    lock = root / "tabpfn/third_party/upstream_sources.lock.json"
    manifest = json.loads(lock.read_text())
    entries = manifest["repositories"]
    if args.models:
        unknown = set(args.models) - {e["name"] for e in entries}
        if unknown:
            parser.error("Unknown model(s): " + ", ".join(sorted(unknown)))
        entries = [e for e in entries if e["name"] in args.models]
    results = []
    for entry in entries:
        target = root / entry["path"]
        if (not re.fullmatch(r"[0-9a-f]{40}", entry["commit"])
                or target.resolve().parent != (root / "tabpfn/third_party/upstream").resolve()):
            raise ValueError("Invalid commit or clone destination in lock file")
        ignored = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", entry["path"] + "/.git/HEAD"],
        ).returncode == 0
        if not ignored:
            raise RuntimeError(f"Parent .gitignore does not protect {entry['path']}")
        try:
            result = restore(root, entry) if args.action == "fetch" else inspect(root, entry)
        except (OSError, subprocess.CalledProcessError) as exc:
            result = {"name": entry["name"], "status": "error", "error": str(exc)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
