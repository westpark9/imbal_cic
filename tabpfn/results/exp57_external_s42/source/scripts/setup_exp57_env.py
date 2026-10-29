#!/usr/bin/env python3
"""Create/reuse an EXP57 Conda prefix on persistent storage, with pinned CUDA PyTorch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements-exp57.txt"
# Official wheels; choose conservatively without relying on CUDA minor compatibility.
# https://pytorch.org/get-started/previous-versions/
PROFILES = [
    ((12, 8), "2.7.1", "cu128"),
    ((12, 4), "2.6.0", "cu124"),
    ((12, 1), "2.5.1", "cu121"),
    ((11, 8), "2.6.0", "cu118"),
]

CUDA_CHECK = r'''
import importlib.metadata as metadata
import json
import sys
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

expected = sys.argv[1]
if torch.__version__ != expected:
    raise RuntimeError(f"Expected torch {expected}, found {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError(f"CUDA initialization failed: torch={torch.__version__}, "
                       f"wheel CUDA={torch.version.cuda}. Check the node driver and "
                       "CUDA_VISIBLE_DEVICES; creating a Conda env does not change the driver.")
capability = torch.cuda.get_device_capability(0)
if capability[0] < 8:
    raise RuntimeError("EXP57 requires compute capability >= 8.0")
with torch.inference_mode():
    q = torch.randn(1, 4, 32, 64, device="cuda", dtype=torch.float16)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise RuntimeError("CUDA attention check returned non-finite values")
# Detect missing runtime libraries/import conflicts before the long experiment.
import tabpfn, numpy, pandas, sklearn, scipy, xgboost, matplotlib, threadpoolctl
print(json.dumps(dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0), compute_capability=capability,
                     tabpfn=metadata.version("tabpfn"), flash_attention="passed")))
'''


def select_profile(smi_output, capability=None):
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_output)
    if not match:
        raise ValueError("Cannot read driver-supported CUDA version from nvidia-smi")
    supported = tuple(map(int, match.groups()))
    if capability is not None and capability[0] < 8:
        raise ValueError("EXP57 requires GPU compute capability >= 8.0")
    if capability is not None and capability[0] >= 10 and supported < (12, 8):
        raise ValueError("Blackwell GPUs need a CUDA 12.8-capable driver for this setup")
    for minimum, version, build in PROFILES:
        if supported >= minimum:
            return dict(torch=version, build=build, cuda=".".join(map(str, minimum)),
                        driver_cuda=".".join(map(str, supported)))
    raise ValueError("This setup needs a driver supporting CUDA >= 11.8")


def read_profile(gpu):
    smi = subprocess.check_output(["nvidia-smi", "-i", gpu], text=True)
    capability = None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        capability = tuple(map(int, output.strip().splitlines()[0].split(".")))
    except (subprocess.CalledProcessError, ValueError, IndexError):
        pass  # Older nvidia-smi may not expose this field; the CUDA check enforces it.
    return select_profile(smi, capability)


def setup(prefix, profile, gpu, dry_run=False):
    prefix = prefix.resolve()
    python = prefix / "bin/python"
    marker = prefix / "exp57_setup.json"
    expected = profile["torch"] + "+" + profile["build"]
    spec = dict(python="3.12", torch=expected,
                requirements_sha256=hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest())
    prior = json.loads(marker.read_text()) if marker.exists() else None
    if prefix.exists() and prior is None:
        raise ValueError(f"Use a NEW dedicated prefix; refusing to change unmanaged environment: {prefix}")
    if prior and any(prior["spec"][key] != spec[key] for key in ("python", "torch")):
        raise ValueError(f"Existing prefix uses a different build. Use a new --prefix ending in {profile['build']}")
    print(json.dumps(dict(prefix=str(prefix), **profile, expected_torch=expected), indent=2), flush=True)
    if dry_run:
        print("Dry run: no environment or package changes.")
        return

    env = os.environ.copy()
    # Keep downloads on the same persistent volume as the environment.
    cache = prefix.parent / ".exp57-cache"
    env.update(PIP_CACHE_DIR=str(cache / "pip"), CONDA_PKGS_DIRS=str(cache / "conda"),
               PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES=gpu,
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    cache.mkdir(parents=True, exist_ok=True)
    fresh = not python.is_file()
    if fresh:
        conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
        if not conda or not Path(conda).is_file():
            raise ValueError("Conda is needed for the FIRST setup. Install Miniforge/Conda and rerun; "
                             "later reuse needs only the saved prefix/bin/python.")
        subprocess.run([conda, "create", "--yes", "--prefix", str(prefix), "--override-channels",
                        "--channel", "conda-forge", "python=3.12", "pip"], env=env, check=True)
    if fresh or not prior or prior.get("state") != "ready" or prior["spec"] != spec:
        marker.write_text(json.dumps(dict(state="installing", spec=spec), indent=2) + "\n")
        constraints = prefix / "exp57_torch_constraints.txt"
        constraints.write_text(f"torch=={expected}\n")
        subprocess.run([str(python), "-m", "pip", "install", f"torch=={expected}",
                        "--index-url", f"https://download.pytorch.org/whl/{profile['build']}"],
                       env=env, check=True)
        # Prevent TabPFN's torch>=2.5 dependency from replacing the selected CUDA build.
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS),
                        "-c", str(constraints)], env=env, check=True)
    else:
        print("Reusing installed environment; no package installation.", flush=True)
    subprocess.run([str(python), "-m", "pip", "check"], env=env, check=True)
    checked = subprocess.check_output([str(python), "-c", CUDA_CHECK, expected], env=env, text=True)
    print(checked, end="", flush=True)
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True)
    (prefix / "exp57_packages.txt").write_text(frozen)
    marker.write_text(json.dumps(dict(state="ready", spec=spec, profile=profile,
                                     cuda_check=json.loads(checked.strip().splitlines()[-1])), indent=2) + "\n")
    print(f"Ready. Run experiments with: {shlex.quote(str(python))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True, help="NEW dedicated path on the persistent mount")
    parser.add_argument("--gpu", default="0", help="Physical GPU index or UUID, as in nvidia-smi")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected build without installing anything")
    args = parser.parse_args()
    try:
        setup(args.prefix, read_profile(args.gpu), args.gpu, args.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"EXP57 setup failed: {error}\n")


if __name__ == "__main__":
    main()
