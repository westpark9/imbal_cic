#!/usr/bin/env bash
# Reproduce EXP93 on a CUDA-compatible host after git pull.
set -euo pipefail
PCAP_REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$PCAP_REPO_ROOT"

# Do not replace the host's driver-compatible Torch installation.  The A100
# EXP91 environment used here was verified as torch 2.6.0+cu124 previously.
PCAP_BASE_PYTHON="${PCAP_BASE_PYTHON:-python}"
"$PCAP_BASE_PYTHON" -c 'import sys,torch; print(sys.executable, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available(), "Set PCAP_BASE_PYTHON to a CUDA-compatible Python"'

PCAP_ENV="$PCAP_REPO_ROOT/.envs/pcap-netfound"
if [ ! -x "$PCAP_ENV/bin/python" ]; then
  "$PCAP_BASE_PYTHON" -m venv --system-site-packages "$PCAP_ENV"
fi
PCAP_PYTHON="$PCAP_ENV/bin/python"
"$PCAP_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Existing .envs/pcap-netfound has an incompatible Torch. Remove this local venv and rerun with PCAP_BASE_PYTHON set."'
"$PCAP_PYTHON" -m pip install -r configs/environments/exp92_netfound.txt
"$PCAP_PYTHON" -c 'import tabpfn,xgboost; print("tabpfn",tabpfn.__version__,"xgboost",xgboost.__version__)'

export PYTHONUNBUFFERED=1 USE_TF=0
"$PCAP_PYTHON" scripts/v7/exp92/fetch_assets.py encoder
"$PCAP_PYTHON" scripts/v7/exp93/test_pcap_benchmark.py
"$PCAP_PYTHON" scripts/results_versions.py run \
  --version v7 --experiment exp93 --scope tabpfn --label cic2018_pcap_expanded \
  --protocol configs/experiments/exp93_protocol.json -- \
  "$PCAP_PYTHON" scripts/v7/exp93/run_pcap_benchmark.py \
    --config configs/experiments/exp93_protocol.json --output '{run_dir}' --adapter netfound-v2-small --batch-size 32
