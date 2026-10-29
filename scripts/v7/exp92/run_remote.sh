#!/usr/bin/env bash
# Reproduce the actual CIC2018 pilot after git pull, including PCAP + checkpoint.
set -euo pipefail
PCAP_REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$PCAP_REPO_ROOT"
PCAP_BASE_PYTHON="${PCAP_BASE_PYTHON:-python}"
"$PCAP_BASE_PYTHON" -c 'import sys,torch; print(sys.executable,torch.__version__); assert torch.cuda.is_available(), "Use a driver-compatible Python via PCAP_BASE_PYTHON (e.g. your exp91-cu124/bin/python). The existing environment will not be changed."'
PCAP_ENV="$PCAP_REPO_ROOT/.envs/pcap-netfound"
if [ ! -x "$PCAP_ENV/bin/python" ]; then
  "$PCAP_BASE_PYTHON" -m venv --system-site-packages "$PCAP_ENV"
fi
PCAP_PYTHON="$PCAP_ENV/bin/python"
"$PCAP_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "The existing .envs/pcap-netfound environment cannot use CUDA. Recreate it from the driver-compatible PCAP_BASE_PYTHON before retrying."'
"$PCAP_PYTHON" -m pip install -r configs/environments/exp92_netfound.txt
export PYTHONUNBUFFERED=1
export USE_TF=0
"$PCAP_PYTHON" scripts/v7/exp92/fetch_assets.py encoder
"$PCAP_PYTHON" scripts/v7/exp92/fetch_assets.py cic2018-pilot
"$PCAP_PYTHON" scripts/results_versions.py run \
  --version v7 --experiment exp92 --scope tabpfn --label cic2018_verified_pcap \
  --protocol configs/experiments/exp92_protocol.json -- \
  "$PCAP_PYTHON" scripts/v7/exp92/run_pipeline.py \
  --config configs/experiments/exp92_protocol.json --output '{run_dir}' --batch-size 16
