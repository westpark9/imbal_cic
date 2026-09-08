#!/usr/bin/env bash
# Rebuild the BoostPFN port that tabpfn/scripts/nfv3_v3_exp35_boostpfn.py imports.
#
#   bash tabpfn/third_party/boostpfn_port/setup_boostpfn_port.sh [TARGET_DIR]
#   default TARGET_DIR = <imbalcic>/tabpfn/third_party/BoostPFN   (exp35's default --boostpfn-root)
#   Normally unnecessary: that directory is vendored in the repo (source tracked; only the
#   checkpoint is missing -> ../fetch_checkpoints.sh).  Use this to rebuild from upstream.
#
# Steps (all deterministic, no hand edits):
#   1. git clone https://github.com/yxzwang/BoostPFN at commit c957ac2 (AISTATS 2025 release)
#   2. apply boostpfn_tracked.patch   -- import rewrites tabpfn.* -> tabpfn_v1.*, sklearn>=1.6 /
#                                       torch>=2.4 / numpy 2 compat in the v1 interface and
#                                       tabular_baselines, dead-URL download guard,
#                                       gb_losses_compat fallback import (see IMBALCIC_PORT_NOTES.md)
#   3. copy gb_losses_compat.py (sklearn 0.24.2 MultinomialDeviance transcription) and the notes
#   4. vendor TabPFN v1: pip download tabpfn==0.1.9 (pure-python wheel, 156 KB), extract its
#      tabpfn/ package as tabpfn_v1/, drop tests/datasets/notebooks, apply tabpfn_v1_vendored.patch
#      (import rewrites + the same compat patches)
#   5. fetch the official TabPFN v1.0.0 checkpoint (103,350,223 B) from the PriorLabs tag and
#      store it under the name the v1 loader probes first (…_epoch_100.cpkt); verify sha256
# Requirements already in the project env: torch, scikit-learn, scipy, joblib, matplotlib, pandas.
# To also run the UPSTREAM benchmark driver (largedataset_boostpfn.py) install: openml catboost hyperopt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMBALCIC="$(cd "$HERE/../../.." && pwd)"
TARGET="${1:-$IMBALCIC/tabpfn/third_party/BoostPFN}"
UPSTREAM_COMMIT=c957ac2
CKPT_URL=https://github.com/PriorLabs/TabPFN/raw/v1.0.0/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt
CKPT_SHA="$(cat "$HERE/checkpoint.sha256")"

if [ -e "$TARGET" ]; then
  echo "refusing to overwrite existing $TARGET (remove it or pass another TARGET_DIR)"; exit 1
fi
mkdir -p "$(dirname "$TARGET")"
git clone -q https://github.com/yxzwang/BoostPFN.git "$TARGET"
git -C "$TARGET" checkout -q "$UPSTREAM_COMMIT"
git -C "$TARGET" apply --whitespace=nowarn "$HERE/boostpfn_tracked.patch"
cp "$HERE/gb_losses_compat.py" "$HERE/IMBALCIC_PORT_NOTES.md" "$TARGET/"

WORK="$(mktemp -d)"
python -m pip download -q tabpfn==0.1.9 --no-deps -d "$WORK"
python - "$WORK" "$TARGET" <<'EOF'
import glob, os, shutil, sys, zipfile
work, target = sys.argv[1], sys.argv[2]
whl = glob.glob(os.path.join(work, "tabpfn-0.1.9-*.whl"))[0]
with zipfile.ZipFile(whl) as z:
    z.extractall(os.path.join(work, "x"))
src = os.path.join(work, "x", "tabpfn")
for d in ("tests", "datasets"):
    shutil.rmtree(os.path.join(src, d), ignore_errors=True)
for nb in glob.glob(os.path.join(src, "*.ipynb")):
    os.remove(nb)
shutil.copytree(src, os.path.join(target, "tabpfn_v1"))
EOF
(cd "$TARGET" && patch -p0 -s < "$HERE/tabpfn_v1_vendored.patch")

mkdir -p "$TARGET/models_diff"
curl -sSL --retry 3 -o "$TARGET/models_diff/prior_diff_real_checkpoint_n_0_epoch_100.cpkt" "$CKPT_URL"
echo "$CKPT_SHA  $TARGET/models_diff/prior_diff_real_checkpoint_n_0_epoch_100.cpkt" | sha256sum -c -
rm -rf "$WORK"

echo "port ready at $TARGET"
(cd "$TARGET" && python -c "
from scripts.transformer_prediction_interface import TabPFNClassifier
from gradient_boost_tabpfn import SamplingGradientboost
from boost_tabpfn import SamplingAdaboost
from utils import splitting_predict_proba
print('imports OK:', SamplingGradientboost.__module__, TabPFNClassifier.__module__)")
