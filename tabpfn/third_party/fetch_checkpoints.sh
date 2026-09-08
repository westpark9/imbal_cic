#!/usr/bin/env bash
# Download the TabPFN v1.0.0 checkpoint (103,350,223 B) that the vendored BoostPFN and LoCalPFN
# code loads.  Weights are not tracked in git (*.cpkt is ignored).  Idempotent.
#
#   bash tabpfn/third_party/fetch_checkpoints.sh
#
# BoostPFN's v1 loader probes epoch 100 first, so the file is stored under the epoch_100 name
# there; LoCalPFN's config.py points at the epoch_42 name.  Same bytes (sha256 below).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL=https://github.com/PriorLabs/TabPFN/raw/v1.0.0/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt
SHA="$(cat "$HERE/boostpfn_port/checkpoint.sha256")"
TMP="$(mktemp)"
for dest in "$HERE/BoostPFN/models_diff/prior_diff_real_checkpoint_n_0_epoch_100.cpkt" \
            "$HERE/LoCalPFN/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"; do
  if [ -f "$dest" ] && echo "$SHA  $dest" | sha256sum -c --quiet - 2>/dev/null; then
    echo "ok (present): $dest"; continue
  fi
  if [ ! -s "$TMP" ]; then curl -sSL --retry 3 -o "$TMP" "$URL"; echo "$SHA  $TMP" | sha256sum -c --quiet -; fi
  mkdir -p "$(dirname "$dest")"; cp "$TMP" "$dest"; echo "fetched: $dest"
done
rm -f "$TMP"
