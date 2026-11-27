#!/usr/bin/env bash
# Copy gated models downloaded on the Mac into the server's HF cache (NFS).
#   bash experiments/scripts/push_hf_cache.sh                      # gated models from configs
#   bash experiments/scripts/push_hf_cache.sh org/name [org/name]  # specific repos
set -euo pipefail

HOST="${DCCON_HOST:-snu-gpu}"
REMOTE_HUB="${DCCON_REMOTE_HUB:-/mnt/main/nsu102/hf/hub}"
LOCAL_HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
repos=("$@")
[[ ${#repos[@]} -eq 0 ]] && repos=(google/embeddinggemma-300m naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)

ssh "$HOST" "mkdir -p '$REMOTE_HUB'"
for repo in "${repos[@]}"; do
  dir="models--${repo//\//--}"
  [[ -d "$LOCAL_HUB/$dir" ]] || { echo "not downloaded locally: $repo" >&2; exit 1; }
  echo "== $repo"
  # -a keeps the cache's relative snapshot -> blob symlinks.
  rsync -a --stats "$LOCAL_HUB/$dir" "$HOST:$REMOTE_HUB/" | grep -E "Total transferred|total size"
done
