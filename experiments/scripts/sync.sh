#!/usr/bin/env bash
# Sync code between this Mac and the GPU server.
#   bash experiments/scripts/sync.sh push   # Mac -> server (code, configs, pilot, generated text data)
#   bash experiments/scripts/sync.sh pull   # server -> Mac (runs/ results only)
set -euo pipefail

HOST="${DCCON_HOST:-snu-gpu}"
REMOTE="${DCCON_REMOTE:-project/dccon-dataset}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

case "${1:-}" in
  push)
    # Images stay on the Mac; experiments only need text. data/week manifests are
    # small and fetched on the server with download_snapshot.py --manifests-only.
    rsync -az --stats \
      --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude .DS_Store \
      --exclude .env --exclude 'data/week/images/' --exclude 'data/week/raw/' \
      --exclude runs/ --exclude 'docs/' --exclude '*.log' \
      "$ROOT/" "$HOST:$REMOTE/"
    ;;
  pull)
    rsync -az --stats "$HOST:$REMOTE/runs/" "$ROOT/runs/"
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 2
    ;;
esac
