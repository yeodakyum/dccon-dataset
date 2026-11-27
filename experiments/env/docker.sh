#!/usr/bin/env bash
# Build and run the experiment images on the GPU server (rootless Docker).
#
#   bash experiments/env/docker.sh build [main vllm pylate]
#   bash experiments/env/docker.sh run main python3 experiments/scripts/smoke_test.py main
#   GPU=2 bash experiments/env/docker.sh run vllm python3 ...
#   OFFLINE=1 bash experiments/env/docker.sh run main python3 ...   # no Hub access
#
# GPUs are attached through CDI (nvidia.com/gpu=gpuN from make_cdi_spec.sh) because
# GPU 0 is faulty and breaks `--gpus`. GPU must be 1, 2, or 3; the default is 1.
# Models live on NFS (/mnt/main/nsu102/hf -> /hf) because / is shared and 92% full.
set -euo pipefail

export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/study-signal-docker.sock}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/experiments/env"
HF_CACHE="${HF_CACHE:-/mnt/main/nsu102/hf}"
GPU="${GPU:-1}"

usage() { echo "usage: $0 build [main|vllm|pylate ...] | run <main|vllm|pylate> <command...>" >&2; exit 2; }

case "${1:-}" in
  build)
    shift
    targets=("$@")
    [[ ${#targets[@]} -eq 0 ]] && targets=(main vllm pylate)
    for name in "${targets[@]}"; do
      docker build -f "$HERE/docker/Dockerfile.$name" -t "dccon-$name" "$HERE"
    done
    ;;
  run)
    shift
    [[ $# -ge 2 ]] || usage
    name="$1"; shift
    mkdir -p "$HF_CACHE" "$ROOT/runs"
    tty=(); [[ -t 0 && -t 1 ]] && tty=(-it)
    docker run --rm "${tty[@]}" \
      --device "nvidia.com/gpu=gpu$GPU" \
      --shm-size=16g \
      -e HF_HUB_OFFLINE="${OFFLINE:-0}" \
      -v "$ROOT:/work" -v "$HF_CACHE:/hf" -w /work \
      "dccon-$name" "$@"
    ;;
  *)
    usage
    ;;
esac
