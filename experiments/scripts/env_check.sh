#!/usr/bin/env bash
# Environment checks to run in the VS Code integrated terminal and screenshot.
# Each check prints to the terminal and saves the same output to
# experiments/results/logs/<check>.log, ending with "Elapsed seconds: N".
#
#   bash experiments/scripts/env_check.sh environment
#   bash experiments/scripts/env_check.sh smoke-main | smoke-vllm | smoke-pylate
set -uo pipefail

HOST="${DCCON_HOST:-snu-gpu}"
REMOTE="${DCCON_REMOTE:-project/dccon-dataset}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOGS="$ROOT/experiments/results/logs"
check="${1:-}"
mkdir -p "$LOGS"

environment() {
  echo "== Mac"
  echo "$(sw_vers -productName) $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string)"
  echo "venv: $("$HOME/.local/share/venvs/dccon/bin/python" --version)"
  echo "repo: $(git -C "$ROOT" rev-parse --short HEAD) ($(git -C "$ROOT" branch --show-current))"
  ssh "$HOST" "bash -s" <<'EOF'
export DOCKER_HOST=unix:///run/user/$(id -u)/study-signal-docker.sock
echo; echo "== Server: $(hostname), $(. /etc/os-release; echo $PRETTY_NAME)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo; echo "== Docker $(docker version --format '{{.Server.Version}}') (rootless)"
docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep '^dccon-'
echo; echo "== CDI GPUs (GPU 0 is faulty and excluded)"
grep -oE 'name: gpu[0-9]+' ~/.config/cdi/nvidia-gpu123.yaml | sed 's/name: /nvidia.com\/gpu=/'
echo; echo "== HF cache /mnt/main/nsu102/hf"
ls /mnt/main/nsu102/hf/hub | grep '^models--' | sed 's/^models--//; s/--/\//'
du -sh /mnt/main/nsu102/hf | cut -f1
EOF
}

smoke() {
  local image="$1" gpu="$2"
  ssh "$HOST" "cd $REMOTE && GPU=$gpu OFFLINE=1 bash experiments/env/docker.sh run $image \
    python3 experiments/scripts/smoke_test.py $image"
}

run() {
  case "$check" in
    environment)  environment ;;
    smoke-main)   smoke main 1 ;;
    smoke-vllm)   smoke vllm 1 ;;
    smoke-pylate) smoke pylate 1 ;;
    *) echo "usage: $0 environment|smoke-main|smoke-vllm|smoke-pylate" >&2; return 2 ;;
  esac
}

started=$SECONDS
run 2>&1 | tee "$LOGS/$check.log"
status=${PIPESTATUS[0]}
echo "Elapsed seconds: $((SECONDS - started))" | tee -a "$LOGS/$check.log"
exit "$status"
