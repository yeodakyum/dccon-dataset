# Experiments (plan.md)

Code is written on the Mac and run on the GPU server (`snu-gpu`) in Docker. Images, gated models, and OpenRouter calls stay on the Mac.

| Where | What | Environment |
|---|---|---|
| Mac | Snapshot images, gated models, test-set generation (OpenRouter, Claude Haiku) | `~/.local/share/venvs/dccon` (`requirements.txt` + `experiments/env/requirements-local.txt`), `.env` |
| Server | Embedders, BM25, FAISS, rerankers | image `dccon-main` |
| Server | SLM reaction extraction | image `dccon-vllm` (vLLM v0.32.0, CUDA 12.9) |
| Server | KURE-v2 late interaction | image `dccon-pylate` |

Model-specific prefixes, pooling, and thinking-mode switches are in `configs/`. Shared rules (512 tokens, fp16, IndexFlatIP, temperature 0, ...) are in `configs/common.yaml`.

## Setup

```bash
# Mac
cp .env.example .env                                     # fill OPENROUTER_API_KEY
python experiments/scripts/download_snapshot.py          # data/week (~2.7 GB)
python experiments/scripts/download_models.py --gated-only
bash experiments/scripts/push_hf_cache.sh                # gated models -> server NFS cache
bash experiments/scripts/sync.sh push                    # code -> ~/project/dccon-dataset

# Server (ssh snu-gpu; cd ~/project/dccon-dataset)
bash experiments/env/make_cdi_spec.sh                    # once (and after driver updates)
bash experiments/env/docker.sh build
bash experiments/env/docker.sh run main python3 experiments/scripts/download_snapshot.py --manifests-only
bash experiments/env/docker.sh run main python3 experiments/scripts/download_models.py --skip-gated
bash experiments/env/docker.sh run main   python3 experiments/scripts/smoke_test.py main
bash experiments/env/docker.sh run vllm   python3 experiments/scripts/smoke_test.py vllm
bash experiments/env/docker.sh run pylate python3 experiments/scripts/smoke_test.py pylate
```

Server notes:

- Docker uses the existing rootless daemon (`/run/user/1101/study-signal-docker.sock`). Override with `DOCKER_HOST`.
- GPU 0 shows `ERR!` in `nvidia-smi` and breaks `docker run --gpus` (NVML error). `bash experiments/env/make_cdi_spec.sh` writes a CDI spec for GPUs 1–3 (`~/.config/cdi/nvidia-gpu123.yaml`), and `docker.sh` attaches one of them. The default is GPU 1; pick another with `GPU=2` or `GPU=3`.
- The HF cache is on NFS at `/mnt/main/nsu102/hf`, mounted at `/hf`, because `/` is shared and 92% full. `OFFLINE=1` blocks Hub access during runs.
- Results go to `runs/`. Copy them to the Mac with `bash experiments/scripts/sync.sh pull`.
