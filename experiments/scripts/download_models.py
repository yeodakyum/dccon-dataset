"""Pre-download every model in experiments/configs into HF_HOME.

    python experiments/scripts/download_models.py                 # embedders, SLMs, rerankers
    python experiments/scripts/download_models.py --only embedders
    python experiments/scripts/download_models.py --optional      # also the "optional" models
    python experiments/scripts/download_models.py --dry-run

Gated repos (EmbeddingGemma, HyperCLOVA X SEED) need an HF token and accepted terms.
The server has no token, so:
    server (docker main): python experiments/scripts/download_models.py --skip-gated
    Mac:                  python experiments/scripts/download_models.py --gated-only
                          bash experiments/scripts/push_hf_cache.sh   # copy them to the server
Failures are reported at the end instead of stopping the whole download.
"""
import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# Formats we never load; skipping them saves several GB on repos that ship many.
IGNORE = ["*.onnx", "*.onnx_data", "onnx/*", "openvino/*", "*.gguf", "*.h5", "*.msgpack",
          "*.ot", "flax_model*", "tf_model*", "rust_model*", "coreml/*", "*.mlpackage/*"]


def hub_ids(group, include_optional):
    """Yield (hub_id, gated) for every model in a config file."""
    config = yaml.safe_load((CONFIGS / f"{group}.yaml").read_text(encoding="utf-8"))
    sections = [config.get("models", {})]
    if include_optional:
        sections.append(config.get("optional", {}))
    for section in sections:
        for name, spec in section.items():
            if name == "backbones":
                yield from ((repo, False) for repo in spec)
            elif isinstance(spec, dict) and spec.get("hub_id"):
                yield spec["hub_id"], bool(spec.get("gated"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["embedders", "slms", "rerankers"], action="append")
    parser.add_argument("--optional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    gated = parser.add_mutually_exclusive_group()
    gated.add_argument("--skip-gated", action="store_true")
    gated.add_argument("--gated-only", action="store_true")
    args = parser.parse_args()

    repos = []
    for group in args.only or ["embedders", "slms", "rerankers"]:
        for repo, is_gated in hub_ids(group, args.optional):
            if (args.skip_gated and is_gated) or (args.gated_only and not is_gated):
                continue
            if repo not in repos:
                repos.append(repo)

    failed = []
    for number, repo in enumerate(repos, start=1):
        print(f"[{number}/{len(repos)}] {repo}", flush=True)
        if args.dry_run:
            continue
        try:
            snapshot_download(repo, ignore_patterns=IGNORE)
        except Exception as error:  # keep going; report at the end
            failed.append((repo, f"{type(error).__name__}: {error}"))
            print(f"  failed: {failed[-1][1].splitlines()[0]}", flush=True)
    if failed:
        print("\nFailed downloads:")
        for repo, message in failed:
            print(f"  {repo}: {message.splitlines()[0]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
