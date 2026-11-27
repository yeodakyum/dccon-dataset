"""Bind VS Code screenshots to the env_check.sh logs they document.

Expects experiments/results/vscode-screenshots/<check>.png for every log in
experiments/results/logs/ and writes manifest.json with SHA-256 hashes.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
SHOTS = RESULTS / "vscode-screenshots"
CHECKS = ("environment", "smoke-main", "smoke-vllm", "smoke-pylate")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = {}
for check in CHECKS:
    image = SHOTS / f"{check}.png"
    log = RESULTS / "logs" / f"{check}.log"
    assert image.exists(), f"Missing screenshot: {image}"
    final_line = log.read_text(encoding="utf-8").rstrip().splitlines()[-1]
    assert final_line.startswith("Elapsed seconds: "), f"Incomplete run: {log}"
    width, height = Image.open(image).size
    manifest[check] = {
        "image_file": str(image.relative_to(ROOT)), "width": width, "height": height,
        "image_sha256": sha256(image),
        "source": "Actual native VS Code integrated terminal screenshot",
        "log_file": str(log.relative_to(ROOT)), "log_sha256": sha256(log),
    }
(SHOTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"{len(manifest)} VS Code screenshots recorded with file hashes.")
