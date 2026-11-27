"""Download the pinned public DCCON snapshot from R2 into data/week/.

Follows PUBLIC_DATASET_ACCESS.md: sequential requests, SHA-256 checks, atomic writes,
and existing files are skipped when their hash matches.

    python experiments/scripts/download_snapshot.py                   # manifests + images (~2.7 GB)
    python experiments/scripts/download_snapshot.py --manifests-only  # text only (GPU server)
"""
import argparse
import hashlib
import json
import ssl
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import certifi

PUBLIC_BASE = "https://pub-f4a951017f0947809066cb604e7fdb81.r2.dev"
PREFIX = "dccon/week/20261112T112907Z"
BASE = f"{PUBLIC_BASE}/{PREFIX}"
MANIFESTS = ("ranking.json", "packages.jsonl", "stickers.jsonl", "errors.jsonl", "summary.json")
TLS = ssl.create_default_context(cafile=certifi.where())
ROOT = Path(__file__).resolve().parents[2]


def fetch(url):
    for attempt in range(5):
        try:
            request = Request(url, headers={"User-Agent": "dccon-dataset-reader/1.0"})
            with urlopen(request, timeout=60, context=TLS) as response:
                return response.read()
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            retry_after = error.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else 2 ** attempt
        except (URLError, TimeoutError):
            if attempt == 4:
                raise
            delay = 2 ** attempt
        time.sleep(delay)


def write_atomic(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(body)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "week")
    parser.add_argument("--manifests-only", action="store_true")
    args = parser.parse_args()

    for name in MANIFESTS:
        write_atomic(args.output / name, fetch(f"{BASE}/{name}"))
    print(f"manifests -> {args.output}")
    if args.manifests_only:
        return

    with (args.output / "stickers.jsonl").open(encoding="utf-8") as source:
        stickers = [row for line in source if line.strip()
                    for row in [json.loads(line)] if row["image_status"] == "ok"]
    fetched = 0
    for number, row in enumerate(stickers, start=1):
        relative = PurePosixPath(row["image_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
            raise ValueError(f"Unexpected image path: {relative}")
        destination = args.output.joinpath(*relative.parts)
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == row["sha256"]:
            continue
        body = fetch(f"{BASE}/{quote(str(relative), safe='/')}")
        if hashlib.sha256(body).hexdigest() != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}")
        write_atomic(destination, body)
        fetched += 1
        if number % 200 == 0:
            print(f"{number}/{len(stickers)} checked, {fetched} downloaded", flush=True)
        time.sleep(0.2)
    print(f"done: {len(stickers)} images, {fetched} newly downloaded")


if __name__ == "__main__":
    main()
