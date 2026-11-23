"""Create an image-free JSONL dataset from Qwen's DCCON annotations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from getpass import getpass
from itertools import islice
import hashlib
import json
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.error import HTTPError, URLError

from tagging_gui import DEFAULT_PROMPT, GIF_PROMPT, MODEL, ROOT, load_stickers, tag_sticker

IMAGE_TOKENS = 2048
ESTIMATED_TEXT_TOKENS = 400


def cost(prompt_tokens, completion_tokens):
    return (prompt_tokens * 0.80 + completion_tokens * 4.00) / 1_000_000


def write_atomic(path, content):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temp = Path(file.name)
        try:
            file.write(content)
            file.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def annotation(result):
    tags = json.loads(result["output"])
    if result["finish_reason"] != "stop" or not isinstance(tags, dict):
        raise ValueError("Incomplete JSON response")
    for field in ("image_text", "uncertain"):
        if not isinstance(tags.get(field), list) or any(not isinstance(x, str) for x in tags[field]):
            raise ValueError(f"Invalid {field}")
    if not isinstance(tags.get("visual"), str) or not tags["visual"].strip():
        raise ValueError("Missing visual description")
    reactions = tags.get("reactions")
    if not isinstance(reactions, list) or len(reactions) > 3 or any(
        not isinstance(item, dict)
        or any(not isinstance(item.get(field), str) or not item[field].strip()
               for field in ("intent", "evidence"))
        for item in reactions
    ):
        raise ValueError("Invalid reactions")
    return {field: tags[field] for field in ("image_text", "visual", "reactions", "uncertain")}


def mixed_sample(stickers, count, seed):
    rng = random.Random(seed)
    groups = {False: {}, True: {}}
    for item in stickers:
        metadata = item[1][0]
        groups[metadata["is_gif"]].setdefault(metadata["package_id"], []).append(item)

    samples = []
    for packages in groups.values():
        if sum(map(len, packages.values())) < count:
            raise ValueError("The dataset needs at least that many stickers of each type")
        pools = list(packages.values())
        for pool in pools:
            rng.shuffle(pool)
        rng.shuffle(pools)
        selected = []
        while len(selected) < count:
            for pool in pools:
                if pool:
                    selected.append(pool.pop())
                    if len(selected) == count:
                        break
        samples.append(selected)
    return [item for pair in zip(*samples) for item in pair]


def tag_with_retry(sticker, prompt, api_key):
    for attempt in range(4):
        try:
            return tag_sticker(sticker, prompt, api_key)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            retry_after = (exc.headers.get("Retry-After") or "").strip()
            delay = min(60, int(retry_after)) if retry_after.isdecimal() else 2 ** (attempt + 1)
            print(f"Groq HTTP {exc.code}; retrying in {delay}s", flush=True)
            sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/week")
    parser.add_argument("--output", type=Path, default=ROOT / "data/groq_tags")
    parser.add_argument("--limit", type=int, help="Select only the first N eligible stickers")
    parser.add_argument("--prompt", type=Path, help="Use a custom prompt with the same output fields")
    parser.add_argument("--per-kind", type=int, help="Sample this many static stickers and animated GIFs")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for --per-kind")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Groq requests (1–4)")
    parser.add_argument("--run", action="store_true", help="Send requests to Groq and write annotations")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.per_kind is not None and (args.per_kind < 1 or args.limit is not None or args.prompt):
        parser.error("--per-kind must be positive and uses the built-in prompts without --limit")

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    synthetic = ROOT / "experiments/pilot1/synthetic"
    if output.is_relative_to(synthetic) or output == dataset or dataset.is_relative_to(output):
        parser.error("--output must be a separate dataset directory")

    prompt = args.prompt.read_text(encoding="utf-8") if args.prompt else DEFAULT_PROMPT
    if not prompt.strip():
        parser.error("Prompt is empty")
    mixed = args.per_kind is not None
    stickers = list(load_stickers(dataset, include_gifs=mixed).items())
    if mixed:
        try:
            stickers = mixed_sample(stickers, args.per_kind, args.seed)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        stickers = stickers[:args.limit]
    count = len(stickers)
    low = cost(count * (IMAGE_TOKENS + ESTIMATED_TEXT_TOKENS), count * 100)
    high = cost(count * (IMAGE_TOKENS + ESTIMATED_TEXT_TOKENS), count * 200)
    if mixed:
        print(f"{args.per_kind:,} static + {args.per_kind:,} GIF stickers · {MODEL}", flush=True)
    else:
        print(f"{count:,} static stickers · {MODEL} · estimate ${low:.2f}–${high:.2f} USD", flush=True)
    if not args.run or not count:
        return

    api_key = os.getenv("GROQ_API_KEY", "").strip() or getpass("Groq API key: ").strip()
    if not api_key:
        parser.error("A Groq API key is required")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run.json"
    config = {"dataset": str(dataset), "model": MODEL, "prompt": prompt}
    if mixed:
        config.update(gif_prompt=GIF_PROMPT, seed=args.seed, sample_ids=[key for key, _ in stickers])
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        parser.error("This output directory belongs to a different dataset, model, or prompt")
    if not config_path.exists():
        write_atomic(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    checkpoint = output / "records"
    checkpoint.mkdir(exist_ok=True)
    def process(item):
        index, (key, sticker) = item
        metadata, image_path, _ = sticker
        image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        record_path = checkpoint / f"{key}.json"
        if record_path.exists():
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if existing["image_sha256"] == image_sha256:
                return existing
        try:
            result = tag_with_retry(sticker, GIF_PROMPT if metadata["is_gif"] else prompt, api_key)
            record = {
                **metadata,
                "image_path": image_path.relative_to(dataset).as_posix(),
                "image_sha256": image_sha256,
                "model": result["model"],
                "review_status": "unreviewed",
                **annotation(result),
                "usage": result["usage"],
            }
        except HTTPError as exc:
            print(f"{key}: Groq HTTP {exc.code}. Rerun to resume.", flush=True)
            return None
        except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
            print(f"{key}: {type(exc).__name__}. Rerun to resume.", flush=True)
            return None
        write_atomic(record_path, json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{index:,}/{count:,} · {key}", flush=True)

        return record

    exported = []
    items = enumerate(stickers, 1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while batch := list(islice(items, args.workers)):
            results = list(pool.map(process, batch))
            exported.extend(record for record in results if record is not None)
            if any(record is None for record in results):
                break
    failed = len(exported) != count

    write_atomic(output / "annotations.jsonl",
                 "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in exported))
    input_tokens = sum(record["usage"].get("prompt_tokens", 0) or 0 for record in exported)
    output_tokens = sum(record["usage"].get("completion_tokens", 0) or 0 for record in exported)
    print(f"Exported {len(exported):,}/{count:,} · recorded token cost ${cost(input_tokens, output_tokens):.2f}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
