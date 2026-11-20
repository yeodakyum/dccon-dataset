#!/usr/bin/env python3
"""Collect a snapshot of DCInside's public DCCON ranking."""

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from http.cookiejar import CookieJar
from io import BytesIO
import json
from pathlib import Path
import re
import ssl
import sys
from tempfile import NamedTemporaryFile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, HTTPSHandler, Request, build_opener

from PIL import Image, ImageSequence
import truststore


SITE = "https://dccon.dcinside.com"
DETAIL_URL = f"{SITE}/index/package_detail"
IMAGE_URL = "https://dcimg5.dcinside.com/dccon.php"
# The image CDN returns 404 for unrecognized User-Agent strings.
USER_AGENT = "Mozilla/5.0"
PERIODS = {"day": "D", "week": "W", "month": "M"}
EXTENSIONS = {"GIF": "gif", "JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(data)
            temporary.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Client:
    def __init__(self, page, delay):
        self.page = page
        self.delay = delay
        self.next_request = 0
        self.cookies = CookieJar()
        tls = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.opener = build_opener(HTTPCookieProcessor(self.cookies), HTTPSHandler(context=tls))

    def get(self, url, *, form=None, limit=2 * 1024 * 1024):
        headers = {"User-Agent": USER_AGENT, "Referer": self.page}
        if form is not None:
            headers.update({"Origin": SITE, "X-Requested-With": "XMLHttpRequest"})
        data = urlencode(form).encode() if form is not None else None
        request = Request(url, data=data, headers=headers)
        for attempt in range(4):
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + self.delay
            try:
                with self.opener.open(request, timeout=25) as response:
                    body = response.read(limit + 1)
                    if len(body) > limit:
                        raise ValueError(f"Response exceeds {limit} bytes")
                    return body
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504}:
                    raise
                retry_after = error.headers.get("Retry-After")
                if retry_after:
                    try:
                        seconds = int(retry_after)
                    except ValueError:
                        try:
                            seconds = (parsedate_to_datetime(retry_after) -
                                       datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            seconds = 0
                    self.next_request = max(self.next_request, time.monotonic() + seconds)
                if attempt == 3:
                    raise
            except (URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise
            self.next_request = max(self.next_request, time.monotonic() + 2 ** attempt)

    def detail(self, package_id):
        csrf = next((cookie.value for cookie in self.cookies if cookie.name == "ci_c"), "")
        body = self.get(DETAIL_URL, form={"ci_t": csrf, "package_idx": package_id})
        return json.loads(body.decode("utf-8-sig"))


def numeric_id(value):
    value = str(value)
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"Invalid numeric ID {value!r}")
    return value


def ranking_snapshot(client, output, period, limit):
    path = output / "ranking.json"
    if path.exists():
        snapshot = read_json(path)
        if (snapshot["period"], snapshot["limit"]) != (period, limit):
            raise ValueError("Existing ranking has a different --period or --limit; use another --output")
    else:
        url = f"https://json.dcinside.com/json1/dccon_{period}_top100.php?jsoncallback=dccon_rank"
        body = client.get(url).decode("utf-8-sig").strip()
        wrapped = re.fullmatch(r"dccon_rank\s*\((.*)\)\s*;?", body, re.S)
        rows = json.loads(wrapped[1] if wrapped else body)
        snapshot = {
            "schema_version": 1, "source_url": url, "ranking_page": client.page,
            "period": period, "limit": limit, "fetched_at": utc_now(), "entries": rows,
        }
    rows = snapshot["entries"]
    if not isinstance(rows, list) or len(rows) < limit:
        raise ValueError(f"Ranking did not return at least {limit} entries")
    ids = [numeric_id(row["package_idx"]) for row in rows[:limit]]
    if len(set(ids)) != limit:
        raise ValueError("Ranking contains duplicate package IDs")
    if not path.exists():
        write_json(path, snapshot)
    return snapshot


def get_package(client, output, package_id):
    path = output / "raw" / "packages" / f"{package_id}.json"
    cached = path.exists()
    record = read_json(path) if cached else {
        "fetched_at": None, "source_url": DETAIL_URL, "response": client.detail(package_id)
    }
    data = record["response"]
    if not isinstance(data, dict) or not isinstance(data.get("info"), dict):
        raise ValueError("Package response is missing info")
    if str(data["info"].get("package_idx")) != package_id:
        raise ValueError("Package response ID does not match the request")
    items = data.get("detail")
    if not isinstance(items, list) or not items:
        raise ValueError("Package response is missing its sticker list")
    ids = [numeric_id(item["idx"]) for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Package response contains duplicate sticker IDs")
    if not cached:
        record["fetched_at"] = utc_now()
        write_json(path, record)
    return record, path.relative_to(output).as_posix()


def image_url(path):
    if not isinstance(path, str) or not path:
        raise ValueError("Missing image path in detail response")
    return f"{IMAGE_URL}?{urlencode({'no': path})}"


def inspect_image(body):
    with Image.open(BytesIO(body)) as image:
        image_format = image.format
        if image_format not in EXTENSIONS:
            raise ValueError(f"Unsupported image format {image_format!r}")
        width, height = image.size
        mode = image.mode
        loop = image.info.get("loop")
        frame_count = getattr(image, "n_frames", 1)
        durations = []
        has_transparency = False
        for frame in ImageSequence.Iterator(image):
            frame.load()
            if frame_count > 1:
                durations.append(frame.info.get("duration"))
            if not has_transparency and ("A" in frame.getbands() or "transparency" in frame.info):
                has_transparency = frame.convert("RGBA").getchannel("A").getextrema()[0] < 255
    return {
        "sha256": sha256(body).hexdigest(), "bytes": len(body),
        "format": image_format, "mime_type": Image.MIME.get(image_format),
        "width": width, "height": height, "mode": mode, "has_transparency": has_transparency,
        "frame_count": frame_count, "animated": frame_count > 1, "loop": loop,
        "frame_durations_ms": durations,
        "duration_ms": sum(durations) if durations and None not in durations else None,
    }


def fetch_image(client, output, package_id, sticker_id, url):
    # A changed source URL gets a new cache key even if the sticker ID stays the same.
    key = sha256(url.encode()).hexdigest()[:20]
    base = output / "images" / package_id / f"{sticker_id}-{key}"
    for extension in EXTENSIONS.values():
        existing = base.with_suffix(f".{extension}")
        if existing.is_file():
            try:
                metadata = inspect_image(existing.read_bytes())
                return {"image_path": existing.relative_to(output).as_posix(), **metadata}
            except (OSError, ValueError, Image.DecompressionBombError):
                break
    body = client.get(url, limit=20 * 1024 * 1024)
    metadata = inspect_image(body)
    path = base.with_suffix("." + EXTENSIONS[metadata["format"]])
    atomic_write(path, body)
    return {"image_path": path.relative_to(output).as_posix(), **metadata}


def nullable_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def package_row(rank, snapshot, record, raw_path):
    data = record["response"]
    info = data["info"]
    tags = data.get("tags") or []
    return {
        "package_id": str(info["package_idx"]),
        "rank": nullable_int(rank.get("rank")),
        "ranking_period": snapshot["period"],
        "ranking_fetched_at": snapshot["fetched_at"],
        "title": info.get("title") or rank.get("title"),
        "description": info.get("description", rank.get("desc")),
        "creator": info.get("seller_name") or rank.get("nick_name"),
        "tags": [tag["tag"] for tag in tags if isinstance(tag, dict) and "tag" in tag],
        "registered_at": info.get("reg_date"),
        "price": nullable_int(info.get("price")),
        "sale_count": nullable_int(info.get("sale_count")),
        "reported_icon_count": nullable_int(info.get("icon_cnt")),
        "detail_icon_count": len(data["detail"]),
        "category": info.get("category"),
        "state": info.get("state"),
        "period_days": nullable_int(info.get("period")),
        "cover_url": image_url(info["main_img_path"]) if info.get("main_img_path") else None,
        "list_image_url": image_url(info["list_img_path"]) if info.get("list_img_path") else None,
        "detail_fetched_at": record["fetched_at"],
        "raw_detail_path": raw_path,
        "rank_raw": rank,
        "info_raw": info,
        "tags_raw": tags,
    }


def crawl(args):
    output = args.output
    client = Client(f"{SITE}/hot/1?period={PERIODS[args.period]}", args.delay)
    snapshot = ranking_snapshot(client, output, args.period, args.limit)
    packages, stickers, errors = [], [], []
    started_at = utc_now()

    def save(complete=False):
        write_jsonl(output / "packages.jsonl", packages)
        write_jsonl(output / "stickers.jsonl", stickers)
        write_jsonl(output / "errors.jsonl", errors)
        write_json(output / "summary.json", {
            "schema_version": 1, "period": args.period, "requested_packages": args.limit,
            "ranking_fetched_at": snapshot["fetched_at"], "started_at": started_at,
            "updated_at": utc_now(), "complete": complete, "metadata_only": args.metadata_only,
            "packages": len(packages), "sticker_rows": len(stickers),
            "images": sum(row["image_status"] == "ok" for row in stickers),
            "errors": len(errors),
        })

    def report_error(stage, package_id, error, **fields):
        errors.append({"stage": stage, "package_id": package_id,
                       "error_type": type(error).__name__, "message": str(error), **fields})
        print(f"{stage} {package_id}: {error}", file=sys.stderr)

    save()
    try:
        if any(not (output / "raw" / "packages" / f"{entry['package_idx']}.json").exists()
               for entry in snapshot["entries"][:args.limit]):
            client.get(client.page)
        for position, entry in enumerate(snapshot["entries"][:args.limit], 1):
            package_id = str(entry["package_idx"])
            print(f"[{position}/{args.limit}] {package_id} {entry.get('title', '')}", flush=True)
            try:
                record, raw_path = get_package(client, output, package_id)
                package = package_row(entry, snapshot, record, raw_path)
            except (OSError, ValueError, KeyError, TypeError) as error:
                report_error("package", package_id, error)
                save()
                continue
            packages.append(package)
            for item in record["response"]["detail"]:
                sticker_id = str(item["idx"])
                row = {
                    "package_id": package_id, "sticker_id": sticker_id, "rank": package["rank"],
                    "title": item.get("title"), "sort": nullable_int(item.get("sort")),
                    "source_extension": item.get("ext"), "source_url": None, "image_path": None,
                    "image_status": "skipped" if args.metadata_only else "error", "item_raw": item,
                }
                stickers.append(row)
                try:
                    row["source_url"] = image_url(item.get("path"))
                    if not args.metadata_only:
                        row.update(fetch_image(client, output, package_id, sticker_id, row["source_url"]))
                        row["image_status"] = "ok"
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    row["image_status"] = "error"
                    report_error("image", package_id, error, sticker_id=sticker_id,
                                 source_url=row["source_url"])
            save()
    except KeyboardInterrupt:
        save()
        raise
    save(complete=not errors)
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", choices=PERIODS, default="week")
    parser.add_argument("--limit", type=int, choices=range(1, 101), metavar="1..100", default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--delay", type=float, default=0.35, help="Seconds between requests (default: 0.35)")
    parser.add_argument("--metadata-only", action="store_true", help="Skip image downloads")
    args = parser.parse_args()
    if not 0 <= args.delay < float("inf"):
        parser.error("--delay must be finite and nonnegative")
    if args.output is None:
        args.output = Path("data") / args.period
    try:
        return crawl(args)
    except KeyboardInterrupt:
        print("Interrupted. Run again with the same options to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Crawl failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
