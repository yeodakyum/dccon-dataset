#!/usr/bin/env python3
"""Upload a DCCON dataset snapshot to Cloudflare R2."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import dotenv_values


IMAGE_PATH = re.compile(r"images/[0-9]+/[0-9]+-[0-9a-f]{20}\.(gif|jpg|png|webp)")
DETAIL_PATH = re.compile(r"raw/packages/[0-9]+\.json")
MANIFESTS = ("packages.jsonl", "stickers.jsonl", "errors.jsonl")


def object_prefix(base, ranking):
    period = ranking["period"]
    if period not in {"day", "week", "month"}:
        raise ValueError(f"Invalid ranking period: {period!r}")
    timestamp = datetime.fromisoformat(ranking["fetched_at"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Ranking timestamp must include a time zone")
    version = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "/".join(part for part in (base.strip("/"), period, version) if part)


class Uploader:
    def __init__(self, client, bucket, prefix, directory, workers):
        self.client = client
        self.bucket = bucket
        self.prefix = prefix
        self.directory = directory
        self.workers = workers
        self.uploaded_images = {}

    def put(self, relative_path, body, digest):
        if self.client is None:
            return True
        key = f"{self.prefix}/{relative_path}"
        try:
            existing = self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            if (existing["ContentLength"] == len(body)
                    and existing.get("Metadata", {}).get("sha256") == digest):
                return False

        content_type = ("application/x-ndjson" if relative_path.endswith(".jsonl")
                        else mimetypes.guess_type(relative_path)[0])
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=content_type or "application/octet-stream",
            Metadata={"sha256": digest},
        )
        return True

    def upload_images(self, images):
        def upload(entry):
            relative_path, expected_digest = entry
            path = self.directory / relative_path
            stat = path.stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            cached = self.uploaded_images.get(relative_path)
            if cached is not None and cached[0] == signature:
                digest = cached[1]
                body = None
            else:
                body = path.read_bytes()
                digest = sha256(body).hexdigest()
            if expected_digest is not None and digest != expected_digest:
                raise ValueError(f"SHA-256 mismatch: {relative_path}")
            changed = body is not None and self.put(relative_path, body, digest)
            return relative_path, signature, digest, changed

        changed = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for path, signature, digest, written in pool.map(upload, images.items()):
                self.uploaded_images[path] = (signature, digest)
                changed += written
        return changed

    def upload_metadata(self, files):
        return sum(self.put(path, body, sha256(body).hexdigest()) for path, body in files.items())


def final_manifest(directory, ranking_body):
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary_body = summary_path.read_bytes()
    summary = json.loads(summary_body)
    if not summary["complete"]:
        return None

    metadata = {"ranking.json": ranking_body}
    metadata.update({name: (directory / name).read_bytes() for name in MANIFESTS})
    if summary_path.read_bytes() != summary_body:
        return None

    packages = [json.loads(line) for line in metadata["packages.jsonl"].splitlines()]
    stickers = [json.loads(line) for line in metadata["stickers.jsonl"].splitlines()]
    images = {}
    for row in stickers:
        if row["image_status"] == "ok":
            path = row["image_path"]
            digest = row["sha256"]
            if not IMAGE_PATH.fullmatch(path) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid image path or hash: {path!r}")
            images[path] = digest

    ranking = json.loads(ranking_body)
    if (summary["ranking_fetched_at"] != ranking["fetched_at"]
            or summary["period"] != ranking["period"]
            or summary["packages"] != ranking["limit"]
            or len(packages) != summary["packages"]
            or len(stickers) != summary["sticker_rows"]
            or len(images) != summary["images"]
            or metadata["errors.jsonl"].strip()
            or len(images) != (0 if summary["metadata_only"] else len(stickers))):
        raise ValueError("Completed summary does not match the dataset manifests")

    for row in packages:
        path = row["raw_detail_path"]
        if not DETAIL_PATH.fullmatch(path):
            raise ValueError(f"Invalid detail path: {path!r}")
        metadata[path] = (directory / path).read_bytes()
    # Consumers can use summary.json as the completion marker.
    metadata["summary.json"] = summary_body
    return images, metadata


def crawl_images(directory):
    # The crawler publishes images via atomic rename. Temporary names fail this pattern.
    return {
        relative_path: None
        for path in (directory / "images").glob("*/*")
        if IMAGE_PATH.fullmatch(relative_path := path.relative_to(directory).as_posix())
        and path.is_file()
    }


def run(args, credentials):
    ranking_path = args.directory / "ranking.json"
    while not ranking_path.exists():
        if not args.follow:
            raise ValueError(f"Ranking snapshot missing: {ranking_path}")
        print(f"Waiting for {ranking_path}", flush=True)
        time.sleep(args.interval)

    ranking_body = ranking_path.read_bytes()
    prefix = object_prefix(args.prefix, json.loads(ranking_body))
    client = None
    if not args.dry_run:
        session = boto3.Session(profile_name=args.profile, **credentials)
        client = session.client(
            "s3",
            endpoint_url=f"https://{args.account_id}.r2.cloudflarestorage.com",
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                retries={"mode": "standard", "total_max_attempts": 5},
                max_pool_connections=args.workers,
            ),
        )

    uploader = Uploader(client, args.bucket, prefix, args.directory, args.workers)
    print(f"{'Dry run for' if args.dry_run else 'Uploading to'} r2://{args.bucket}/{prefix}/", flush=True)
    while True:
        complete = final_manifest(args.directory, ranking_body)
        if complete is not None:
            images, metadata = complete
            changed = uploader.upload_images(images)
            if (args.directory / "summary.json").read_bytes() != metadata["summary.json"]:
                raise ValueError("Crawler restarted during final upload. Run this command again.")
            changed += uploader.upload_metadata(metadata)
            print(f"Complete: {len(images)} images, {len(metadata)} metadata objects, "
                  f"{changed} {'would write' if args.dry_run else 'written in final pass'}.")
            return
        if not args.follow:
            raise ValueError("Crawl is incomplete. Use --follow to upload as files arrive.")

        images = crawl_images(args.directory)
        changed = uploader.upload_images(images)
        print(f"Images available: {len(images)}. "
              f"{changed} {'would write' if args.dry_run else 'written'}. Waiting for crawl completion.",
              flush=True)
        time.sleep(args.interval)


def main():
    settings = dotenv_values(Path(__file__).resolve().with_name(".env"), interpolate=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/week"))
    parser.add_argument("--bucket", default=settings.get("R2_BUCKET") or os.environ.get("R2_BUCKET"))
    parser.add_argument("--account-id", default=settings.get("R2_ACCOUNT_ID") or os.environ.get("R2_ACCOUNT_ID"))
    parser.add_argument("--prefix", default="dccon", help="Base prefix before period and timestamp")
    parser.add_argument("--profile", help="AWS credentials profile containing the R2 access key")
    parser.add_argument("--follow", action="store_true", help="Upload finished images while crawling")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Validate local inputs without R2 requests")
    args = parser.parse_args()

    if args.interval < 1 or not 1 <= args.workers <= 32:
        parser.error("--interval must be positive and --workers must be between 1 and 32")
    if any(part in {".", ".."} for part in args.prefix.split("/")):
        parser.error("--prefix cannot contain '.' or '..' segments")
    credentials = {}
    if not args.dry_run:
        if not args.bucket:
            parser.error("Set R2_BUCKET or pass --bucket")
        if not args.account_id or not re.fullmatch(r"[0-9a-fA-F]{32}", args.account_id):
            parser.error("Set R2_ACCOUNT_ID or pass --account-id with your 32-digit hex account ID")
        key_fields = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
        if not args.profile and any(field in settings for field in key_fields):
            access_key, secret_key = (settings.get(field) for field in key_fields)
            if not access_key or not secret_key:
                parser.error("Fill in R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY in .env")
            credentials = {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key}
    elif not args.bucket:
        args.bucket = "<bucket>"

    try:
        run(args, credentials)
        return 0
    except KeyboardInterrupt:
        print("Stopped. Run the same command again to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, BotoCoreError, ClientError) as error:
        print(f"Upload failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
