# Use the public DCCON dataset from Python

## Public URLs

The `dccon-dataset` bucket is hosted on **Cloudflare R2**, which provides an
S3-compatible API. Public readers can use ordinary HTTPS requests without an
account, API key, or AWS credentials.

- [Dataset catalog](https://pub-f4a951017f0947809066cb604e7fdb81.r2.dev/catalog.json)
- [Snapshot summary](https://pub-f4a951017f0947809066cb604e7fdb81.r2.dev/dccon/week/20261112T112907Z/summary.json)
- [Sticker manifest](https://pub-f4a951017f0947809066cb604e7fdb81.r2.dev/dccon/week/20261112T112907Z/stickers.jsonl)
- [Package manifest](https://pub-f4a951017f0947809066cb604e7fdb81.r2.dev/dccon/week/20261112T112907Z/packages.jsonl)

```text
Public bucket base: https://pub-f4a951017f0947809066cb604e7fdb81.r2.dev
Snapshot prefix:    dccon/week/20261112T112907Z
```

Append the object key directly to the public bucket base. Do **not** insert the
bucket name into this public URL. The bucket root does not provide a directory
listing; use `catalog.json` to discover published snapshots.

## Available snapshot

Checked on November 20, 2026:

| Property | Value |
| --- | --- |
| Ranking | Weekly top 100 packages |
| Ranking fetched at | November 12, 2026, 11:29:07 UTC |
| Packages | 100 |
| Sticker rows | 8,245 |
| Available image objects | 8,218 |
| Failed image rows | 27 |
| Image bytes in the bucket | 2,681,717,476 bytes (about 2.68 GB) |
| Crawl complete | `false` |

This is a **partial snapshot**. Its original `summary.json` and `errors.jsonl`
are published unchanged. Always filter sticker rows to `image_status == "ok"`.
The published image keys and sizes were checked against those successful rows.
The ranking timestamp describes the ranking fetch; package details can have
later fetch times. `catalog.json` is a publication index, not a crawl completion
marker; inspect `summary.json` for completion and metadata-only status.

## Quick start: metadata and one image

Requires Python 3.10 or newer. Install Pillow to decode images and certifi for HTTPS certificates:

```bash
python -m pip install Pillow certifi
```

This example pins the snapshot for reproducibility and uses the standard library
for HTTP. It retries temporary server failures and rate limits.

```python
import hashlib
import json
import ssl
import time
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import certifi
from PIL import Image

PUBLIC_BASE = "https://pub-f4a951017f0947809066cb604e7fdb81.r2.dev"
PREFIX = "dccon/week/20261112T112907Z"
BASE = f"{PUBLIC_BASE}/{PREFIX}"
TLS = ssl.create_default_context(cafile=certifi.where())


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


def read_jsonl(name):
    return [json.loads(line) for line in fetch(f"{BASE}/{name}").splitlines()
            if line.strip()]


summary = json.loads(fetch(f"{BASE}/summary.json"))
print("Crawl complete:", summary["complete"])
packages = {row["package_id"]: row for row in read_jsonl("packages.jsonl")}
stickers = [row for row in read_jsonl("stickers.jsonl")
            if row["image_status"] == "ok"]

row = stickers[0]
url = f"{BASE}/{quote(row['image_path'], safe='/')}"
image_bytes = fetch(url)
if hashlib.sha256(image_bytes).hexdigest() != row["sha256"]:
    raise ValueError("Image checksum mismatch")

image = Image.open(BytesIO(image_bytes))
image.load()  # Loads the first frame; the original GIF bytes retain all frames.
print(packages[row["package_id"]]["title"], row["sticker_id"], image.size)
print("Image URL:", url)
# image.show()  # Optional desktop preview
```

`image_path` is relative to the snapshot prefix. Use it instead of `source_url`
when downloading from this bucket. GIFs may be animated: converting to RGB or
loading only the first frame discards animation information. The manifest
includes `frame_count`, `frame_durations_ms`, and `loop`.

## Download all available images

Run the following after the quick-start code (it reuses `fetch`, `BASE`, and
`stickers`). Files retain the dataset's directory structure. Existing files are
skipped only if their SHA-256 matches. Expect about 2.68 GB of image downloads.
The loop is sequential to keep public endpoint traffic modest.

```python
from pathlib import Path, PurePosixPath

output = Path("dccon-week-20261112")
output.mkdir(parents=True, exist_ok=True)

for name in ("ranking.json", "packages.jsonl", "stickers.jsonl",
             "errors.jsonl", "summary.json"):
    (output / name).write_bytes(fetch(f"{BASE}/{name}"))

for number, row in enumerate(stickers, start=1):
    relative = PurePosixPath(row["image_path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
        raise ValueError(f"Unexpected image path: {relative}")
    destination = output.joinpath(*relative.parts)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == row["sha256"]:
            continue
    body = fetch(f"{BASE}/{quote(str(relative), safe='/')}")
    if hashlib.sha256(body).hexdigest() != row["sha256"]:
        raise ValueError(f"Checksum mismatch: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(body)
    temporary.replace(destination)
    if number % 100 == 0:
        print(f"{number}/{len(stickers)} images processed")
    time.sleep(0.2)
```

This downloads the manifests and available images. Full raw package responses
are also available at `BASE + "/" + package["raw_detail_path"]` for each row in
`packages.jsonl`.

## Discover snapshots

The catalog lists published snapshot prefixes and their partial/complete status.
Pin a specific prefix in experiments rather than silently changing datasets.

```python
catalog = json.loads(fetch(f"{PUBLIC_BASE}/catalog.json"))
for snapshot in catalog["snapshots"]:
    print(snapshot["prefix"], snapshot["images"], snapshot["complete"])
```

Join package and sticker metadata on `package_id`. For training and evaluation,
split by `package_id` to keep each package in one split. Use `sha256` to detect
byte-identical images, including duplicates across packages.

## HTTPS versus S3 / boto3

Public HTTPS is the supported credential-free access method. A public R2 bucket
does not turn its S3 API into an anonymous S3 endpoint. Do not point an unsigned
`boto3` client at the public `r2.dev` URL: it serves object downloads rather than
S3 API operations such as `ListObjectsV2`.

The authenticated S3 API remains available to the bucket owner through the
existing `upload_r2.py` configuration (`region_name="auto"`). Public consumers
do not need `.env`, an account ID, or any access keys. Keep the owner's R2 keys
private; they are not part of this access guide.

## Hosting and maintenance

The existing public development URL was already enabled when checked. Dataset
metadata and `catalog.json` were subsequently published to make the uploaded
images discoverable. CORS changes are unnecessary for Python HTTP clients.

The `r2.dev` endpoint is rate limited and intended for development. For heavier
shared use, connect a custom domain under **R2 → dccon-dataset → Settings →
Custom Domains → Add** and replace `PUBLIC_BASE` in the examples with that domain.
Use Cloudflare's R2 custom-domain connection flow rather than manually pointing
a CNAME at `r2.dev`.

For future completed crawls, run `upload_r2.py` as documented in `README.md`, then
add the uploaded prefix and its actual counts/status to `catalog.json` using the
owner's authenticated S3 client. The existing uploader does not update the
catalog automatically. Publish a catalog entry only after its referenced objects
are uploaded. Preserve existing entries. Never change `complete` to `true`
unless the crawler actually completed successfully.

Cloudflare documentation:

- [Public buckets, custom domains, rate limits, and directory listing](https://developers.cloudflare.com/r2/buckets/public-buckets/)
- [Using boto3 with R2](https://developers.cloudflare.com/r2/examples/aws/boto3/)
