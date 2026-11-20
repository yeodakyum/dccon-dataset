# DCCON dataset

`crawl_dccon.py` collects DCInside's popular DCCON packages and their individual stickers. A package is a sticker set. The default sample is the top **100 packages** in the weekly ranking, with every sticker returned by each package's detail endpoint. Daily and monthly rankings are also supported.

## Collection methodology

1. Fetch the ranking from `https://json.dcinside.com/json1/dccon_<period>_top100.php?jsoncallback=dccon_rank`. Parse the JSONP as JSON data and select the first `--limit` entries in API order. Save the complete response list, source URL, and UTC fetch time in `ranking.json`.
2. Open the DCInside popularity page to initialize cookies. For each selected `package_idx`, POST it and the `ci_t` value from cookie `ci_c` to `https://dccon.dcinside.com/index/package_detail`. Save the full response with its own fetch time in `raw/packages/<package_id>.json`. Iterate the response's `detail` list. The site's displayed sticker count can differ from this list's length.
3. Download each sticker from `https://dcimg5.dcinside.com/dccon.php?no=<path>`, using the detail item's `path`. Preserve the original bytes and determine the extension by decoding the image with Pillow. Record dimensions, transparency, frame count, frame durations, loop value, byte size, and SHA-256. The CDN request uses `Mozilla/5.0` and a DCInside referer. Keep the default 0.35-second request interval and retry behavior when changing HTTP code.

## Files and resume behavior

- `packages.jsonl` contains one row per successful package, including its rank, title, description, creator display name, tags, and source fields. `stickers.jsonl` contains one row per sticker in those packages. Join on `package_id`. Use rows with `image_status == "ok"` for image inputs and resolve `image_path` relative to the dataset directory. For a package-level train/evaluation split, group by `package_id`. SHA-256 identifies byte-identical images.
- Images are stored at `images/<package_id>/<sticker_id>-<source_url_hash>.<ext>`. Image and JSON writes use atomic replacement. The crawler updates the manifests and `summary.json` after each package. Use `errors.jsonl` for failure details and check `summary.json` for `complete` and `metadata_only`. A metadata-only crawl can complete with zero downloaded images.
- Rerun with the same output directory, period, and limit to resume. The crawler reuses the ranking snapshot, cached detail responses, and decodable images. Use a fresh output directory for a fresh ranking or a different limit. The snapshot time describes the ranking fetch. Package details can have later fetch times.

## Running and checking changes

Use Python 3.10 or later and install the crawler dependencies from `requirements.txt`. On the original Mac, the working virtual environment is at `~/.local/share/venvs/dccon`. The former project-local environment had macOS-offloaded files that caused pip to hang.

```bash
python crawl_dccon.py
python crawl_dccon.py --limit 3 --output data/smoke
```

The first command runs the full weekly crawl into `data/week/`. The second runs a live smoke test with three packages. Check `data/smoke/summary.json` for completion and `data/smoke/errors.jsonl` for failures.
