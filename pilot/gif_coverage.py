"""Compare GIF frame samples with the decoded loop using Pillow."""
import json
import random
from bisect import bisect_right
from collections import defaultdict
from itertools import accumulate
from pathlib import Path
from statistics import median

from PIL import Image, ImageChops, ImageStat

DATA = Path(__file__).resolve().parent.parent / "data" / "week"


def sample_gifs():
    packages = defaultdict(list)
    seen = set()
    with (DATA / "stickers.jsonl").open(encoding="utf-8") as manifest:
        for line in manifest:
            row = json.loads(line)
            if row["image_status"] != "ok" or row["format"] != "GIF" or not row["animated"]:
                continue
            if row["sha256"] not in seen:
                packages[row["package_id"]].append(row)
                seen.add(row["sha256"])
    rng = random.Random(1113)
    return [rng.choice(packages[key]) for key in rng.sample(sorted(packages), 36)]


def coverage_errors(row):
    with Image.open(DATA / row["image_path"]) as source:
        frame_count = source.n_frames
        delays = row["frame_durations_ms"]
        if len(delays) != frame_count or any(delay <= 0 for delay in delays):
            delays = [1] * frame_count
        frames = []
        background = Image.new("RGBA", source.size, (242, 242, 242, 255))
        for index in range(frame_count):
            source.seek(index)
            frame = Image.alpha_composite(background, source.convert("RGBA"))
            frames.append(frame.convert("RGB").resize((24, 24)))

    ends = list(accumulate(delays))
    distances = [[0.0] * frame_count for _ in frames]
    for i, frame in enumerate(frames):
        for j in range(i):
            rgb_error = ImageStat.Stat(ImageChops.difference(frame, frames[j])).mean
            distances[i][j] = distances[j][i] = sum(rgb_error) / (3 * 255)

    def time_sample(count):
        return {bisect_right(ends, ends[-1] * (i / count)) for i in range(count)}

    def loss(indices):
        return sum(min(distance[j] for j in indices) * delay
                   for distance, delay in zip(distances, delays)) / ends[-1]

    selected = [0]
    while len(selected) < min(4, frame_count):
        remaining = (index for index in range(frame_count) if index not in selected)
        selected.append(min(remaining, key=lambda index: loss([*selected, index])))

    return {
        "first": loss([0]),
        "four_by_time": loss(time_sample(4)),
        "eight_by_time": loss(time_sample(8)),
        "four_by_coverage": loss(selected),
    }


def main():
    results = [coverage_errors(row) for row in sample_gifs()]
    print(f"{len(results)} byte-distinct GIFs, one per package")
    for method in results[0]:
        print(f"{method:18} {median(row[method] for row in results):.2%}")


if __name__ == "__main__":
    main()
