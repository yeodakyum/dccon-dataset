"""Serve DCCON Pilots 1–5 on one local port."""

import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import mimetypes
import os
from pathlib import Path
import sys
from threading import Lock
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
RESULTS = EXPERIMENTS / "pilot2/results"
PILOTS = {1: "pilot1.sticker_gui", 2: "pilot2.compare", 3: "pilot3.app", 4: "pilot4.app"}
EXAMPLES_PER_STICKER = 30
CLUSTER_DISTANCE = 0.20
sys.path.insert(0, str(EXPERIMENTS))

from pilot4.app import EMBEDDING_DIM, EMBEDDING_MODEL, make_encoder


def json_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def cluster_representatives(similarities, merges, limit):
    groups = {index: [index] for index in range(len(similarities))}
    for index, (left, right, distance, _) in enumerate(merges, start=len(similarities)):
        if len(groups) <= limit and distance > CLUSTER_DISTANCE:
            break
        groups[index] = groups.pop(int(left)) + groups.pop(int(right))
    representatives = []
    for members in groups.values():
        members.sort()
        centrality = similarities[members][:, members].mean(axis=1)
        representatives.append((members[int(centrality.argmax())], len(members)))
    return sorted(representatives)


class Comparison:
    def __init__(self, dataset, cache):
        from pilot2.compare import load_annotations

        annotations = RESULTS / "annotations-320.jsonl"
        examples_path = RESULTS / "terra-examples-320.jsonl"
        self.rows, self.images = load_annotations(annotations, dataset)
        examples = json_rows(examples_path)
        by_id = {record["id"]: record["examples"] for record in examples}
        if len(self.rows) != 320 or len(examples) != 320 or by_id.keys() != self.images.keys():
            raise ValueError("320개 주석과 예문의 ID를 확인해 주세요.")
        self.passages = []
        self.groups = []
        self.metadata = []
        for row in self.rows:
            texts = by_id[row["id"]]
            if (len(texts) != EXAMPLES_PER_STICKER
                    or len(set(texts)) != EXAMPLES_PER_STICKER
                    or len(row["kote_emotions"]) != 3):
                raise ValueError(f"예문 30개와 감정 3개가 필요합니다 ({row['id']}).")
            fields = [texts, row["image_text"], [r["intent"] for r in row["reactions"]], [row["visual"]]]
            slices = []
            for field in fields:
                start = len(self.passages)
                self.passages.extend("passage: " + text for text in field)
                slices.append(slice(start, len(self.passages)))
            self.groups.append(slices)
            self.metadata.append({key: row[key] for key in ("id", "title", "package_title", "kote_emotions")})
        digest = sha256(json.dumps([EMBEDDING_MODEL, EMBEDDING_DIM, self.passages], ensure_ascii=False).encode()).hexdigest()[:20]
        self.cache_path = cache / f"passages-{digest}.npy"
        self.encoder = None
        self.vectors = None
        self.lock = Lock()
        self.last_query = None
        self.emotion = None
        self.example_clusters = None
        self.example_representatives = {}

    def score(self, query, example_limit=None):
        with self.lock:
            result = self.score_query(query)
            if example_limit is None:
                return result
            representatives = self.representatives_for(example_limit)
            scores = []
            for group, raw, clusters in zip(self.groups, result["scores"], representatives):
                indices, sizes = zip(*clusters)
                example_scores = [raw["examples"][index] for index in indices]
                best = max(indices, key=raw["examples"].__getitem__)
                scores.append({
                    **raw,
                    "examples": example_scores,
                    "example_cluster_sizes": sizes,
                    "best": self.passages[group[0].start + best].removeprefix("passage: "),
                })
            return {**result, "scores": scores}

    def representatives_for(self, limit):
        if self.example_clusters is None:
            import numpy as np
            from scipy.cluster.hierarchy import linkage

            clusters = []
            upper = np.triu_indices(EXAMPLES_PER_STICKER, k=1)
            for examples, *_ in self.groups:
                vectors = self.vectors[examples]
                similarities = (vectors @ vectors.T).clip(-1, 1)
                np.fill_diagonal(similarities, 1)
                distances = 1 - similarities[upper]
                clusters.append((similarities, linkage(distances, method="complete")))
            self.example_clusters = clusters
        if limit not in self.example_representatives:
            self.example_representatives[limit] = [
                cluster_representatives(similarities, merges, limit)
                for similarities, merges in self.example_clusters
            ]
        return self.example_representatives[limit]

    def score_query(self, query):
        import numpy as np
        from pilot3.app import EmotionClassifier

        if query == self.last_query:
            return self.last_result
        if self.emotion is None:
            self.emotion = EmotionClassifier()
        if self.encoder is None:
            self.encoder = make_encoder()
        if self.vectors is None:
            if self.cache_path.is_file():
                self.vectors = np.load(self.cache_path, allow_pickle=False)
            else:
                self.vectors = self.encoder.encode(
                    self.passages, normalize_embeddings=True, show_progress_bar=True,
                ).astype(np.float32, copy=False)
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                from tempfile import NamedTemporaryFile

                with NamedTemporaryFile(dir=self.cache_path.parent, delete=False) as file:
                    temp_path = Path(file.name)
                    try:
                        np.save(file, self.vectors, allow_pickle=False)
                        file.close()
                        temp_path.replace(self.cache_path)
                    finally:
                        temp_path.unlink(missing_ok=True)
        vector = self.encoder.encode(["query: " + query], normalize_embeddings=True)[0]
        similarities = (self.vectors @ vector.astype(np.float32, copy=False)).clip(-1, 1).round(6)
        emotions = self.emotion.top_emotions(query)
        scores = []
        for examples, captions, reactions, visual in self.groups:
            example_scores = similarities[examples]
            best = examples.start + int(example_scores.argmax())
            scores.append({
                "examples": example_scores.tolist(),
                "caption": float(similarities[captions].max()) if captions.start < captions.stop else None,
                "reaction": float(similarities[reactions].max()) if reactions.start < reactions.stop else None,
                "visual": float(similarities[visual][0]),
                "best": self.passages[best].removeprefix("passage: "),
            })
        self.last_result = {"scores": scores, "emotions": emotions}
        self.last_query = query
        return self.last_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/week")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/pilot5")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--ollama-model", default="qwen3.5:2b")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    local_hf = ROOT / "data/pilot3-runtime/hf"
    if local_hf.is_dir():
        os.environ.setdefault("HF_HOME", str(local_hf))

    comparison = Comparison(args.dataset.resolve(), args.cache)
    page = (HERE / "index.html").read_bytes()
    handlers = {}
    handler_locks = {pilot: Lock() for pilot in PILOTS}

    def handler_for(pilot):
        with handler_locks[pilot]:
            if pilot not in handlers:
                module = importlib.import_module(PILOTS[pilot])
                flags = ["--dataset", str(args.dataset)]
                if pilot in (1, 2, 3):
                    flags += ["--ollama-model", args.ollama_model]
                if pilot == 1:
                    flags += ["--image-search"]
                if pilot == 2:
                    flags += ["--annotations", str(RESULTS / "annotations-320.jsonl")]
                handlers[pilot] = module.make_pilot_handler(module.parse_args(flags))
            return handlers[pilot]

    class Handler(BaseHTTPRequestHandler):
        def send_data(self, body, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data, status=200):
            self.send_data(json.dumps(data, ensure_ascii=False).encode(), "application/json", status)

        def route(self):
            path = urlsplit(self.path).path
            prefix, slash, suffix = self.path.lstrip("/").partition("/")
            pilot = next((n for n in PILOTS if prefix == f"pilot{n}"), None)
            if pilot:
                if not slash:
                    self.send_response(308)
                    self.send_header("Location", path + "/")
                    self.end_headers()
                    return
                if self.command == "GET" and not suffix:
                    filename = {1: "sticker_gui.html", 2: "compare.html"}.get(pilot, "index.html")
                    html = (EXPERIMENTS / prefix / filename).read_bytes()
                    navigation = b'<nav aria-label="Pilots" style="padding: .6rem 1rem; background: #172336">'
                    for number in range(5, 0, -1):
                        current = ' aria-current="page"' if number == pilot else ""
                        navigation += (f'<a href="/pilot{number}/"{current} style="color: white; margin-right: 1rem">'
                                       f'Pilot {number}</a>').encode()
                    body_tag = html.find(b"<body")
                    body = html.index(b">", body_tag) + 1 if body_tag >= 0 else html.index(b"<main>")
                    html = html[:body] + navigation + b"</nav>" + html[body:]
                    self.send_data(html, "text/html; charset=utf-8")
                    return
                try:
                    handler = handler_for(pilot)
                except (ValueError, OSError, RuntimeError, ImportError) as exc:
                    self.send_json({"error": str(exc)}, 503)
                    return
                self.path = "/" + suffix
                delegate = object.__new__(handler)
                delegate.__dict__ = self.__dict__
                getattr(delegate, "do_" + self.command)()
                return
            if self.command == "POST":
                self.search(path)
            elif path in ("/", "/pilot5/"):
                self.send_data(page, "text/html; charset=utf-8")
            elif path == "/pilot5":
                self.send_response(308)
                self.send_header("Location", "/pilot5/")
                self.end_headers()
            elif path == "/pilot5/api/corpus":
                self.send_json(comparison.metadata)
            elif path.startswith("/pilot5/sticker/"):
                image = comparison.images.get(unquote(path.removeprefix("/pilot5/sticker/")))
                if image is None:
                    self.send_error(404)
                    return
                self.send_data(image.read_bytes(), mimetypes.guess_type(image.name)[0] or "application/octet-stream")
            else:
                self.send_error(404)

        do_GET = do_POST = route

        def search(self, path):
            if path != "/pilot5/api/search":
                self.send_error(404)
                return
            if (self.headers.get_content_type() != "application/json"
                    or self.headers.get("Origin") != f"http://127.0.0.1:{self.server.server_port}"):
                self.send_json({"error": "Pilot 5 화면에서 요청해 주세요."}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4096:
                    raise ValueError
                request = json.loads(self.rfile.read(length))
                query = request["query"].strip()
                if not 0 < len(query) <= 300:
                    raise ValueError
            except (ValueError, KeyError, TypeError, AttributeError):
                self.send_json({"error": "메시지를 1~300자로 입력해 주세요."}, 400)
                return
            example_limit = request.get("example_limit")
            if example_limit is not None and (
                type(example_limit) is not int or not 1 <= example_limit <= EXAMPLES_PER_STICKER
            ):
                self.send_json({"error": "콘당 최대 예문 수는 1~30 사이의 정수여야 합니다."}, 400)
                return
            try:
                data = comparison.score(query, example_limit)
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                self.send_json({"error": str(exc)}, 503)
                return
            self.send_json(data)

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Pilot 1–5 · http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
