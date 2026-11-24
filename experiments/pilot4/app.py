"""Generate visual chat examples and rank every sticker in the 32-item sample."""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path[:0] = [str(EXPERIMENTS / name) for name in ("pilot1", "pilot2", "tagging")]

from compare import load_annotations
from search_stickers import index_passages
from tagging_gui import MODEL, validate_examples

STICKER_COUNT = 32
EMBEDDING_MODEL = "nlpai-lab/KURE-v3"
PILOT2_SAMPLE = EXPERIMENTS / "pilot2/results/annotations-32.jsonl"
SUBAGENT_EXAMPLES = HERE / "results/subagent-examples-32.jsonl"
SOURCE_LABELS = {"subagent": "서브에이전트 예문", "qwen": "Qwen 비전 예문"}


class Corpus:
    def __init__(self, dataset, annotations, examples=None):
        self.dataset = dataset
        self.annotations = annotations
        self.examples = examples
        self.index = None
        self.signature = None
        self.rows = {}
        self.images = {}

    def load(self):
        path = self.examples or self.annotations
        try:
            stat = path.stat()
        except FileNotFoundError:
            self.signature = self.index = None
            self.rows = {}
            self.images = {}
            return
        signature = stat.st_ino, stat.st_mtime_ns
        if signature == self.signature:
            return
        rows, images = load_annotations(self.annotations, self.dataset)
        if len(rows) != STICKER_COUNT or sum(row["is_gif"] for row in rows) != STICKER_COUNT // 2:
            raise ValueError(f"정지 이미지 16개와 GIF 16개가 필요합니다. 현재 {len(rows)}개입니다.")
        if self.examples is None:
            if any(row["model"] != MODEL for row in rows):
                raise ValueError("Qwen 비전 태깅으로 생성된 예문이 필요합니다.")
            examples = {row["id"]: row.get("examples") for row in rows}
        else:
            with path.open(encoding="utf-8") as file:
                records = [json.loads(line) for line in file]
            examples = {record["id"]: record["examples"] for record in records}
            if len(records) != STICKER_COUNT or examples.keys() != images.keys():
                raise ValueError("서브에이전트 예문 파일에 표본의 콘 32개가 각각 한 번씩 있어야 합니다.")
        for row in rows:
            row["examples"] = validate_examples(examples[row["id"]])
        self.rows = {row["id"]: row for row in rows}
        self.images = images
        self.index = None
        self.signature = signature

    def search(self, encoder, query, cutoff=None):
        if self.index is None:
            self.passages = [(key, text) for key, row in self.rows.items() for text in row["examples"]]
            self.index = index_passages(encoder, [
                {"embedding_input": "passage: " + text} for _, text in self.passages
            ])
            self.last_query = None
        if self.last_query != query:
            vector = encoder.encode(["query: " + query], normalize_embeddings=True)
            self.query_results = self.index.search(vector, len(self.passages))
            self.last_query = query
        scores, positions = self.query_results
        matches = {}
        for score, position in zip(scores[0], positions[0]):
            key, example = self.passages[position]
            if key not in matches:
                row = self.rows[key]
                matches[key] = {
                    "id": key, "title": row["title"], "package_title": row["package_title"],
                    "best_example": None, "examples": row["examples"], "score": 0.0, "included": 0,
                }
            score = float(score)
            if cutoff is None or score >= cutoff:
                match = matches[key]
                if match["included"] == 0:
                    match["best_example"] = example
                match["score"] += score
                match["included"] += 1
        ranking = sorted(matches.values(), key=lambda match: (-match["score"], match["id"]))
        return [{**match, "rank": rank, "score": round(match["score"], 6)}
                for rank, match in enumerate(ranking, 1)]


class Pilot:
    def __init__(self, dataset, output):
        self.dataset = dataset.resolve()
        self.output = output.resolve()
        self.corpora = {
            "subagent": Corpus(self.dataset, PILOT2_SAMPLE, SUBAGENT_EXAMPLES),
            "qwen": Corpus(self.dataset, self.output / "annotations.jsonl"),
        }
        self.batch_lock = Lock()
        self.search_lock = Lock()
        self.state = "idle"
        self.lines = deque(maxlen=10)
        self.encoder = None

    def status(self):
        with self.batch_lock:
            state, lines = self.state, list(self.lines)
        sources = {}
        errors = {}
        with self.search_lock:
            for source, corpus in self.corpora.items():
                if source == "qwen" and state == "running":
                    continue
                try:
                    corpus.load()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors[source] = str(exc)
                else:
                    if corpus.rows:
                        sources[source] = SOURCE_LABELS[source]
        if "qwen" in errors:
            state = "failed"
            lines.append(errors["qwen"])
        if "qwen" in sources:
            state = "ready"
        return {"state": state, "lines": lines, "sources": sources, "source_errors": errors}

    def start(self, key):
        with self.batch_lock:
            if self.state == "running":
                raise ValueError("배치가 이미 실행 중입니다.")
            self.lines.clear()
            self.state = "running"
            Thread(target=self.run, args=(key,), daemon=True).start()

    def run(self, key):
        command = [
            sys.executable, str(EXPERIMENTS / "tagging/tag_stickers.py"),
            "--dataset", str(self.dataset), "--output", str(self.output), "--per-kind", "16",
            "--workers", "3", "--examples", "--run",
        ]
        state = "failed"
        try:
            with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace", env={**os.environ, "GROQ_API_KEY": key}) as process:
                for line in process.stdout:
                    with self.batch_lock:
                        self.lines.append(line.rstrip().replace(key, "[redacted]"))
                if process.wait() == 0:
                    state = "idle"
        except OSError as exc:
            with self.batch_lock:
                self.lines.append(str(exc).replace(key, "[redacted]"))
        finally:
            with self.batch_lock:
                self.state = state

    def search(self, query, source="qwen", cutoff=None):
        from sentence_transformers import SentenceTransformer

        with self.search_lock:
            corpus = self.corpora[source]
            corpus.load()
            if not corpus.rows:
                raise ValueError("이 출처의 예문 32개 묶음이 아직 준비되지 않았습니다.")
            if self.encoder is None:
                import torch

                device = "mps" if torch.backends.mps.is_available() else "cpu"
                self.encoder = SentenceTransformer(
                    EMBEDDING_MODEL, device=device, truncate_dim=1024,
                    model_kwargs={
                        "attn_implementation": "sdpa",
                        "dtype": torch.float16 if device == "mps" else torch.float32,
                    },
                )
            return corpus.search(self.encoder, query, cutoff)

    def image(self, suffix):
        source, separator, key = suffix.partition("/")
        if not separator:
            source, key = "qwen", source
        corpus = self.corpora.get(source)
        return corpus.images.get(key) if corpus else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/week")
    parser.add_argument("--output", type=Path, default=ROOT / "data/pilot4")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    pilot = Pilot(args.dataset, args.output)
    page = (HERE / "index.html").read_bytes()
    env_key = os.environ.get("GROQ_API_KEY", "").strip()

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

        def do_GET(self):
            if self.path == "/":
                self.send_data(page, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self.send_json({**pilot.status(), "configured": bool(env_key), "model": MODEL})
            elif self.path.startswith("/sticker/") and (image := pilot.image(self.path[9:])):
                try:
                    body = image.read_bytes()
                except OSError:
                    self.send_error(404)
                    return
                self.send_data(body, mimetypes.guess_type(image.name)[0] or "application/octet-stream")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path not in ("/api/generate", "/api/search"):
                self.send_error(404)
                return
            if (self.headers.get_content_type() != "application/json"
                    or self.headers.get("Origin") != f"http://127.0.0.1:{self.server.server_port}"):
                self.send_json({"error": "Pilot 4 화면에서 요청해 주세요."}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4096:
                    raise ValueError
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError
            except (ValueError, TypeError):
                self.send_json({"error": "요청 형식을 확인해 주세요."}, 400)
                return
            if self.path == "/api/generate":
                key = payload.get("api_key", "")
                if isinstance(key, str):
                    key = key.strip() or env_key
                if not isinstance(key, str) or not key or any(not 33 <= ord(char) <= 126 for char in key):
                    self.send_json({"error": "Groq API 키를 입력해 주세요."}, 400)
                    return
                try:
                    pilot.start(key)
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, 409)
                    return
                self.send_json({"state": "running"}, 202)
                return
            query = payload.get("query")
            if not isinstance(query, str) or not 0 < len(query.strip()) <= 300:
                self.send_json({"error": "검색할 메시지를 1~300자로 입력해 주세요."}, 400)
                return
            source = payload.get("source", "qwen")
            if not isinstance(source, str) or source not in SOURCE_LABELS:
                self.send_json({"error": "예문 출처를 확인해 주세요."}, 400)
                return
            cutoff = payload.get("cutoff")
            if cutoff is not None and (type(cutoff) not in (int, float) or not -1 <= cutoff <= 1):
                self.send_json({"error": "컷오프는 -1부터 1 사이의 숫자여야 합니다."}, 400)
                return
            try:
                matches = pilot.search(query.strip(), source, cutoff)
                self.send_json({"matches": matches, "source": source})
            except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as exc:
                self.send_json({"error": str(exc)}, 422)

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Pilot 4 · http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
