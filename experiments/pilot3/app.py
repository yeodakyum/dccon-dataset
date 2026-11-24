"""Run the combined local DCCON pilot."""

import argparse
from collections import deque
from functools import partial
from http.server import ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import ssl
import subprocess
import sys
from threading import Lock, Thread
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import truststore

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path[:0] = [str(EXPERIMENTS / folder) for folder in ("pilot1", "pilot2", "tagging")]

import compare as mixed
from image_search import ImageSearch
from search_stickers import DEFAULT_PILOT, MODEL, index_passages, load_stickers, search
from sticker_gui import fuse_results, rewrite_query
from tagging_gui import MODEL as GROQ_MODEL, make_handler

KOTE_MODEL = "searle-j/kote_for_easygoing_people"
EMOTION_SEARCH = "examples_emotion"
TERRA_SEARCH = "terra_examples"
PILOT2_RESULTS = EXPERIMENTS / "pilot2/results"
EMOTION_WEIGHT = 0.10

MODELS = {"e5": MODEL, "kure": "nlpai-lab/KURE-v3"}
CORPORA = {
    "synthetic": "합성 주석 · Pilot 1",
    "mixed320": "이미지 주석 · Pilot 2 (320개 표본)",
    "mixed32": "이미지 주석 · Pilot 2 (32개 표본)",
}
METHODS = {
    "examples": "답장 예문",
    "situations": "사용 상황",
    "fused": "예문 + 상황 + 이미지 문구 일치",
    "context": "최근 대화 + 사용 상황",
    "descriptions": "이미지 설명",
    "image": "SigLIP 2 · 원본 이미지",
    "intent": "SLM 반응 의도 → 사용 상황",
    "visual": "SLM 표정·동작 → SigLIP 2",
    EMOTION_SEARCH: "예문 + 감정",
    TERRA_SEARCH: "Terra 예문 30개 유사도 합산",
    "faiss": "대화 원문 → FAISS",
    "rewrite": "SLM 반응 태그·묘사 → FAISS",
    "choose": "FAISS 상위 10개 → SLM 선택",
}
MIXED_SAMPLES = {"mixed320": ("pilot2_320", 160), "mixed32": ("pilot2", 16)}
SYNTHETIC_METHODS = ("examples", "situations", "fused", "context", "descriptions", "image", "intent", "visual")
MIXED_METHODS = ("faiss", "rewrite", "choose")


REPORTS = {
    "worklog": ("통합 작업 기록", ROOT / "docs/2026-11-23-to-24-worklog.md"),
    "pilot1": ("Pilot 1 검색 실험", EXPERIMENTS / "pilot1/2026-11-23-search-experiments.md"),
    "pilot2": ("Pilot 2 실험 및 선행 연구", EXPERIMENTS / "pilot2/EXPERIMENTS.md"),
    "instructions": ("Pilot 2 주석 배치와 실행 방법", EXPERIMENTS / "pilot2/README.md"),
    "ab": ("기존 E5와 한국어 튜닝 E5 비교 결과", EXPERIMENTS / "pilot2/results/embedding-ab.json"),
}


SLM_METHODS = {"intent", "visual", "rewrite", "choose"}


def complete_groq(api_key, messages, *, json_mode=False):
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "reasoning_effort": "none",
        "temperature": 0.2,
        "max_completion_tokens": 256,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        tls = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        with urlopen(request, timeout=60, context=tls) as response:
            content = json.load(response)["choices"][0]["message"]["content"]
    except HTTPError as exc:
        exc.close()
        raise ValueError(f"Groq API 요청에 실패했습니다 (HTTP {exc.code}).") from None
    except (URLError, TimeoutError, OSError):
        raise ValueError("Groq 서버 응답을 받지 못했습니다.") from None
    except (ValueError, KeyError, IndexError, TypeError):
        raise ValueError("Groq 응답을 읽지 못했습니다.") from None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq가 빈 응답을 반환했습니다.")
    return content.strip()


def ask_groq(api_key, model, prompt, data, *, context_size, output_format="json"):
    return complete_groq(api_key, [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
    ], json_mode=True)


class EmotionClassifier:
    def __init__(self):
        self.lock = Lock()
        self.emotion_model = None

    def top_emotions(self, draft):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        with self.lock, torch.inference_mode():
            if self.emotion_model is None:
                tokenizer = AutoTokenizer.from_pretrained(KOTE_MODEL)
                classifier = AutoModelForSequenceClassification.from_pretrained(KOTE_MODEL).eval()
                self.emotion_model = tokenizer, classifier
            tokenizer, classifier = self.emotion_model
            inputs = tokenizer(draft, truncation=True, max_length=512, return_tensors="pt")
            scores = classifier(**inputs).logits[0].sigmoid().tolist()
            labels = classifier.config.id2label
        ranked = sorted(
            ((labels[i], score) for i, score in enumerate(scores) if labels[i] != "없음"),
            key=lambda item: item[1], reverse=True,
        )
        return [{"label": label, "score": score} for label, score in ranked[:3]]


class Workbench:
    def __init__(self, dataset, ollama_model, synthetic, annotations_root):
        self.dataset = dataset
        self.ollama_model = ollama_model
        self.annotations_root = annotations_root
        self.lock = Lock()
        self.encoders = {}
        self.indexes = {}
        self.image_index = None
        self.emotion_classifier = EmotionClassifier()
        self.rows = {"synthetic": load_stickers(synthetic, include_examples=True)}
        self.examples_by_sticker = {
            (str(row["package_id"]), str(row["sticker_id"])): row["examples"]
            for row in self.rows["synthetic"]
        }
        self.images = {}
        for row in self.rows["synthetic"]:
            path = (dataset / row["image_path"]).resolve()
            if path.is_relative_to(dataset) and path.is_file():
                self.images[f'{row["package_id"]}-{row["sticker_id"]}'] = path
        for corpus in MIXED_SAMPLES:
            self.refresh_corpus(corpus)

        self.passages = {}
        for field in ("examples", "situations"):
            self.passages[field] = [
                {"package_id": row["package_id"], "sticker_id": row["sticker_id"],
                 "image_path": row["image_path"], f"{field[:-1]}_text": passage,
                 "embedding_input": "passage: " + passage}
                for row in self.rows["synthetic"] for passage in row[field]
            ]
        examples_path = PILOT2_RESULTS / "terra-examples-320.jsonl"
        if examples_path.is_file():
            self.load_terra_passages(examples_path)

    def load_terra_passages(self, path):
        rows, images = mixed.load_annotations(PILOT2_RESULTS / "annotations-320.jsonl", self.dataset)
        examples = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            texts = record["examples"]
            if (record["id"] in examples or not isinstance(texts, list) or len(texts) != 30
                    or any(not isinstance(text, str) or not text.strip() or len(text.splitlines()) != 1
                           for text in texts)
                    or len({text.strip() for text in texts}) != 30):
                raise ValueError(f"Invalid Terra examples: {record['id']}")
            examples[record["id"]] = texts
        if examples.keys() != images.keys():
            raise ValueError("Terra examples must cover the Pilot 2 sample.")
        self.rows.setdefault("mixed320", rows)
        self.images.update(images)
        self.passages[TERRA_SEARCH] = [
            {**row, "example_text": text, "embedding_input": "passage: " + text}
            for row in rows for text in examples[row["id"]]
        ]

    def corpus_methods(self, corpus):
        if corpus == "synthetic":
            return SYNTHETIC_METHODS
        if corpus == "mixed320":
            methods = (*MIXED_METHODS, EMOTION_SEARCH)
            return (*methods, TERRA_SEARCH) if TERRA_SEARCH in self.passages else methods
        return MIXED_METHODS

    def refresh_corpus(self, corpus):
        directory, _ = MIXED_SAMPLES[corpus]
        path = self.annotations_root / directory / "annotations.jsonl"
        if path.is_file():
            rows, images = mixed.load_annotations(path, self.dataset)
            if rows:
                with self.lock:
                    self.rows = {**self.rows, corpus: rows}
                    self.images.update(images)
                    for model in MODELS:
                        self.indexes.pop((model, corpus), None)
                        if corpus == "mixed320":
                            self.indexes.pop((model, EMOTION_SEARCH), None)

    def config(self):
        rows_by_corpus = self.rows
        return {
            "models": MODELS, "ollama": self.ollama_model, "groq": GROQ_MODEL,
            "slm_methods": sorted(SLM_METHODS),
            "methods": {corpus: {key: METHODS[key] for key in self.corpus_methods(corpus)}
                        for corpus in rows_by_corpus},
            "corpora": {
                key: {"label": CORPORA[key], "count": len(rows)}
                for key, rows in rows_by_corpus.items()
            },
            "reports": {key: title for key, (title, path) in REPORTS.items() if path.is_file()},
        }

    def lookup(self, model, collection, query, limit=5):
        from sentence_transformers import SentenceTransformer

        with self.lock:
            grouped = collection in self.passages or collection == EMOTION_SEARCH
            key = model, collection
            if model not in self.encoders:
                device = "cpu"
                options = {}
                if model == "kure":
                    import torch

                    if torch.backends.mps.is_available():
                        device = "mps"
                    options = {
                        "truncate_dim": 1024,
                        "model_kwargs": {
                            "attn_implementation": "sdpa",
                            "dtype": torch.float16 if device == "mps" else torch.float32,
                        },
                    }
                self.encoders[model] = SentenceTransformer(MODELS[model], device=device, **options)
            encoder = self.encoders[model]
            if key not in self.indexes:
                if collection == EMOTION_SEARCH:
                    rows = self.emotion_passages()
                elif grouped:
                    rows = self.passages[collection]
                else:
                    rows = self.rows[collection]
                self.indexes[key] = index_passages(encoder, rows), rows
            index, rows = self.indexes[key]
            return search(encoder, index, rows, [query], limit,
                          group_stickers=grouped, sum_stickers=collection == TERRA_SEARCH)["results"][0]

    def emotion_passages(self):
        passages = []
        for row in self.rows["mixed320"]:
            key = str(row["package_id"]), str(row["sticker_id"])
            examples = self.examples_by_sticker.get(key)
            if examples:
                texts, field = examples, "example_text"
            elif row["image_text"]:
                texts, field = row["image_text"], "caption_text"
            else:
                texts = [item["intent"] for item in row["reactions"]] or [row["visual"]]
                field = "annotation_text"
            for text in texts:
                passages.append({**row, field: text, "embedding_input": "passage: " + text})
        return passages

    def search_examples_emotion(self, model, draft):
        emotions = self.emotion_classifier.top_emotions(draft)
        result = self.lookup(model, EMOTION_SEARCH, draft, limit=len(self.rows["mixed320"]))
        for match in result["matches"]:
            tags = set(match.get("kote_emotions", []))
            matched = [item for item in emotions if item["label"] in tags]
            bonus = EMOTION_WEIGHT * sum(item["score"] for item in matched) / len(emotions)
            match["matched_emotions"] = [item["label"] for item in matched]
            match["combined_score"] = round(match["similarity"] + bonus, 6)
        result["matches"].sort(key=lambda match: -match["combined_score"])
        result["matches"] = result["matches"][:5]
        for rank, match in enumerate(result["matches"], 1):
            match["rank"] = rank
        return {**result, "emotions": emotions}

    def lookup_images(self, query):
        with self.lock:
            if self.image_index is None:
                self.image_index = ImageSearch(self.dataset, self.rows["synthetic"])
            return self.image_index.search([query])[0]

    def run(self, model, corpus, method, draft, context, manual, *, groq_key=""):
        started = perf_counter()
        ask = partial(ask_groq, groq_key) if groq_key else mixed.ask_ollama
        complete = partial(complete_groq, groq_key) if groq_key else None
        if corpus == "synthetic":
            if method in {"examples", "situations", "descriptions"}:
                result = self.lookup(model, "synthetic" if method == "descriptions" else method,
                                     manual or draft)
            elif method == "fused":
                query = manual or draft
                example = self.lookup(model, "examples", query, 20)
                situation = self.lookup(model, "situations", query, 20)
                result = fuse_results(query, example, situation, self.rows[corpus])
            elif method == "context":
                query = f"최근 대화: {context}\n내가 보낼 말: {draft}" if context else draft
                result = self.lookup(model, "situations", query)
            elif method == "image":
                result = self.lookup_images(manual or draft)
            else:
                visual = method == "visual"
                generated = rewrite_query(self.ollama_model, context, draft, visual=visual, complete=complete)
                result = (self.lookup_images(generated) if visual
                          else self.lookup(model, "situations", generated))
        else:
            query = f"입력할 대화: {draft}\n이전 대화: {context}" if context else draft
            if method == EMOTION_SEARCH:
                result = self.search_examples_emotion(model, draft)
            elif method == TERRA_SEARCH:
                result = self.lookup(model, TERRA_SEARCH, query)
            elif method == "rewrite":
                generated = mixed.rewrite_query(self.ollama_model, context, draft, ask=ask)
                result = self.lookup(model, corpus, generated) if generated else {"query": "", "matches": []}
            elif method == "choose":
                candidates = self.lookup(model, corpus, query, mixed.CANDIDATE_COUNT)
                selected = mixed.choose_stickers(self.ollama_model, context, draft, candidates["matches"], ask=ask)
                result = {**candidates, "matches": selected, "candidates": candidates["matches"]}
            else:
                result = self.lookup(model, corpus, query)
        return {**result, "seconds": round(perf_counter() - started, 3)}


class BatchTagging:
    def __init__(self, workbench):
        self.workbench = workbench
        self.lock = Lock()
        self.lines = deque(maxlen=60)
        self.state = "idle"

    def status(self):
        with self.lock:
            return {"state": self.state, "lines": list(self.lines)}

    def start(self, corpus, key):
        with self.lock:
            if self.state == "running":
                raise ValueError("태깅 배치가 이미 실행 중입니다.")
            self.lines.clear()
            self.state = "running"
            Thread(target=self.run, args=(corpus, key), daemon=True).start()

    def run(self, corpus, key):
        directory, per_kind = MIXED_SAMPLES[corpus]
        command = [
            sys.executable, str(EXPERIMENTS / "tagging/tag_stickers.py"),
            "--dataset", str(self.workbench.dataset),
            "--output", str(self.workbench.annotations_root / directory),
            "--per-kind", str(per_kind), "--workers", "4", "--run",
        ]
        state = "failed"
        try:
            with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace", env={**os.environ, "GROQ_API_KEY": key}) as process:
                for line in process.stdout:
                    with self.lock:
                        self.lines.append(line.rstrip().replace(key, "[redacted]"))
                if process.wait() == 0:
                    self.workbench.refresh_corpus(corpus)
                    state = "done"
        except (OSError, ValueError, KeyError) as exc:
            with self.lock:
                self.lines.append(str(exc).replace(key, "[redacted]"))
        finally:
            with self.lock:
                self.state = state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/week")
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--annotations-root", type=Path, default=ROOT / "data")
    parser.add_argument("--ollama-model", default="qwen3.5:2b")
    parser.add_argument("--port", type=int, default=8768)
    return parser.parse_args(argv)


def make_pilot_handler(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    workbench = Workbench(args.dataset.resolve(), args.ollama_model,
                          args.synthetic, args.annotations_root)
    batch = BatchTagging(workbench)
    tag_handler = make_handler(args.dataset)
    page = (HERE / "index.html").read_bytes()

    class Handler(tag_handler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.send_data(page, "text/html; charset=utf-8")
            elif path == "/api/workbench":
                self.send_json(workbench.config())
            elif path == "/api/batch":
                self.send_json(batch.status())
            elif path == "/tagging":
                self.send_response(302)
                self.send_header("Location", "tagging/")
                self.end_headers()
            elif path.startswith("/tagging/"):
                self.path = path.removeprefix("/tagging")
                super().do_GET()
            elif path.startswith("/sticker/") and (source := workbench.images.get(path[9:])):
                try:
                    self.send_data(source.read_bytes(),
                                   mimetypes.guess_type(source.name)[0] or "application/octet-stream")
                except OSError:
                    self.send_error(404)
            elif path.startswith("/reports/") and (entry := REPORTS.get(path[9:])):
                try:
                    self.send_data(entry[1].read_bytes(), "text/plain; charset=utf-8")
                except OSError:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/tagging/api/tag":
                self.path = "/api/tag"
                super().do_POST()
                return
            if self.path not in {"/api/search", "/api/batch"}:
                self.send_error(404)
                return
            if (self.headers.get_content_type() != "application/json"
                    or self.headers.get("Origin") != f"http://127.0.0.1:{self.server.server_port}"):
                self.send_json({"error": "Pilot 3 화면에서 요청해 주세요."}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 24_000:
                    raise ValueError
                data = json.loads(self.rfile.read(length))
                if self.path == "/api/batch":
                    if not isinstance(data, dict):
                        raise ValueError
                    corpus = data.get("corpus")
                    key = data.get("api_key", "")
                    if corpus not in MIXED_SAMPLES or not isinstance(key, str):
                        raise ValueError
                    key = key.strip() or os.environ.get("GROQ_API_KEY", "").strip()
                    if not key or any(not 33 <= ord(char) <= 126 for char in key):
                        self.send_json({"error": "Groq API 키를 확인해 주세요."}, 400)
                        return
                    try:
                        batch.start(corpus, key)
                    except ValueError as exc:
                        self.send_json({"error": str(exc)}, 409)
                        return
                    self.send_json(batch.status(), 202)
                    return
                slm = data.get("slm", "ollama")
                model, corpus, method = (data[key] for key in ("model", "corpus", "method"))
                draft, context, manual = (data[key].strip() for key in ("draft", "context", "manual"))
                if (slm not in ("ollama", "groq") or model not in MODELS or corpus not in workbench.rows or method not in workbench.corpus_methods(corpus)
                        or not 0 < len(draft) <= 300 or len(context) > 1000 or len(manual) > 300):
                    raise ValueError
            except (AttributeError, KeyError, TypeError, ValueError):
                self.send_json({"error": "검색 설정과 입력 길이를 확인해 주세요."}, 400)
                return
            groq_key = ""
            if slm == "groq" and method in SLM_METHODS:
                key = data.get("api_key", "")
                if not isinstance(key, str):
                    self.send_json({"error": "Groq API 키를 확인해 주세요."}, 400)
                    return
                groq_key = key.strip() or os.environ.get("GROQ_API_KEY", "").strip()
                if not groq_key or any(not 33 <= ord(char) <= 126 for char in groq_key):
                    self.send_json({"error": "Groq API 키를 확인해 주세요."}, 400)
                    return
            try:
                self.send_json(workbench.run(model, corpus, method, draft, context, manual,
                                             groq_key=groq_key))
            except (URLError, TimeoutError, ValueError, KeyError, TypeError, OSError,
                    RuntimeError, ImportError) as exc:
                self.send_json({"error": str(exc)}, 502)

    return Handler


def main():
    args = parse_args()
    Handler = make_pilot_handler(args)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Pilot 3 · http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
