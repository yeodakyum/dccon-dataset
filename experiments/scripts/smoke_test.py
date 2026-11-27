"""Check that a server venv can load a small model and run one step on the GPU.

    python experiments/scripts/smoke_test.py main    # sentence-transformers + FAISS + BM25
    python experiments/scripts/smoke_test.py vllm    # vLLM structured JSON output
    python experiments/scripts/smoke_test.py pylate  # KURE-v2 MaxSim
"""
import json
import sys
import time

QUERY = "친구가 마감 5분 전에 파일이 날아갔대. 위로가 필요한 상황"
DOCS = ["A: 나 장학금 받았어! B: 미쳤다 ㅋㅋ 축하해",
        "A: 과제 파일 날아갔어 마감 5분 남았는데 B: 헐 괜찮아? 내가 도와줄게"]


def report(name, started):
    print(f"[ok] {name} ({time.perf_counter() - started:.1f}s)")


def check_main():
    import bm25s
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"gpus {torch.cuda.device_count()}, {torch.cuda.get_device_name(0)}")
    started = time.perf_counter()
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda",
                                model_kwargs={"torch_dtype": torch.float16})
    query = model.encode([QUERY], prompt_name="query", normalize_embeddings=True)
    docs = model.encode(DOCS, normalize_embeddings=True)
    index = faiss.IndexFlatIP(docs.shape[1])
    index.add(np.asarray(docs, dtype="float32"))
    scores, ids = index.search(np.asarray(query, dtype="float32"), 2)
    assert ids[0][0] == 1, f"expected the file-loss dialogue first, got {ids[0]} {scores[0]}"
    report("Qwen3-Embedding-0.6B + FAISS IndexFlatIP", started)

    started = time.perf_counter()
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(DOCS, stopwords=None, show_progress=False), show_progress=False)
    retriever.retrieve(bm25s.tokenize([QUERY], stopwords=None, show_progress=False), k=2,
                       show_progress=False)
    report("bm25s", started)


def check_vllm():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = {"type": "object",
              "properties": {"emotion": {"type": "string"}, "intent": {"type": "string"},
                             "tone": {"type": "string"}, "summary": {"type": "string"}},
              "required": ["emotion", "intent", "tone", "summary"]}
    started = time.perf_counter()
    llm = LLM("Qwen/Qwen3.5-0.8B", dtype="float16", max_model_len=2048,
              gpu_memory_utilization=0.4, limit_mm_per_prompt={"image": 0, "video": 0})
    params = SamplingParams(temperature=0, max_tokens=128,
                            structured_outputs=StructuredOutputsParams(json=schema))
    messages = [{"role": "user", "content": f"다음 답장에 필요한 감정, 의도, 톤을 JSON으로: {DOCS[1]}"}]
    output = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
    text = output[0].outputs[0].text
    print(json.loads(text))
    report("vLLM Qwen3.5-0.8B structured JSON", started)


def check_pylate():
    from pylate import models, rank

    started = time.perf_counter()
    model = models.ColBERT("nlpai-lab/KURE-v2", device="cuda")
    query = model.encode([QUERY], is_query=True)
    docs = model.encode(DOCS, is_query=False)
    ranked = rank.rerank(documents_ids=[[0, 1]], queries_embeddings=query,
                         documents_embeddings=[docs])
    assert ranked[0][0]["id"] == 1, ranked
    report("PyLate KURE-v2 MaxSim", started)


if __name__ == "__main__":
    {"main": check_main, "vllm": check_vllm, "pylate": check_pylate}[sys.argv[1]]()
