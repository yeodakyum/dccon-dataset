"""Search the synthetic DCCON pilot with multilingual E5 and FAISS."""

import argparse
import json
from pathlib import Path

MODEL = "intfloat/multilingual-e5-small"
DEFAULT_PILOT = Path(__file__).resolve().parent / "synthetic" / "batch_100"


def load_stickers(directory, include_examples=False):
    stickers = []
    seen = set()
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["package_id"], record["sticker_id"])
        if key in seen:
            raise ValueError(f"Duplicate sticker: {key}")
        seen.add(key)
        tags = dict.fromkeys(
            tag for example in record["examples"] for tag in example["usage_tags"]
        )
        parts = [record["image_description"]]
        image_text = record.get("image_text")
        if image_text:
            parts.append("이미지 문구: " + ", ".join(image_text))
        if tags:
            parts.append("사용 상황: " + ", ".join(tags))
        description = " ".join(parts)
        sticker = {
            "sticker_id": record["sticker_id"],
            "package_id": record["package_id"],
            "image_path": record["image_path"],
            "description": description,
            "embedding_input": "passage: " + description,
        }
        if include_examples:
            example_texts = [
                message["text"].strip()
                for example in record["examples"]
                if (message := example.get("sticker_message", {})).get("send_sticker") is True
                and isinstance(message.get("text"), str)
                and message["text"].strip()
            ]
            sticker["examples"] = list(dict.fromkeys(example_texts))
        stickers.append(sticker)
    if not stickers:
        raise ValueError(f"No sticker annotations in {directory}")
    return stickers


def build_index(directory, include_examples=False):
    from sentence_transformers import SentenceTransformer
    stickers = load_stickers(directory, include_examples)
    model = SentenceTransformer(MODEL, device="cpu")
    return model, index_passages(model, stickers), stickers


def index_passages(model, rows):
    import faiss

    if not rows:
        raise ValueError("임베딩할 텍스트가 없습니다.")
    vectors = model.encode([row["embedding_input"] for row in rows], normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(model, index, stickers, inputs, top_k=5, *, group_stickers=False):
    query_inputs = ["query: " + query for query in inputs]
    queries = model.encode(query_inputs, normalize_embeddings=True)
    depth = index.ntotal if group_stickers else min(top_k, index.ntotal)
    scores, positions = index.search(queries, depth)
    results = []
    for query, query_input, query_scores, query_positions in zip(inputs, query_inputs, scores, positions):
        matches = []
        seen = set()
        for score, position in zip(query_scores, query_positions):
            sticker = stickers[position]
            key = sticker["package_id"], sticker["sticker_id"]
            if key in seen:
                continue
            seen.add(key)
            matches.append({"rank": len(matches) + 1, "similarity": round(float(score), 4), **sticker})
            if len(matches) == top_k:
                break
        results.append({"query": query, "embedding_input": query_input, "matches": matches})
    return {"indexed_stickers": index.ntotal, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="One or more Korean search queries")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1 or any(not query.strip() for query in args.query):
        parser.error("Provide nonempty queries and a positive --top-k")
    result = search(*build_index(args.annotations), args.query, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
