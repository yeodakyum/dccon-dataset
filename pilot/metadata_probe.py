"""Inspect how conversation inputs change a simple local DCCON search."""
import json
import re
import unicodedata
from collections import Counter
from math import log, sqrt
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "week"
PAIRS = (
    ("미쳤다 ㅋㅋ", (
        ("나 장학금 받았어!", "축하 축하해 잘했다"),
        ("마감 5분 전에 파일 날아갔어.", "안타까움 위로 괜찮아"),
    )),
    ("와 진짜?", (
        ("아까 내 말 때문에 상처받았어.", "미안 미안해 사과"),
        ("네가 갖고 싶다던 거 내가 사 왔어.", "감사 고마워 고맙다"),
    )),
    ("됐다...", (
        ("몇 번을 물어봐도 결국 안 된대.", "포기 좌절 슬픔"),
        ("방금 새로고침하니까 제출 완료 떴어!", "안도 다행 성공"),
    )),
    ("아니 ㅋㅋ", (
        ("고양이가 키보드 눌러서 교수님한테 ㅁㅁㅁ 보냈어.", "웃음 웃겨 폭소"),
        ("내가 진짜 힘들다고 했는데 또 장난으로 넘기네.", "화남 짜증 그만해"),
    )),
)


def read_rows(name):
    with (DATA / name).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def ngrams(value):
    tokens = re.findall(r"[가-힣a-z]+", unicodedata.normalize("NFKC", value).lower())
    return Counter(part[i:i+size] for part in tokens for size in (2, 3)
                   for i in range(len(part) - size + 1))


def main():
    packages = {p["package_id"]: p for p in read_rows("packages.jsonl")}
    rows = sorted(read_rows("stickers.jsonl"), key=lambda r: (r["rank"], int(r["sticker_id"])))
    seen = set()
    documents = []
    document_frequency = Counter()
    for row in rows:
        if row["image_status"] != "ok" or row["sha256"] in seen:
            continue
        seen.add(row["sha256"])
        package = packages[row["package_id"]]
        terms = ngrams(" ".join([row["title"], package["title"], *package["tags"]]))
        documents.append((row, terms))
        document_frequency.update(terms.keys())

    idf = {term: 1 + log((len(documents) + 1) / (count + 1))
           for term, count in document_frequency.items()}
    vectors = []
    for row, terms in documents:
        weights = {term: count * idf[term] for term, count in terms.items()}
        norm = sqrt(sum(w*w for w in weights.values())) or 1
        vectors.append((row, {term: weight / norm for term, weight in weights.items()}))

    def search(query):
        weights = {term: count * idf[term] for term, count in ngrams(query).items()
                   if term in idf}
        scored = [(sum(weight * vector.get(term, 0) for term, weight in weights.items()), row)
                  for row, vector in vectors]
        ranked = sorted(scored, key=lambda result: -result[0])
        return [row for score, row in ranked[:5] if score > 0]

    print(f"{len(documents)} byte-distinct successful stickers")
    for last, variants in PAIRS:
        print(f"\nLast message {last!r}")
        for i, (history, intent) in enumerate(variants, 1):
            for label, query in (("last", last), ("history", f"{history} {last}"),
                                 ("human reply intent", intent)):
                matches = search(query)
                print(f"{i} {label:18} {query!r} "
                      + (", ".join(f"{row['title']} ({row['sticker_id']})" for row in matches)
                         or "no lexical matches"))


if __name__ == "__main__":
    main()
