# 파일럿 2 — 디시콘 후보 검색과 문맥 기반 선택

Pilot 2는 Groq로 생성한 이미지 주석을 검색합니다. 기존 합성 주석은 [Pilot 1](../pilot1/2026-11-23-search-experiments.md)에 있습니다. 아래 명령은 저장소 루트에서 실행합니다.

## 저장된 실험 재현

[공개 주간 스냅샷](../../docs/PUBLIC_DATASET_ACCESS.md)의 매니페스트와 이미지를 `data/week/`에 준비합니다. 저장된 이미지 주석을 가져오면 Groq 요청 없이 같은 표본으로 검색할 수 있습니다. 경로에 이미 주석이 있으면 기존 파일을 사용합니다.

```bash
mkdir -p data/pilot2 data/pilot2_320
[ -f data/pilot2/annotations.jsonl ] || cp experiments/pilot2/results/annotations-32.jsonl data/pilot2/annotations.jsonl
[ -f data/pilot2_320/annotations.jsonl ] || cp experiments/pilot2/results/annotations-320.jsonl data/pilot2_320/annotations.jsonl
```

[32개 주석](results/annotations-32.jsonl)과 [320개 주석](results/annotations-320.jsonl)은 모델이 생성한 실험본이다. 320개 주석의 `kote_emotions`도 사람 평가 전이다. 저장된 파일로 검색을 재현할 수 있으며, 배치를 재개하려면 원래 실행의 `run.json`과 `records/`가 필요합니다.

## 새 혼합 표본 생성

정지 이미지 16개와 움직이는 GIF 16개를 주간 스냅샷에서 뽑습니다. 각 형식에서 패키지를 무작위 순서로 순회해 표본이 한 패키지에 몰리는 것을 줄입니다. 기본 시드는 42입니다.

정지 이미지는 원본을 쓰고, GIF는 재생 시간 0%, 25%, 50%, 75%, 100%를 한 장으로 합쳐 Groq Qwen에 보냅니다.

```bash
python experiments/tagging/tag_stickers.py --per-kind 16 --output data/pilot2 --run
```

`GROQ_API_KEY` 환경 변수를 사용합니다. 설정하지 않으면 터미널이 키 입력을 요청합니다. 중단되면 같은 명령으로 재개할 수 있습니다.

- `run.json`에는 모델, 프롬프트, 시드, 선택된 콘 ID가 있습니다.
- `records/`에는 콘별 체크포인트가 있습니다.
- `annotations.jsonl`에는 콘 ID, 원본 경로와 해시, OCR 문구, 시각 묘사, 반응 의도와 근거, 불확실성이 있습니다. 원본 경로는 `data/week/` 기준이며 주석은 검토 전 상태입니다.

## 추천 실험

비교 GUI의 기본 데이터는 `data/pilot2_320/annotations.jsonl`입니다. 앞서 만든 32개를 포함해 정지 이미지와 GIF를 각각 160개 사용합니다. 같은 시드로 다시 실행하면 체크포인트를 재사용합니다.

```bash
python experiments/tagging/tag_stickers.py --per-kind 160 --output data/pilot2_320 --workers 4 --run
```

아래 명령으로 검색 의존성을 설치합니다. 두 SLM 방식은 로컬 Ollama에서 `qwen3.5:2b`를 사용합니다.

```bash
python -m pip install -r experiments/pilot3/requirements.txt
python experiments/pilot2/compare.py
```

브라우저에서 `http://127.0.0.1:8767/`을 열고 **이전 대화**와 **입력할 대화**를 작성합니다.

| 방식 | 결과 |
| --- | --- |
| FAISS | 두 입력의 원문을 검색해 상위 5개를 표시 |
| SLM output → FAISS | SLM이 만든 반응 태그와 표정·동작 묘사를 주석과 같은 형식으로 검색해 상위 5개를 표시 |
| FAISS → SLM choose | 첫 번째 방식과 같은 순위의 상위 10개 주석과 두 입력을 SLM에 보내 최대 3개 선택 |

세 방식 모두 같은 E5 임베딩 모델과 FAISS 인덱스를 사용합니다. 인덱스에는 이미지 문구, 반응 의도, 시각 묘사를 넣습니다. 최종 선택에는 반응 근거와 불확실성도 제공합니다. GUI에서는 원본 GIF가 재생되며, SLM이 본 후보 10개와 원래 검색 순위를 펼쳐 볼 수 있습니다.

검색어 생성은 짧은 `intent` 태그 최대 2개와 `visual` 한 구절을 받습니다. 일정이나 장소만 전달하는 답장에는 빈 태그를 반환할 수 있습니다. 후보 선택도 상황에 맞는 콘이 없으면 빈 목록을 반환할 수 있습니다.

기존 E5와 한국어 검색 튜닝 모델을 나란히 실행하려면 다른 포트를 사용합니다.

```bash
python experiments/pilot2/compare.py --embedding-model dragonkue/multilingual-e5-small-ko --port 8770
```

[실험 기록과 선행연구](EXPERIMENTS.md)
