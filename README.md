# DCCON 데이터셋과 추천 실험

디시인사이드 인기 디시콘의 패키지 정보와 원본 이미지를 수집하고, 보낼 채팅과 앞 대화에 어울리는 디시콘을 검색하는 프로젝트입니다.

| 경로 | 내용 |
| --- | --- |
| [`crawler/`](crawler/README.md) | 일간·주간·월간 인기 순위 수집, 매니페스트와 이미지 저장, Cloudflare R2 업로드 |
| [`experiments/tagging/`](experiments/tagging/README.md) | Groq Qwen 이미지·GIF 태깅 GUI와 재개 가능한 배치 |
| [`experiments/pilot1/`](experiments/pilot1/2026-11-23-search-experiments.md) | 합성 주석 378개를 이용한 예문·상황·이미지 검색 비교와 화면 캡처 |
| [`experiments/pilot2/`](experiments/pilot2/README.md) | 이미지 주석 32·320개와 원문 FAISS, SLM 검색어 생성, SLM 후보 선택 비교 |
| [`experiments/pilot3/`](experiments/pilot3/README.md) | 추천 검색, 임베딩·SLM 전환, 이미지 태깅을 묶은 로컬 GUI |
| [`experiments/pilot4/`](experiments/pilot4/README.md) | 서브에이전트·Qwen 비전 채팅 예문과 디시콘 32개 전체 순위 |
| [`experiments/pilot5/`](experiments/pilot5/README.md) | 디시콘 320개 실시간 순위 비교와 Pilot 1~5 통합 GUI |
| [`docs/`](docs/2026-11-23-to-24-worklog.md) | 공개 데이터 접근 방법과 11월 23~24일 통합 실험 기록 |

## 데이터 받기

공개 주간 스냅샷에는 패키지 100개, 스티커 레코드 8,245개, 다운로드에 성공한 이미지 8,218개가 있습니다. 실패 27건이 있어 스냅샷의 `complete`는 `false`입니다. 기존 추천 실험을 재현하려면 해당 스냅샷의 매니페스트와 이미지를 저장소의 `data/week/`에 다운로드하세요. [공개 데이터 다운로드 및 Python 예제](docs/PUBLIC_DATASET_ACCESS.md)

새 순위를 수집하려면 저장소 루트에서 다음을 실행하세요. 기본값은 주간 인기 100세트이며, `--limit`으로 시험 수집 규모를 정할 수 있습니다. [크롤러 실행 및 재개 방법](crawler/README.md)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r crawler/requirements.txt
python crawler/crawl_dccon.py --limit 3 --output data/smoke
```

## 추천 GUI 실행

Pilot 1~5 화면을 한 서버에서 열려면 [Pilot 5 실행 방법](experiments/pilot5/README.md)을 참고하세요.

원본 스냅샷을 받은 다음 저장소 루트에서 실행하고 브라우저에서 Pilot 3를 엽니다. 이미지 주석 32·320개를 함께 쓰려면 Pilot 2의 주석 준비 절차를 먼저 실행하세요. [Pilot 2 주석 준비](experiments/pilot2/README.md#저장된-실험-재현) · [Pilot 3](http://127.0.0.1:8768/)

```bash
python -m pip install -r experiments/pilot3/requirements.txt
python experiments/pilot3/app.py
```

원본 이미지와 실행 중 생성되는 파일은 로컬 `data/`에 저장합니다. Git에는 합성 주석, 이미지 주석 표본, 검색 결과와 실험 코드가 있습니다. 비교 결과는 소규모 정성 점검이며 추천 정확도는 사람 평가 전입니다. [통합 실험 기록](docs/2026-11-23-to-24-worklog.md) · [Pilot 1–5 영어 비교 보고서와 결과 화면](docs/pilot-comparison.md)
