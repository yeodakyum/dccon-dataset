# DCCON Pilot 3

Pilot 1의 합성 주석 검색, Pilot 2의 이미지 주석 검색과 SLM 비교, Groq 이미지 태깅 실험실을 로컬 GUI 하나에서 실행합니다.

## 실행

저장소 루트에서 Python 3.10 이상 환경을 사용합니다.

```bash
python -m pip install -r experiments/pilot3/requirements.txt
python experiments/pilot3/app.py
```

브라우저에서 [Pilot 3](http://127.0.0.1:8768/)를 엽니다. `data/week/`의 원본 이미지와 매니페스트를 사용하며, `--dataset`으로 다른 경로를 지정할 수 있습니다. Pilot 2 주석은 `data/pilot2/annotations.jsonl`과 `data/pilot2_320/annotations.jsonl`에 파일이 있을 때 메뉴에 표시됩니다. [저장된 주석으로 실행하기](../pilot2/README.md#저장된-실험-재현)

모델 캐시를 저장소의 로컬 데이터 폴더에 보관하려면 `HF_HOME`을 지정합니다.

```bash
HF_HOME="$PWD/data/model-cache" python experiments/pilot3/app.py
```

## 추천 검색

- **합성 주석**은 답장 예문, 상황, 혼합 순위, 문맥 포함 검색, 이미지 설명, SigLIP 2, SLM 의도 검색, SLM 표정·동작 이미지 검색을 제공합니다. 직접 작성한 의도 검색어와 표정·동작도 따로 실행합니다.
- **이미지 주석**은 32개 또는 320개 표본에서 원문 FAISS, SLM 검색어 생성 후 FAISS, FAISS 후보를 SLM으로 선택하는 방식을 제공합니다. 원본 GIF가 재생되고 추천 근거, 실제 임베딩 입력, SLM 후보 목록을 펼쳐볼 수 있습니다.
- **텍스트 모델**은 기존 `intfloat/multilingual-e5-small`(384차원)과 `nlpai-lab/KURE-v3`(1,024차원) 중 선택합니다. 모델과 검색 컬렉션별로 첫 요청에서 인덱스를 만들고 서버가 실행되는 동안 재사용합니다. 최초 모델 사용 시 캐시에 없으면 가중치를 다운로드합니다. SigLIP 2 검색은 자체 모델을 사용합니다.
- **SLM 모델**에서 로컬 Ollama(기본값 `qwen3.5:2b`)와 Groq(`qwen/qwen3.8-27b`)를 전환합니다. SLM 검색 방식을 선택하면 해당 모델로 검색어를 만들거나 후보를 선택합니다. 로컬 모델은 `--ollama-model`로 바꿀 수 있습니다. Groq는 추천 검색 화면의 API 키 입력 또는 `GROQ_API_KEY` 환경 변수를 사용합니다. Groq 선택 시 SLM 검색에 쓰는 대화와 필요한 후보 주석이 Groq로 전송됩니다.

320개 표본에서 **예문 + 감정**을 선택하면 보낼 말의 KOTE 상위 감정 3개를 표시합니다. KOTE KcELECTRA는 첫 실행 때 모델을 내려받고 로컬에서 추론합니다. 답장 예문이 있는 콘은 예문을 검색하고, 없으면 이미지 문구나 기존 주석을 검색합니다. 텍스트 유사도에 일치한 감정 점수의 합을 3으로 나눈 값의 0.10배를 더해 상위 5개를 표시합니다. 콘에는 기존 `kote_emotions` 태그를 사용합니다.

## 태깅과 실험 기록

**이미지 태깅**에서 정지 이미지·GIF를 고르고, Groq에 전송할 GIF 스트립을 확인하고, 프롬프트를 수정해 JSON 주석을 시험할 수 있습니다. Groq 키는 화면 입력 또는 `GROQ_API_KEY` 환경 변수로 전달합니다. 단건 태깅 출력은 화면에 표시됩니다. 같은 탭에서 정지 이미지와 GIF가 섞인 32개·320개 배치를 시작하거나 재개할 수 있습니다. 완료하면 추천 검색에 새 주석이 반영됩니다. 배치 CLI는 [Pilot 2 실행 방법](../pilot2/README.md)에 있습니다.

**실험 기록**에서 Pilot 1·2의 기존 실험, 이전 E5 계열 A/B 결과와 통합 작업 기록을 열 수 있습니다. [11월 23~24일 통합 기록](../../docs/2026-11-23-to-24-worklog.md)
