# DCCON Pilot 4

Pilot 2의 정지 이미지 16개와 GIF 16개에 대해 디시콘마다 채팅 예문 30개를 작성하고 전체 순위를 검색합니다. [예문 프롬프트](../tagging/examples_prompt.md)

- **서브에이전트 예문**은 서브에이전트 3개가 정지 이미지와 GIF의 시간순 프레임 스트립을 콘별로 확인해 작성한 960개입니다. 저장된 예문으로 바로 검색할 수 있습니다. [예문 JSONL](results/subagent-examples-32.jsonl)
- **Qwen 비전 예문**은 Groq의 `qwen/qwen3.8-27b`가 원본 이미지나 GIF 프레임을 보고 이미지 주석과 함께 생성합니다. 기본 주간 스냅샷과 시드 42로 같은 이미지를 선택합니다.

## 실행

저장소 루트에서 실행합니다. `data/week/`에 원본 데이터가 필요합니다. [원본 데이터 받기](../../docs/PUBLIC_DATASET_ACCESS.md)

```bash
python -m pip install -r experiments/pilot3/requirements.txt
python experiments/pilot4/app.py
```

[Pilot 4](http://127.0.0.1:8771/)에서 예문 출처를 선택하고 보낼 말을 입력하면 1위부터 32위까지 표시합니다. 콘마다 KURE-v3(1,024차원)로 구한 예문 30개의 코사인 유사도를 합산합니다. 결과 카드에서 가장 가까운 예문을 바로 보고 전체 30개를 펼칠 수 있습니다. 검색은 로컬에서 실행합니다.

**유사도 컷오프**를 움직이면 선택한 기준 이상인 예문만 합산해 현재 순위를 다시 계산합니다. 기본값 **끄기**는 30개를 모두 합산합니다. 각 카드에 점수와 합산된 예문 수를 표시하며, 기준을 통과한 예문이 없는 콘은 0점입니다.

Qwen 예문을 생성하려면 **Qwen 비전 배치**를 열어 Groq API 키를 입력하고 **32개 생성 또는 재개**를 누릅니다. 서버에 `GROQ_API_KEY`를 설정했다면 입력란을 비워 둡니다. 배치가 완료되면 예문 출처 목록에 Qwen이 표시됩니다.

[기존 태깅 실험실](../tagging/README.md)에서 **채팅 예문 30개 함께 생성**을 켜면 한 장을 골라 프롬프트를 수정하고 원본 응답을 확인할 수 있습니다.

## 배치 명령

같은 출력 경로로 재실행하면 콘별 체크포인트를 재사용합니다. 결과는 `data/pilot4/annotations.jsonl`에 저장합니다.

```bash
python experiments/tagging/tag_stickers.py --per-kind 16 --workers 3 --examples --output data/pilot4 --run
```
