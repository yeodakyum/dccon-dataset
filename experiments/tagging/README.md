# Groq 이미지 태깅

`qwen/qwen3.8-27b`로 디시콘의 문구, 시각 묘사, 반응 의도와 근거, 불확실성을 기록합니다. GIF는 재생 시간 0·25·50·75·100%의 장면을 한 장으로 합쳐 보냅니다.

## 태깅 실험실

저장소 루트에서 실행합니다. 기본 원본 경로는 `data/week/`이며, `--dataset`으로 바꿀 수 있습니다. [공개 원본 다운로드](../../docs/PUBLIC_DATASET_ACCESS.md)

```bash
python -m pip install -r crawler/requirements.txt
python experiments/tagging/tagging_gui.py
```

[태깅 실험실](http://127.0.0.1:8766/)에서 정지 이미지나 GIF를 고르고, Groq에 보낼 프레임과 프롬프트, 응답 시간과 토큰 수를 확인할 수 있습니다. 키는 화면에서 입력하거나 `GROQ_API_KEY` 환경 변수로 전달합니다. 같은 기능은 [Pilot 3](../pilot3/README.md)에서도 사용할 수 있습니다.

**채팅 예문 30개 함께 생성**을 켜면 모델이 이미지 주석과 한국어 채팅 예문을 같은 응답에서 생성합니다. [예문 프롬프트](examples_prompt.md) · [Pilot 4 배치와 전체 순위](../pilot4/README.md)

## 배치

다음 명령은 표본만 선택합니다. `--run`을 추가하면 Groq에 요청하고 주석을 저장합니다. 중단 후 같은 설정과 출력 경로로 다시 실행하면 체크포인트를 재사용합니다. [32·320개 배치 실행 방법](../pilot2/README.md)

```bash
python experiments/tagging/tag_stickers.py --per-kind 16 --output data/pilot2
```
