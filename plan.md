임베딩 모델과 SLM이 각각 맡을 일을 먼저 정하고, 그다음 후보 모델, 실험 순서, 공정 비교 규칙, 일정을 정리했어요.

0. 모델이 맡는 역할
단계	쓰는 모델	입력 → 출력	비교 여부
오프라인 시나리오 생성	큰 VLM 1개	이미지 → JSON 설명 + 사용 대화	고정 (바꾸면 인덱스가 매번 달라짐)
반응 추출	SLM	최근 대화 → 감정·의도·톤 JSON + 한 줄 설명	비교
검색	임베딩 모델	반응 설명 ↔ 사용 대화	비교
재정렬	BERT cross-encoder, 리랭커	(대화, 후보) → 점수	마지막에 비교
1. 임베딩 후보 (10개 + BM25 기준선)

한국어 검색 점수는 KURE-v2 모델 카드에 실린 MTEB(kor, v2) 검색 9개 과제 평균 nDCG@10이에요. 공개된 표 중에 한국어 숫자가 정리된 게 이것뿐이라 기준으로 삼았어요.

구분	모델	크기	한국어 검색	넣는 이유
한국어 특화	nlpai-lab/KURE-v1	568M	0.7616	한국어 검색 표준 기준선
한국어 특화	dragonkue/snowflake-arctic-embed-l-v2.0-ko	568M	0.7653	같은 크기 단일 벡터 중 최상위권
한국어 특화	Kanana-2-1.3b-embedding (카카오)	1.3B	미공개	가장 최신 한국어 임베딩
다국어	BAAI/bge-m3	568M	0.7509	다국어 기준선, KURE의 원본 모델
다국어	intfloat/multilingual-e5-large	560M	(KoE5 0.7337)	오래된 표준 기준선
다국어	google/embeddinggemma-300m	300M	미측정	경량 후보
다국어	microsoft/harrier-oss-v1-0.6b	0.6B	미측정	MMTEB 1위권
다국어	Qwen/Qwen3-Embedding-0.6B	0.6B	0.7140	현재 기본 임베더
상한선	Qwen/Qwen3-Embedding-4B	4B	0.7737	크기를 키우면 얼마나 오르는지 확인
별도 트랙	nlpai-lab/KURE-v2	154M	0.8160	한국어 검색 1위, 구조가 달라서 따로 비교

모델별 근거:

Kanana-2: 카카오가 10월 29일에 3B와 1.3B 두 종을 공개했어요. 유사 규모의 공개 임베딩 모델인 'llama-nemotron-embed-1b-v2', 'Qwen3-Embedding-4B'와 비교한 평가에서 가장 높은 한국어 성능과 유사한 수준의 영어 성능을 보였다고 해요. 다만 카카오 자체 평가예요. 3B는 시간이 남을 때만 넣으세요. 
kakaocorp
EmbeddingGemma: 500M 미만 모델 중 MTEB 다국어·영어·코드 리더보드 1위예요. 
arxiv
harrier: 마이크로소프트의 다국어 임베딩 모델로 decoder-only 구조에 last-token pooling을 쓰고, 0.6B가 Multilingual MTEB v2에서 69.0을 받았어요. 그런데 11월 AACL 논문은 granite-311m과 harrier-0.6b가 전체 순위는 상위권인데 특정 언어에서는 무너지는 "숨은 실패" 모델로 꼽았어요. 한국어에서도 무너지는지 직접 확인할 가치가 있고, 그 자체가 블로그 소재예요. 
harrier oss v1 270m
 +2
KURE-v2: A.X-Encoder 기반의 late-interaction(multi-vector) 모델로, 토큰마다 128차원 벡터를 만들어 MaxSim으로 점수를 매겨요. PyLate 라이브러리가 필요하고 FAISS 단일 벡터 인덱스와 구조가 달라서 별도 트랙으로 둡니다. 
huggingface

참고로 Qwen3-Embedding-8B는 0.7826이에요. GPU 여유가 있을 때만 상한선으로 넣으세요. 
huggingface

2. SLM 후보 (반응 추출용, 0.8~4B)
구분	모델	크기	메모
한국어 특화	LGAI-EXAONE/EXAONE-4.0-1.2B	1.2B	추론 모드는 끄고 사용
한국어 특화	kakaocorp/kanana-2-1.3b-instruct	1.3B	2026년 7월 공개
한국어 특화	kakaocorp/kanana-2-3b-instruct	3B	위와 같은 시리즈
한국어 특화	HyperCLOVA X SEED 1.5B (Text-Instruct)	1.5B	2025년 4월 공개
글로벌	Qwen/Qwen3.5-0.8B / 2B / 4B	0.8/2/4B	같은 계열에서 크기 효과 확인
글로벌	Gemma 4 E2B / E4B (instruct)	유효 2B/4B	JSON 출력 지원
글로벌	microsoft/Phi-4-mini-instruct	3.8B	영어 중심 대조군
상한선	시나리오 생성에 쓴 큰 모델	–	"정답 반응"으로 사용
EXAONE: 어휘가 한국어와 영어 토큰이 대략 같은 비율이에요. LG IC-PBL 수업이니 발표 때 언급하기도 좋아요. 
arxiv
Kanana-2: 1.3B는 3B에서 단계적 가지치기와 증류로 만든 모델이라, 3B와 1.3B를 비교하면 크기 대비 효율을 볼 수 있어요. 
huggingface
Qwen3.5: 2026년 3월 2일에 0.8B, 2B, 4B, 9B 네 가지로 공개됐고 모두 Apache 2.0이에요. 
nyu
Gemma 4: 함수 호출, 구조화된 JSON 출력, 시스템 지시를 기본 지원해서 JSON 반응 추출에 맞아요. 
blog
0.8B급 주의: 작은 모델은 복잡한 지시에서 JSON 스키마를 잘 안 지킨다는 지적이 있어요. 그래서 JSON 유효율을 지표에 넣습니다. 
producthunt

정확한 Hub ID는 Kanana-2 임베딩, HyperCLOVA X SEED, Gemma 4가 아직 확인 안 됐어요. 받기 전에 Hub에서 한 번 확인하세요.

3. 실험 순서

전부 곱하면 10 × 10 = 100개 조합이라, 세 단계로 줄여요.

Stage 1. 임베딩 비교 (SLM 없이)

쿼리는 두 가지로 고정해요. (a) 최근 대화 원문, (b) 큰 모델이 쓴 정답 반응 설명.
이렇게 하면 임베더 성능만 따로 볼 수 있어요.
지표: Recall@5, Recall@50, MRR, 대조쌍 정확도, 인코딩 시간, 인덱스 크기.
여기서 3개를 남겨요. 전체 1위, 경량 1위, 한국어 특화 1위.

Stage 2. SLM 비교 (임베더는 Stage 1의 1위로 고정)

지표는 세 종류예요.
하류 성능: Recall@5, MRR, 대조쌍 정확도
출력 품질: JSON 유효율, 정답 반응과의 감정 일치율(KOTE 기준으로 정규화)
비용: 지연시간 p50/p95, VRAM
기준선은 Stage 1의 (a)예요. SLM 없이 대화 원문을 그대로 쿼리로 쓴 결과죠. 이것보다 못한 SLM은 넣을 이유가 없다는 뜻이에요.
(b)와의 차이는 "작은 모델이라서 잃는 성능"을 보여 줘요.

Stage 3. 조합과 재정렬

상위 임베더 3개 × 상위 SLM 3개 = 9개 조합으로, 둘 사이에 궁합이 있는지 봐요.
최고 조합에서 재정렬 모델을 비교해요. 기존 BERT 백본에 리랭커 두 개를 추가합니다.
Qwen/Qwen3-Reranker-0.6B
dragonkue/bge-reranker-v2-m3-ko: bge-reranker-v2-m3를 한국어로 미세조정한 모델
4. 공정 비교 규칙
접두사와 지시문은 모델 카드 그대로 써요. 예를 들어 e5는 query: /passage: , Qwen3-Embedding과 harrier는 쿼리 쪽에 지시문, EmbeddingGemma는 과제 프롬프트, bge-m3와 KURE는 없음이에요. 잘못 넣으면 점수가 크게 떨어지니 설정 파일에 모델별로 적어 두세요.
나머지 조건은 전부 같게 맞춰요.
L2 정규화 + 코사인 유사도
최대 512토큰, 가장 오래된 턴부터 자르기
fp16
FAISS는 IndexFlatIP(정확 검색)를 써요. 벡터가 약 6만 개(8,052개 × 시나리오 7~8개)라 근사 검색이 필요 없고, 근사 오차가 결과에 섞이지 않아요.
SLM 설정도 통일해요.
프롬프트 하나, temperature 0, 최대 128토큰
생각/추론 모드 끄기
각 모델의 chat template 사용
JSON은 vLLM의 구조화 출력으로 강제하거나, 실패하면 1회 재시도하고 그 횟수도 기록
같은 계열 편향을 막아요. 테스트 대화를 Qwen으로 만들면 Qwen 임베더와 SLM이 유리해져요. 테스트셋 생성기는 비교 대상에 없는 계열을 쓰세요.
5. 계산량과 일정
0.6B 임베더로 인덱스 6만 문장을 만드는 건 GPU 한 장에서 몇 분이면 끝나요.
4B 임베더와 SLM 10개 × 테스트 대화 약 1,500개도 vLLM 배치로 돌리면 모델당 수 분~수십 분이에요.
GPU 한 장(Colab A100이나 학교 서버)으로 충분해요.