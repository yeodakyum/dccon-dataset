# 디시콘 인기 TOP100 크롤러

[공개 데이터 다운로드 및 Python 예제](../docs/PUBLIC_DATASET_ACCESS.md)

디시인사이드 인기 순위에서 **디시콘 세트 100개**를 고르고, 각 세트의 상세 응답에 들어 있는 개별 콘 전체를 수집한다.

## 실행

저장소 루트에서 Python 3.10 이상으로 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r crawler/requirements.txt
python crawler/crawl_dccon.py
```

```bash
# 일간 TOP100
python crawler/crawl_dccon.py --period day

# 월간 TOP100
python crawler/crawl_dccon.py --period month

# 3세트 시험 수집
python crawler/crawl_dccon.py --limit 3 --output data/smoke

# 이미지 요청 없이 주간 100세트의 메타데이터만 수집
python crawler/crawl_dccon.py --metadata-only --output data/week-metadata
```

기본 저장 위치는 `data/week/`이며, 다른 기간을 선택하면 `data/<기간>/`에 저장한다. 요청 간격 기본값은 0.35초다. `--delay 1`처럼 늘릴 수 있다. 이미지 수에 따라 수집 시간이 달라진다.

같은 `--output`, `--period`, `--limit`으로 재실행하면 해당 폴더의 순위와 성공한 상세 응답을 재사용한다. 기존 이미지가 디코딩되면 파일을 재사용하고, 누락되거나 손상된 파일은 다시 요청한다. 새 순위 스냅샷은 새 `--output` 경로에 저장한다. 상세 응답의 수집 시점은 순위 조회 시점과 다를 수 있다.

## 출력

| 경로                             | 내용                                                                                                                                                                                 |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ranking.json`                   | 조회 기간·시각·URL, 순위 API가 반환한 원본 항목 전체                                                                                                                                 |
| `raw/packages/<package_id>.json` | 세트 상세 API의 전체 응답과 조회 시각                                                                                                                                                |
| `packages.jsonl`                 | 세트당 한 줄. 순위, 제목, 설명, 제작자 표시명, 태그, 등록일, 분류, 상태, 가격, 사이트의 `sale_count`, 대표·목록 이미지 URL, 원본 순위·상세 필드                                      |
| `stickers.jsonl`                 | 개별 콘당 한 줄. 세트 ID, 콘 ID, 제목, 순서, 원본 필드, 원본 URL, 상대 파일 경로, 형식·MIME·바이트 크기·가로/세로·픽셀 모드·투명 여부·SHA-256·프레임 수·프레임별 재생 시간·반복 설정 |
| `images/<package_id>/`           | 개별 콘 이미지 원본. 세트 대표·목록 이미지는 URL을 제공한다.                                                                                                                         |
| `errors.jsonl`                   | 실패한 세트·콘 ID, 실패 단계, 오류 메시지                                                                                                                                            |
| `summary.json`                   | 수집 성공 수, 오류 수, 완료 여부, 메타데이터 전용 실행 여부                                                                                                                          |

`package_id`로 두 JSONL 파일을 결합할 수 있다. `image_status == "ok"`인 행은 검증된 파일을 가리킨다. `sha256`은 바이트가 동일한 파일의 중복을 찾는 데 쓸 수 있다. 같은 세트의 콘이 학습·평가 양쪽에 섞이지 않도록 분할하려면 `package_id`를 그룹 키로 사용한다. 사이트 응답에 없는 정규화 필드는 `null`로 기록된다. 프레임 시간은 밀리초이며 정지 이미지의 프레임별 시간 배열은 비어 있다. 등록 시각은 사이트의 문자열 그대로 저장한다.

세트의 표시 개수(`reported_icon_count`)와 실제 상세 목록 길이(`detail_icon_count`)가 다를 수 있다. 크롤러는 상세 목록의 모든 항목을 저장한다. 수집 오류가 있으면 부분 결과와 오류 내역을 남기고 종료 코드 1을 반환한다. `image_status`는 이미지 수집에 실패하면 `error`, 메타데이터만 수집하면 `skipped`로 기록된다. 이미지에서 추출하는 필드는 다운로드에 성공한 행에 들어간다.

## Cloudflare R2 업로드

저장소 루트의 `.env`에 `R2_ACCOUNT_ID`, `R2_BUCKET`, Cloudflare에서 발급한 S3 `R2_ACCESS_KEY_ID`와 `R2_SECRET_ACCESS_KEY`를 입력한다. 키에는 이 버킷의 **Object Read & Write** 권한이 필요하다. [R2 API 자격 증명](https://developers.cloudflare.com/r2/api/tokens/)

크롤링 중이면 새 터미널에서 다음 명령을 실행한다.

```bash
source .venv/bin/activate
python -m pip install -r crawler/requirements-r2.txt
python crawler/upload_r2.py --follow
```

`--follow`는 기본 30초 간격으로 `data/week/`를 확인해 저장이 끝난 이미지부터 전송한다. 크롤링이 완료되면 이미지 SHA-256을 매니페스트와 대조한 뒤 JSONL, 원본 응답, `summary.json` 순으로 업로드하고 종료한다. 크롤러가 중단되거나 오류를 남긴 경우 업로더는 크롤러의 완료 신호를 기다린다. 업로더를 중단했다가 같은 명령으로 재실행하면 R2에 이미 같은 SHA-256과 크기로 저장된 파일을 건너뛴다.

```bash
# 크롤링 완료 후 한 번에 업로드
python crawler/upload_r2.py

# 월간 스냅샷과 자격 증명 프로필 사용
python crawler/upload_r2.py --directory data/month --profile r2

# 네트워크 요청 없이 완료된 로컬 스냅샷 검증
python crawler/upload_r2.py --dry-run
```

기본 R2 경로는 `dccon/<기간>/<순위 조회 UTC 시각>/`이다. `--prefix`로 앞부분을 바꿀 수 있다. 폴더 구조를 유지하므로 `stickers.jsonl`의 상대 경로가 그대로 동작한다. 배포에 사용할 때는 해당 경로의 `summary.json`이 올라온 뒤 공유한다. 버킷의 공개 설정은 Cloudflare에서 별도로 관리한다.
