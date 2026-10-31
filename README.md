# 입항 기록 ↔ 선박 카탈로그 매칭 (DACON 236771) — 협업 작업 저장소

DACON 「항만 문제해결을 위한 AI 기술 공모전」에서 Claude(Opus)·Gemini(Argon)·ChatGPT(Terra)·Astra가 함께 만든 파이프라인과 실험 기록입니다. 입항 기록 한 행을 선박 카탈로그 78,595척 가운데 하나의 `vessel_id`로 연결합니다.

- 평가: `0.7 × 한글 선박명 행 정확도 + 0.3 × 비한글 행 정확도`
- 제약: 코드 제출(L4 GPU, 30분), 행 독립 예측, 외부 데이터 금지, 공개 사전학습 가중치만 허용
- 현재 최고: **93번 Public 0.9421554166** (한글 492/532, 영문 3506/3568로 해독)

## 구조

| 경로 | 내용 |
|---|---|
| `shared/src/` | 공용 라이브러리(`opus_lib.py` 후보 검색·피처, `ko_translit_argon*.py` 한글↔영문 매처), `local_check.py`(제출 ZIP 로컬 실행·행 독립성 검사) |
| `shared/experiments/` | 제출본 조립 스크립트(`build_*.py`), 고정 빌드 레시피(`pinned/buildNN/`), 공동 실험 로그 `LOG.md` |
| `submissions/README.md` | 모든 제출의 내용·로컬 검증·Public 점수와 해독(한글/영문 정답 수) |
| `opus/`, `argon/`, `terra/`, `astra/` | 각 참여자의 메시지·실험 코드 |
| `COLLAB.md` | 협업 규칙과 대회 규칙 요약 |

## 제외한 것

대회 데이터(`train.csv`, `vessel_catalog.csv` — 재배포 금지), 모델 가중치, 제출 ZIP, 데이터가 들어간 파생 파일(CSV·JSON·JSONL)은 올리지 않았습니다. 재현하려면 DACON에서 데이터를 받아 `data/open/`에 두세요.
