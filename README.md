# 입항 기록 ↔ 선박 카탈로그 매칭 (DACON 236771) — 협업 작업 저장소

DACON 「항만 문제해결을 위한 AI 기술 공모전」에서 Claude(Opus)·Gemini(Argon)·ChatGPT(Terra)·Astra가 함께 만든 파이프라인과 실험 기록입니다. 입항 기록 한 행을 선박 카탈로그 78,595척 가운데 하나의 `vessel_id`로 연결합니다.

- 평가: `0.7 × 한글 선박명 행 정확도 + 0.3 × 비한글 행 정확도`
- 제약: 코드 제출(L4 GPU, 30분), 행 독립 예측, 외부 데이터 금지, 공개 사전학습 가중치만 허용
- 최종 선택: **V5_S1S3 Public 0.968471206** (한글 512/532, 영문 3506/3568로 해독)
- 재현 방법: [`REPRODUCE.md`](REPRODUCE.md)

## 점수 흐름 (Public, 소수 10자리 해독)

| 제출 | 핵심 | 한글 | 영문 | Public |
|---|---|---|---|---|
| 93 | LightGBM 랭커(한글 6 + 영문 12), KoMatcher 한글↔영문 매처 | 492 | 3506 | 0.9422 |
| 110 | + 사용자 v18식 넓은 퍼지 후보(발음 키 Indel 상위 500, 오타 되돌림) | 493 | 3506 | 0.9435 |
| 112 | + **Qwen2.5-72B로 카탈로그 7.8만 척 영문명을 한글 표기**로 미리 변환(`shared/src/kocat.py`), 자모 유사도 피처(kc_sim, kc_margin)로 한글 랭커 재학습, 110이 확신하지 못하는 행에만 적용 | 501 | 3506 | 0.9540 |
| 115 | **3자 한글 투표**: 112 + 사용자 v20 + HanSunwook 1031_08, 행별 다수결(`shared/experiments/build_vote.py`) | 507 | 3506 | 0.9619 |
| F1 | 어려운 한글 행은 v20, 나머지는 112 (시간 가드) | 508 | 3506 | 0.9632 |
| **V5_S1S3** | F1 + 112와 Erde가 합의하면 v20 답을 뒤집기(S1·S3) (`shared/experiments/build_vote5.py`) | **512** | 3506 | **0.9685** |

실패에서 배운 것: 합성 노이즈 fold 검증 이득은 대부분 Public으로 옮겨지지 않았다(99 −13, 107 −1, 111 −3). 옮겨진 것은 새 정보를 넣은 변경(112의 72B 한글 표기표)과 서로 다른 계열 모델의 결합(115)이었다. 진단 스크립트의 들여쓰기 버그(105~108)로 잘못된 결론을 하루 동안 따라간 기록도 `shared/experiments/LOG.md`에 남겼다.

## 구조

| 경로 | 내용 |
|---|---|
| `shared/src/` | 공용 라이브러리(`opus_lib.py` 후보 검색·피처, `ko_translit_argon*.py` 한글↔영문 매처), `local_check.py`(제출 ZIP 로컬 실행·행 독립성 검사) |
| `shared/experiments/` | 제출본 조립 스크립트(`build_*.py`), 고정 빌드 레시피(`pinned/buildNN/`), 공동 실험 로그 `LOG.md` |
| `submissions/README.md` | 모든 제출의 내용·로컬 검증·Public 점수와 해독(한글/영문 정답 수) |
| `shared/kocat/` | 72B 카탈로그 한글 표기 생성 노트북(`4_catalog_ko.ipynb`). 표 자체(CSV)는 데이터 파생물이라 제외 |
| `shared/experiments/combo_kit/` | 결합·투표 ZIP 조립과 점검(`check_combo.py`), Colab 노트북 |
| `opus/`, `argon/`, `terra/`, `astra/` | 각 참여자의 메시지·실험 코드 |
| `COLLAB.md` | 협업 규칙과 대회 규칙 요약 |

## 제외한 것

대회 데이터(`train.csv`, `vessel_catalog.csv` — 재배포 금지), 모델 가중치, 제출 ZIP, 데이터가 들어간 파생 파일(CSV·JSON·JSONL)은 올리지 않았습니다. 재현하려면 DACON에서 데이터를 받아 `data/open/`에 두세요.
