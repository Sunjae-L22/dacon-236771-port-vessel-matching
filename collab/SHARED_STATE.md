# 공동 실험 상태 보드 (SHARED_STATE)

- **대회명**: 항만 문제해결을 위한 AI 기술 공모전 (DACON 236771)
- **마감 일정**: 팀 병합 10/26, 리더보드 제출 마감 11/01, 대회 종료 11/02 10:00
- **현재 Public LB 1위 점수**: `0.97502` (문정동초고수) / Top 5 컷: `0.96924`
- **우리의 목표**: Public/Private LB **`0.980+`** (1위 달성)

## 1. 데이터 및 베이스라인 핵심 진단 (Verified by Gemini)

| 항목 | 수치 / 현황 | 시사점 |
| :--- | :--- | :--- |
| **Train 규모** | 12,222행 (고유 선박 `target_vessel_id` 5,851개) | 비한글 11,031행 (90.3%), **한글 1,191행 (9.7%)** |
| **Catalog 규모** | 78,595행 (`vessel_id` 고유) | **78,595행 전체(100%)가 영문(비한글) 선박명** (`Cat Hangul = 0`) |
| **평가 지표 가중치** | `0.7 × 한글 정확도 + 0.3 × 비한글 정확도` | 샘플 비중은 9.7%인 **한글 선박명 매칭이 점수의 70%**를 좌우함 |
| **베이스라인 후보 추출 Recall** | 한글 **68.60%** (817/1,191) / 비한글 **98.54%** (10,870/11,031) | 베이스라인은 한글→영문 변환이 전혀 없어 한글 쿼리를 오직 `gross_tonnage` 근접 12개로만 추출함 (이론적 최대 점수 상한선 **0.77581**) |
| **`call_sign` 특성** | Train 내 고유 `call_sign` 5,851개 = 고유 `target_vessel_id` 5,851개 | **충돌 0건 (100% 1:1 전단사 매핑)**. 학습 데이터에 등장한 선박이 Test에 재등장할 경우 `call_sign` 매핑만으로 100% 식별 가능 |
| **미사용 핵심 피처** | `call_sign`, `vessel_type`(한-영), `flag_name`(한-영), `deadweight_tonnage`, `size/unit` | 베이스라인은 선박명 단순 문자열과 총톤수(`gross_tonnage`)만 사용 중 |

## 2. 실험 및 모델링 트래커

| 실험 ID | 담당 | 내용 | CV (Unseen GroupKFold) | CV (Overall / Seen 포함) | LB 점수 | 상태 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `B0_baseline` | 공식 | RandomForest + 10개 기본 피처 (Top-40 후보) | 측정 예정 | 측정 예정 (Recall 상한 0.77581) | - | 분석 완료 |
| `E1_cv_framework` | 공동 | GroupKFold(`target_vessel_id` 기준) + Seen/Unseen 분리 평가 파이프라인 | - | - | - | 대기 중 |
| `E2_ko_en_phonetic` | 공동 | 한글 선박명 ↔ 영문 발음/음차 변환기 + `vessel_type`/`flag_name` 매핑 | - | - | - | 대기 중 |
| `E3_candidate_v2` | 공동 | 다중 채널 후보 추출 (Exact CS + Phonetic/Fuzzy + Tonnage + Type/Flag 필터) → Recall >99.3% 목표 | - | - | - | 대기 중 |
| `E4_gbm_ranker` | 공동 | LightGBM / CatBoost Pairwise & Listwise Reranker + `rapidfuzz` 정밀 피처 | - | - | - | 대기 중 |

## 3. 현재 작업 분담 제안

- **Gemini (`argon/`)**:
  - 협업 허브(`collab/`) 및 기초 EDA 구축 완료
  - 한글↔영문 해운 선박명 음차/발음 변환 규칙(`ko_to_en_phonetic`, 영문→한글 발음 변환, 선박 접두/접미사 `제7하나`↔`No. 7 Hana`, `~호` 제거 등) 및 `vessel_type` / `flag_name` 확률 매핑 테이블 설계
- **Claude Code (`opus/`)**:
  - 로컬 Python 가상환경/패키지 확인 및 표준 CV 평가 스크립트(`GroupKFold` on `target_vessel_id` + `Seen/Unseen` 리포트) 구축
  - 후보군 추출(Candidate Retrieval) Recall을 한글 99%+, 비한글 99.8%+로 끌어올리는 고속 인덱서 구현 및 LightGBM/CatBoost 랭커 파이프라인 작성
