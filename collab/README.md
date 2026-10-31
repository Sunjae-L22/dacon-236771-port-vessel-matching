# 항만 문제해결을 위한 AI 기술 공모전 (DACON 236771) — AI 협업 가이드

이 저장소(`/Users/sunjae/Desktop/항만`)는 **Gemini (`argon`)**와 **Claude Code (`opus`)**가 협업하여 데이콘 **항만 문제해결을 위한 AI 기술 공모전(236771)** 1등(목표 점수 `0.980+`, 현재 Public LB 1위 `0.97502`)을 달성하기 위한 공유 작업 공간입니다.

## 1. 디렉토리 구조

```text
/Users/sunjae/Desktop/항만/
├── CLAUDE.md                 # Claude Code 자동 진입 가이드
├── collab/                   # 에이전트 간 메시지 및 상태 공유 허브
│   ├── README.md             # 협업 프로토콜 및 규칙 (본 문서)
│   ├── SHARED_STATE.md       # 공동 실험 보드, CV/LB 점수 트래커, 역할 분담 현황
│   └── messages/             # 순번 기반 비동기 메시지 스레드
│       ├── 001_from_gemini_kickoff.md
│       └── 002_from_claude_...md
├── argon/                    # Gemini 전용 실험/스크립트 작업 공간
├── opus/                     # Claude Code 전용 실험/스크립트 작업 공간
└── data/
    ├── open/                 # 공식 제공 데이터 (train.csv, vessel_catalog.csv, sample_submission.csv)
    └── baseline/             # 주최측 베이스라인 코드 및 모델
```

## 2. 메시지 교환 규칙 (`collab/messages/`)

1. **파일명 규칙**: `NNN_from_<agent>_<short_topic>.md` 형식으로 생성합니다.
   - 예: `001_from_gemini_kickoff.md`, `002_from_claude_cv_and_phonetic.md`, `003_from_gemini_ranker_v1.md`
2. **메시지 필수 포함 항목**:
   - **요약 (TL;DR)**: 이번 턴에 완료한 작업 및 핵심 수치(CV Recall, Hangul Acc, Non-Hangul Acc, Weighted Acc)
   - **변경된 파일 및 아티팩트 경로**: 상대방이 바로 확인하거나 앙상블/통합할 수 있는 코드·데이터 경로
   - **다음 턴 제안 / 요청 사항**: 상대 에이전트가 이어서 수행하면 좋을 작업 제안
3. **공동 상태 업데이트**: 중요한 실험 결과(CV 점수 갱신, 새로운 매핑 사전 생성, 제출 파일 생성 등)가 나오면 반드시 `collab/SHARED_STATE.md`도 함께 업데이트합니다.

## 3. 대회 핵심 제약사항 체크리스트 (실격 방지 필수 규칙)

- **평가 산식**: `0.7 × 한글 기록 정확도 + 0.3 × 비한글 기록 정확도` (`vessel_name`에 한글 `[가-힣]` 포함 여부 기준)
- **외부 데이터 금지**: 오직 `train.csv`와 `vessel_catalog.csv`만 사용 가능 (사전학습 모델 가중치 및 일반적인 한영 음차/발음 규칙 코드는 허용)
- **평가 행 독립성 (Row Independence — 매우 중요)**:
  - `test.csv`의 각 행은 **오직 해당 행의 입력값 + 학습/카탈로그 아티팩트**만으로 예측해야 함
  - `test.csv` 내 다른 행의 정보를 참조하거나 테스트 배치 통계/빈도/그룹 투표를 사용하면 실격 처리됨 (운영진이 단일 행·부분집합·순서 변경 상태로 재실행 검증)
- **제출 규격 (`submit.zip`)**:
  - 최상위 경로에 `script.py`, `requirements.txt`, `model/` 배치
  - 평가 서버 사양: Ubuntu 22.04, **NVIDIA L4 (VRAM 22.4GB)**, 6 vCPU, 26GB RAM, Python 3.11.17, 추론 제한 30분 (인터넷 차단)
  - 기본 설치 패키지: `torch==2.7.1+cu128`, `transformers==4.57.6`, `lightgbm==4.6.0`, `catboost==1.2.8`, `rapidfuzz==3.14.3`, `Unidecode==1.4.0`, `scikit-learn==1.5.2`, `pandas==2.0.3`, `numpy==1.26.4` (기본 설치 패키지는 `requirements.txt`에서 생략 권장)
