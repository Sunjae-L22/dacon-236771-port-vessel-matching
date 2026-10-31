# Opus ↔ Argon 협업 프로토콜 (항만 선박 매칭 대회)

대회: https://dacon.io/competitions/official/236771 (리더보드 제출 마감 2026-11-01 10:00 (10/30 밤 사용자 재확인; 11-02는 리더보드 제출 마감이 아님), 하루 10회 제출)
목표: Public LB 1위 (2026-10-22 기준 1위 0.9765)

## 폴더
- `opus/`   : Claude(Opus)가 쓰는 메시지. Argon은 읽기만.
- `argon/`  : Gemini(Argon)가 쓰는 메시지. Opus는 읽기만.
- `terra/`  : ChatGPT(Terra)가 쓰는 메시지 (10/27 합류). 다른 사람은 읽기만.
- `shared/src/` : 공용 코드. 파일 상단에 작성자 표기. 남의 파일은 직접 고치지 말고 메시지로 제안하거나 복사본(`*_argon.py`)을 만든다.
- `shared/experiments/LOG.md` : 모든 실험 결과를 한 줄씩 추가 (append only).
- `data/open/` : 원본 데이터 (train.csv, vessel_catalog.csv, sample_submission.csv). 수정 금지.
- `data/baseline/` : 공식 베이스라인 제출 구조.

## 메시지 규칙
- 파일명: `NNN_YYYYMMDD-HHMM_제목.md` (NNN은 각자 001부터 증가)
- 첫 줄에 `To: argon|opus`, `Re: <상대 메시지 파일명>` 표기
- 내용: 무엇을 했는지 / 결과(CV 수치) / 상대에게 요청하는 작업 / 다음 계획
- 작업 할당은 `## 요청` 섹션에 체크박스로. 끝나면 답장에서 `[x]` 표시.

## 검증 기준 (둘이 반드시 같은 CV를 사용)
- 지표: 0.7 × 한글명 정확도 + 0.3 × 비한글명 정확도
- 분할: `shared/src/cv.py`의 고정 분할 사용 (call_sign 그룹 단위 5-fold, seed 42)
- 실험 로그 형식: `| 날짜 | 작성자 | 실험명 | CV weighted | CV hangul | CV non-hangul | 비고 |`

## 대회 규칙 핵심 (위반 시 실격)
- 외부 선박 DB / 외부 매핑표 사용 금지. 학습 데이터는 train.csv, vessel_catalog.csv만.
- 공개 범용 사전학습 가중치는 사용 가능.
- Test 각 행은 독립적으로 예측 (다른 test 행/분포 사용 금지, 순서·부분집합 바꿔도 결과 동일).
- 추론 30분 이내, L4 GPU / 6 vCPU / 26GB RAM, 추론 시 인터넷 없음.
- 제출 구조: submit.zip = model/ + script.py + requirements.txt

## 제출 파일 번호
- 제출본은 `submissions/NN_submit.zip` (01, 02, …). `shared/src/make_submit.py`가 다음 번호로 자동 저장.
- 제출 기록은 `submissions/README.md`에: `| 번호 | 날짜 | 내용 | CV | Public LB |`
