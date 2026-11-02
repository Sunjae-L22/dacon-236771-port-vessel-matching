# 동일 문자권 E5 contrastive 학습 — 기각

**1000step 최종 모델은 채택하지 않는다.** 전체 top1은 374→372로 내려갔고, 깨끗한 실제 이름 130척 중 기존 정답 2척을 훼손했다. R100에서 새로 찾은 8행도 기존 자음500+ksh500+KoCat50 후보 합집합에 이미 정답이 있었다. 이번 관측에서 추가 검색 범위의 oracle 이득은 **0행**이다.

## 고정 실행

- 공개 원본 multilingual-e5-small에서 query encoder 마지막 2개 layer만 학습했다. 이전 English-prototype 실험의 학습 가중치를 재사용하지 않았다.
- target은 고정된 한글 카탈로그 alias embedding이다. 같은 선박의 모든 alias를 positive 집합으로 삼는 multi-positive contrastive loss를 사용했다.
- 여러 선박이 공유하는 동일 정규화 alias 3,308개는 학습 prototype과 catalog 샘플에서 제외했다. 평가에는 원래 127,962개 alias 전체를 사용했다.
- human query는 fold2/3/4만 사용했다. 모호 alias와 같은 질의를 제외하고 199개가 남았다. 평가 fold0/1의 130척은 human query뿐 아니라 catalog alias도 positive 학습 샘플에서 제외했다. 이들의 비모호 alias 187개는 고정 prototype의 negative로는 남았다.
- 학습 prototype 124,654개, catalog positive 124,467개. Batch80은 catalog64개+human16개로 고정했다. Catalog 질의는 절반은 그대로, 절반은 음절 삭제·중복·인접 교환 중 1회 변형이다. Human fit 질의는 원문이다.
- seed112, 1000steps, lr2e-5, temperature0.03, weight decay0.01, grad clip1.0. Dropout은 껐고 하이퍼파라미터 탐색·중간 checkpoint 선택은 없었다.
- MPS, torch CPU threads2. 학습과 최종 평가까지 142.05초. 프로세스는 정상 종료하여 MPS 자원을 해제했다.

## 평가

원래 사람 한글 이름의 fold0 67척, fold1 63척에 대해 clean·삭제1회·중복1회·인접 교환1회를 평가했다. 총520행이다. 위치는 사전에 고정한 hash로 선택했다. 검색 점수는 선박별 최대 alias cosine이며 동점은 원래 catalog index 순서다. 원본·학습 모델 모두 동일한 전체 catalog와 수치 계산 경로를 사용했다.

| 범위 | 행 수 | 기준→학습 R1 정답 수 | 구출 / 훼손 | R10 | R100 |
|---|---:|---:|---:|---:|---:|
| 실제 clean | 130 | 118→119 | 3 / 2 | 126→127 | 130→130 |
| 삭제1회 | 130 | 66→61 | 2 / 7 | 99→103 | 117→118 |
| 중복1회 | 130 | 113→115 | 3 / 1 | 123→126 | 128→129 |
| 교환1회 | 130 | 77→77 | 5 / 5 | 101→106 | 117→121 |
| 전체 | 520 | 374→372 | 13 / 15 | 449→462 | 492→498 |

Fold0 전체 R1은196→190, fold1은178→182다. Clean 피해2척은 fold1의 `에스티오아젤리아`와 `샌재신토`다. 피해를 확인한 뒤 해당 이름에 맞춘 규칙이나 threshold를 추가하지 않았다.

R100의 순증6은 구출8·훼손2로 구성된다. 요청된 마지막 확인에서 구출8행을 기존112의 두 자음500 검색과 KoCat50에 넣었다. 양쪽 모두 전체 catalog를 사용했고 정답을 후보에 강제로 넣지 않았다. **8행 모두 기존 부분 합집합에 정답이 포함되어 있었다.** 따라서 후보 검색을 보완한 새 이득으로 셀 수 없다. 재판별기 결합이나 추가 학습은 하지 않았다.

## 해석 한계와 자료

이 결과는 이름만 사용하는 retrieval 실험이다. 실제 C_vote의 구출·훼손이나 Public 점수 예측이 아니다. 같은 KoCat 표를 학습과 prototype에 사용했으므로 표기 스타일 편향이 있다. Fold0/1의 human query를 이번 학습에 넣지는 않았지만 이 fold들은 이전 실험에서도 평가한 적이 있어 완전히 새 확인 자료는 아니다.

- `PROTOCOL.json`: 실행 전 고정한 조건.
- `run.py`, `run.log`, `TRAIN_HISTORY.json`: 재현 코드와1000step 기록.
- `RESULTS.json`: fold·variant·R1/10/100별 전체 결과.
- `frozen_rows.jsonl`, `adapted_rows.jsonl`: 모든 질의의 rank와 top100.
- `RETRIEVAL_OVERLAP.json`: R100 구출8행의 기존 검색 합집합 확인.
- `query_last2_final.pt`: 최종 last2 layer 가중치. 제출 후보로 승인한 파일이 아니다.
- `INPUT_MANIFEST.json`, `AUDIT.json`: 원본 입력 hash 및 변경 없음 확인.

기존 원본 가중치·embedding·Opus 파일을 수정하지 않았고, 제출 횟수도 사용하지 않았다.
