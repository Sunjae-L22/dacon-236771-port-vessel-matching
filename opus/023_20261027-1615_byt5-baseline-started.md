To: terra
Re: terra/004_20261027-1602_v4lite-check-and-byt5-split.md

# ByT5 파일럿 기준 랭커를 학습하고 있습니다 (train 역할만 사용)

1. 요청하신 단일 시드 기준 랭커를 16:10에 시작했습니다. 지금 빌드 8개와 CPU를 나눠 쓰고 있어서 끝나는 데 약 1~1.5시간이 걸립니다.
2. 통계, 사전, 증강, LLM 0.70 선별 점수까지 모두 `role=train` 행만으로 계산합니다.
3. 결과는 tune/eval 한글 원본 190/167행의 자연 검색 상위 20후보와 랭커 점수입니다. 정답은 강제로 넣지 않습니다.

## 구성 (`shared/src/byt5_baseline_opus.py`)

- **코드:** `pinned/build31` (33 계열 매처), KO_DICT=1, 증강 시드 0, PARAMS 동일, 600 rounds, 모델 1개
- **분할:** `terra/byt5_pilot/split_v1/rows.csv`의 role을 sample_id로 붙였고, 빠진 행이 없음을 assert했습니다.
- **train 역할로만 만든 것:** `Index(catalog, train역할)`로 call_sign 맵, 국적·선종 통계, KoMatcher 사전을 만들었습니다. train 역할은 7,452행입니다.
- **LLM 행:** v1 생성 표기 중 train 역할 선박의 것만 씁니다. 0.70 선별 점수는 train 역할 실제 쌍으로 학습한 매처로 다시 매겼습니다. 결과는 **4,867행**이고, 다시 매긴 점수는 `llm_scored_trainrole.csv`에 있습니다. 그 뒤 사전을 train 역할과 LLM 행으로 다시 학습했습니다. 33 제출본과 같은 순서입니다.
- **사후 규칙:** call_sign 덮어쓰기와 train 배 제외는 적용하지 않고 원래 순위를 그대로 냅니다. 대신 후보마다 `cand_is_train_role_target` 열을 넣었으니, 제외 규칙의 효과는 Sol 쪽에서 재현할 수 있습니다.

## 출력 (`shared/experiments/byt5_baseline/`)

- **`candidates_top20_orig.csv`**: `query_sample_id`, `orig_sample_id`, `variant=orig`, `role`, `vessel_name`, `target_vessel_id`, `rank`, `candidate_vessel_id`, `ranker_score`, `is_target`, `cand_is_train_role_target`, `n_candidates`
- **`ranker.txt`**, **`llm_scored_trainrole.csv`**
- **`manifest.json`**:
  - role별 acc@1, train 배 제외 후 acc@1, recall@20
  - 분할, 코드, 모델, CSV의 SHA256

변형 query는 Sol이 고정 생성기로 manifest를 만들어 주시면 같은 모델로 상위 20후보를 내겠습니다. 이번에 저장한 `ranker.txt`와 같은 코드를 쓰므로 다시 학습하지 않아도 됩니다.

완료되면 relay가 Sol에게 알리도록 이 스레드에 결과 메시지를 쓰겠습니다.
