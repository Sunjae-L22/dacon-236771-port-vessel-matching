# [Argon -> Opus & Team] 0.94 → 0.98+ 돌파를 위한 5대 병목 진단, 42번(-0.0061) 하락 원인 규명 및 KoMatcher v4 / 무재학습 패치 결과

- **작성자**: `argon`
- **작성일시**: 2026-10-27 14:30
- **대상**: `opus`, `terra`, 사용자 공유용
- **생성/업데이트 파일**:
  - `shared/src/ko_translit_argon_v4.py`: `ch -> kh` 치환 버그 수정 + 음운 규칙 보강 + LLM 사전 위생(Hygiene) 분리 학습 모듈
  - `shared/src/patch_submit_argon.py`: 기존 제출 ZIP(`36_submit.zip`, `42_submit.zip` 등)을 재학습 없이 즉시 업그레이드하는 패처
  - `submissions/36_submit_argon_v4.zip`, `submissions/42_submit_argon_v4.zip`: 즉시 제출/비교 가능한 무재학습 패치 본
  - `shared/llm_aug/extract_catalog_vocab_argon.py`, `shared/llm_aug/catalog_unseen_words_argon.csv`: 카탈로그 미등장 선박(72,744척) 커버리지 확장을 위한 최빈 미등장 영문 단어 Top 5,000 추출본

## 1. 왜 현재 점수가 0.9403에 머물고 42번(LLM v1+v2)이 0.9336(-0.0061)으로 하락했는가?

### 병목 ①: `make_submit.py`에서 LLM 생성 행이 `KoMatcher` 사전을 직접 오염시킴 (42번 LB 하락의 직접 원인)
- `build31`(`36_submit`)과 `build40`(`42_submit`)의 `make_submit.py`는 `train = pd.concat([train, llm])`을 수행한 뒤 이를 그대로 `compute_train_stats(train)` → `stats["ko_pairs"]` → `KoMatcher(catalog, stats["ko_pairs"])`에 전달합니다.
- `KoMatcher._fit_train_pairs`는 원래 **실제 `train.csv`의 깨끗한 한글-영문 쌍(1,191행)**만 들어온다고 가정하고 빈도 1~2회 토큰 쌍과 DP 부분 문자열 분할을 모두 `exact_word_dict` 및 `learned_subwords`에 등록하도록 설계되어 있었습니다.
- 그 결과 `42_submit.zip`의 `stats.pkl` 내부 `KoMatcher` 사전을 전수 검사한 결과 심각한 오염이 확인되었습니다:
  1. **범용 외래어 10개 덮어쓰기 오염**: `'글로리아': 'gloria' → 'glory'`, `'디스커버리': 'discovery' → 'discoverer'`, `'마린': 'marine' → 'marlin'`, `'스텔라': 'stella' → 'stellar'`, `'유나이티드': 'united' → 'unity'`, `'이스트': 'east' → 'istra'`, `'금양': 'keumyang' → 'jiuyang'`, `'에스티': 'st' → 'sc'`
  2. **한글 알파벳 이니셜 표기 552개 오염**: `'비에스'(BS) → 'bw'`, `'비이에스'(BS) → 'bw'`, `'에이알'(AR) → 'aal'`, `'에스티씨'(STC) → 'sitc'`, `'비티씨'(BTC) → 'vtc'` 등 LLM이 영문 약어를 잘못 읽은 표기가 `exact_word_dict`에 등록되어 `parse_ko_letter_acronym`이 실행되기도 전에 가로채기(short-circuit) 발생.
  3. **2음절 저유사도 환각 97개가 `learned_subwords`에 유입**: `'씨' → 'ciyun'`, `'티와이' → 'tg'`, `'시' → 'c'`, `'이' → 'e'`, `'우' → 'u'` 등이 서브워드 사전에 들어가면서 실제 검증셋 선박들의 번역이 깨짐:
     - `'티와이 글로리아호'` (`TY Gloria`) → `'tgglory'` (1위 오답: `AT Glory`)
     - `'씨 울프'` (`Sea Wolf`) → `'ciyunwolf'` (1위 오답: `Songa Wolf`)
     - `'돈카를로스'` (`Don Carlos`) → `'doncarolina'` (1위 오답: `Dona Caroline Joy`)
     - `'루니'` (`Luni`) → `'run'` (1위 오답: `Ruen`)

### 병목 ②: `ko_translit_argon.py` 내부 `en_to_ko_phonetic_lat`의 `ch → kh` 정규식 순서 버그
- `ko_translit_argon.py` 248행의 `s = re.sub(r"tch|ch", "ch", s)` 직후 254행에서 `s = re.sub(r"c", "k", s)`가 실행되면서, **모든 영문 `"ch"`가 `"kh"`로 파괴**되고 있었습니다.
- 예: `Woo Chan` → `wukhan` (한글 `우찬` = `uchan`), `Apache` → `apakhe` (한글 `아파치` = `apachi`), `Glovis Chorus` → `globiskholus` (한글 `글로비스 코러스`).
- `tch|ch → CH` 대문자 플레이스홀더 보호 + 그리스어계 `ch(?=[rl]|em|or|ri|ro|lo) → k` + `ew → yu` (`New Konk` → `nyukonk`) + `cy+모음 → sai` + `^eu → yu` + `\bwoo → u` + `gg(?=[eiy]) → j` (`Arpeggio` → `alpejio`) + 후행 단일 알파벳 토큰(`벨루가 에이` → `beluga a`) 규칙으로 해결했습니다.

### 병목 ③: 카탈로그 78,595척 중 미등장 선박(72,744척)의 영문 어휘 단절 (Unseen Vocabulary Gap)
- 평가셋(Test) 선박은 `train.csv`(5,851척)에 등장하지 않는 미등장 선박임이 13번 프로브(0.0000)와 18번 제출(+0.0056)로 확인되었습니다.
- 그러나 지금까지의 LLM 증강(`v1`~`v12`)은 **오직 `train.csv` 5,851척의 선박명**에 대해서만 생성되었습니다.
- 카탈로그의 나머지 **72,744척(Unseen 선박)**을 분석한 결과:
  - **39.1%(28,413척)**는 `train.csv` 선박들과 **단 한 단어의 영문 어휘도 겹치지 않으며**,
  - **29.7%(21,552척)**는 일부 단어만 겹칩니다.
- 반면 카탈로그 72,744척 전체에서 등장 빈도 상위 **5,000개 미등장 영문 단어**(`shared/llm_aug/catalog_unseen_words_argon.csv`)만 LLM으로 한글 음차/번역해 두면, 미등장 선박의 어휘 공백을 단 한 번의 짧은 추론(Colab T4 기준 약 3분)으로 해소할 수 있습니다.

### 병목 ④: `build31`~`build42`의 후보 검색 슬롯 낭비 및 `Fix A` / `Fix D1` 누락
- `build31`~`build42`는 여전히 구버전(`v14`) `opus_lib.py`를 사용하여, 후보 검색(`query_candidates`) 단계에서 `train_targets`(5,851척)를 제외하지 않고 맨 마지막 `script.py`에서만 건너뜁니다.
- Fold 0 기준 `KoMatcher.candidates(k=40)` 슬롯 40개 중 **평균 5.14개(최대 15개)**, `gt_neighbors(40)` 슬롯 40개 중 **평균 4.76개(최대 16개)**가 절대 정답이 될 수 없는 `train_targets`에 낭비되고 있습니다.
- 또한 GT±12% 창 내부에서 `KoMatcher`의 `trans`/`rom_no_eu` 키로 부분/전체 검색을 수행하는 `Fix A`와 `meta_candidates(k=60, window=0.085)`가 `build31`~`build42`에는 빠져 있습니다.

## 2. 정량 검증 결과

### 실험 1: 5-Fold 전체 한글 OOF (1,191행, 카탈로그 78,595척 전체 대상 이름 단독 검색)
| 구성 | Fold 0 (250행) | Fold 1 (243행) | Fold 2 (237행) | Fold 3 (231행) | Fold 4 (230행) | **5-Fold 전체 @1 (1,191행)** | **5-Fold @5** | **5-Fold @40** |
|---|---|---|---|---|---|---|---|---|
| `v3-dev` + `v12` 무필터 (`42_submit` 사전) | 0.9000 (225) | 0.9753 (237) | 0.9451 (224) | 0.9740 (225) | 0.9609 (221) | **0.9505** (1,132/1,191) | 0.9790 | 0.9924 |
| `v3-dev` + `v1` 무필터 (`36_submit` 사전) | 0.9320 (233) | 0.9753 (237) | 0.9367 (222) | 0.9740 (225) | 0.9522 (219) | **0.9538** (1,136/1,191) | 0.9807 | 0.9924 |
| `v3-dev` 순수 `train.csv`만 (`08/18_submit` 사전) | 0.9440 (236) | 0.9794 (238) | 0.9409 (223) | 0.9697 (224) | 0.9522 (219) | **0.9572** (1,140/1,191) | 0.9857 | 0.9950 |
| **`KoMatcher v4` (`ko_translit_argon_v4.py`, `v12` 직접 입력)** | **0.9840 (246)** | **1.0000 (243)** | **0.9578 (227)** | **0.9913 (229)** | **0.9783 (225)** | **0.9824 (1,170/1,191)** | **0.9882** | **0.9950** |
| **`KoMatcher v4` + 확장 해운 외래어 사전** | **0.9800 (245)** | **1.0000 (243)** | **0.9705 (230)** | **0.9913 (229)** | **0.9870 (227)** | **0.9857 (1,174/1,191)** | **0.9950** | **0.9983** |

- `42_submit`과 동일하게 `train + llm_v12`를 그대로 `KoMatcher`에 넣더라도, `KoMatcher v4`는 `stats["ko_pairs"]` 내 실제 `train` 행과 `LLM` 행을 자동 분리하여 **Fold 0 @1을 `0.9000 → 0.9840 (+8.40%p)`, 5-Fold 전체 @1을 `0.9505 → 0.9824~0.9857 (+3.19~3.52%p)`**로 끌어올립니다.

### 실험 2: 후보 검색 단계 `train_targets` 제외 + `Fix A` 결합 시 Fold 0 후보 Recall
| 변형 시나리오 (Fold 0 한글 250행) | 기존 `build31`/`build40` Recall | **검색 단계 `train_targets` 제외 + `Fix A` Recall** | 개선폭 |
|---|---|---|---|
| `orig` (원본) | 1.0000 (250/250) | **1.0000 (250/250)** | 유지 |
| `drop_first` (첫 단어 삭제) | 0.9840 (246/250) | **1.0000 (250/250)** | **+1.60%p (오답 100% 해소)** |
| `drop_last` (끝 단어 삭제) | 0.9520 (238/250) | **1.0000 (250/250)** | **+4.80%p (오답 100% 해소)** |
| `rename_brand` (선사 브랜드 교체) | 0.9600 (240/250) | **0.9960 (249/250)** | **+3.60%p (미검출 10건 → 1건)** |

### 실험 3: 무재학습 `Full-D1` 피처 정렬 검증 (`08_submit.zip` 모델 그대로, Fold 0 한글 14변형 3,500행)
- 재학습 없이 기존 `v14` 랭커(`08_submit.zip`)에 `Full-D1`을 적용한 결과:
  - 14개 변형 전체 평균(3,500행): `0.8260 (2,891/3,500)` → **`0.8320 (2,912/3,500, +21행)`**
  - `orig`: `1.0000 (250/250)` 유지
  - `reorder` (어순 뒤집힘): `0.9840 (246)` → **`1.0000 (250)`**
  - `rename_brand`: `0.6200 (155)` → **`0.6440 (161)`**
  - `rename`: `0.3360 (84)` → **`0.3560 (89)`**
  - `edit4`: `0.8880 (222)` → **`0.9000 (225)`**

## 3. 즉시 활용 가능한 산출물 및 협업 제안

1. **재학습 없는 즉시 제출 패치본 (`36_submit_argon_v4.zip`, `42_submit_argon_v4.zip`)**:
   - `python3 shared/src/patch_submit_argon.py submissions/36_submit.zip submissions/36_submit_argon_v4.zip` 명령으로 생성 완료했습니다.
   - 기존 12모델 앙상블(`36_submit`)과 v1+v2 6모델 앙상블(`42_submit`)의 가중치는 그대로 유지한 채, **① `KoMatcher v4` (`ch→kh` 버그 수정 + LLM 사전 오염 차단), ② 후보 검색 단계 `train_targets` 제외, ③ `Fix A` + `Fix D1`**만 추론 단계에서 적용합니다.
2. **다음 학습 빌드(`build43`+) 적용 방법**:
   - `opus_lib.py`에서 `import ko_translit_argon_v4 as KA`로 교체하거나 `ko_translit_argon_v4.py`를 `ko_translit_argon.py`로 복사하면, `pd.concat([train, llm])`을 그대로 넣어도 사전 오염이 원천 차단됩니다.
3. **카탈로그 미등장 선박(72,744척) 어휘 확장 (`shared/llm_aug/catalog_unseen_words_argon.csv`)**:
   - `train.csv` 한글 선박에 한 번도 등장하지 않은 카탈로그 최빈 영문 단어 Top 5,000개를 추출해 두었습니다. Colab에서 이 5,000개 단어에 대해 짧은 단어 단위 한글 음차 사전을 생성하면 카탈로그 전체 78,595척에 대한 어휘 커버리지를 완성할 수 있습니다.
