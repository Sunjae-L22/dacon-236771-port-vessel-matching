# Channel 독립 QA — 2026-11-01

검토 대상: `astra/research_20261101/channel_model.py`, `astra/research_20261101/translit/normalized_channel.py` 및 실제 `NORMALIZED_FIT.json`/공식 카탈로그/KoCat 표의 스키마. 대상 코드는 수정하지 않았다. C_vote 실행이나 한국어 실험 결과를 읽고 판단한 감사가 아니다.

**판정:** 현재 고정 파라미터와 올바른 caller exclusion을 사용하는 정상 경로에서 후보 검색 누락, 확률 질량 오류, 이전 행 변수 누수를 찾지 못했다. 다만 통합 시 확인할 제외 목록 계약과 두 가지 입력 검증 취약점이 있다. 이 판정은 예측 개선이나 0.99 정답 확률을 보장하지 않는다.

## 발견 사항

### 1. [통합 시 필수] 제외 선박은 caller가 명시해야 한다

`channel_model.py:17`, `:24`, `:30`의 `excluded=()`가 기본값이며, `params['train_vessel_ids']`를 자동 적용하지 않는다. `self.ids`도 전체 카탈로그 ID 집합이며 제외 후 집합이 아니다. 이는 API 선택이지만 호출자가 학습 선박 제외를 자동으로 해 준다고 가정하면 평가 오염 또는 기존 라우터와 후보 공간 불일치가 생긴다.

확인한 `fresh_v2_eval.py:21`은 fit IDs를 명시하여 통과한다. 이번 실제 카탈로그 검사에서도 fit IDs 4,342개를 넘겼을 때 제외된 owner는 0개였다. **새 64척 token-composition 실행기와 최종 caller는 별도 검토 대상**이다. 오류 채널 fitting ID, 전체 공식 train ID, 평가 holdout ID는 서로 다른 집합이므로 어떤 집합을 제외하고 어떤 정답을 남기는지 manifest에 구분해야 한다. 모든 train ID를 무조건 제외하면 train 기반 평가 정답 자체가 사라질 수 있다.

### 2. [P2, 입력 검증] 확률 사전의 정확한 키를 검사하지 않는다

`channel_model.py:19`는 값의 합, 양수, 유한성만 검사한다. `operation_probabilities={delete:.2, dup:.2, swap:.2, typo:.4}`도 합이 1이므로 생성자가 받아들인다. 하지만 실제 계산은 세 연산만 열거하므로 총질량은 1이 아니다. step 사전에 `3` 같은 키가 섞여도 유사하다. 필수 키가 빠진 경우는 나중에 KeyError가 난다.

**현재 파일은 정상:** 연산 키는 정확히 `delete,dup,swap`, step 키는 정확히 `0,1,2`이고 값도 정상이다. 따라서 현재 결과를 무효로 만드는 발견은 아니며, 파라미터 파일 교체·패키징 실수를 앞에서 차단하는 방어 검사가 부족하다는 뜻이다. standalone `normalized_channel.score()`도 별도 파라미터 검증을 하지 않으므로 호출자의 계약에 의존한다.

### 3. [P2, 입력 검증] NaN score share는 교체를 통과한다

`channel_model.py:50`은 `<.99`만 검사한다. 직접 만든 trace의 `score_mass_share=float('nan')`는 이 조건이 false라서 `channel_dominant`로 반환된다. `+inf` 또는 1보다 큰 값도 교체 가능하다. 재현: 한글 행, baseline `BASE`, gap 1, winner `V1`, NaN share → `V1`.

**현재 Channel 정상 출력에서는 재현되지 않는다:** 정상 양수 유한 파라미터, 비어 있지 않은 aliases, 양수 total로 계산하므로 자연적으로 NaN을 만들 경로는 찾지 못했다. 저장 trace 재사용·다른 producer 연결 시에는 취약하다. `gap`은 이미 유한성을 확인한다.

### 4. [스키마 주의] unsupported 반환은 일부 필드가 없다

`query()`의 unsupported/비한글 반환에는 `supported_aliases`, `total_channel_mass`가 없다(`:33`, `:39`). 지원되는 반환에는 이 두 필드가 있다(`:41`). `decide()` 자체는 문제 없이 처리하지만 로그 writer가 모든 행에서 직접 인덱싱하면 KeyError가 생길 수 있다. `.get()` 사용 여부는 새 실행기에서 확인해야 한다.

또한 중복 `vessel_id` 및 중복 table `vessel_name`을 생성자가 거절하지 않는다. 전자는 alias mass를 중복 집계할 수 있고 후자는 마지막 행이 앞 행을 덮는다. **실제 입력에서는 중복 catalog ID가 없고 table name도 유일하여 현재 위험은 실현되지 않았다.**

## 후보 검색 완전성

- `process.extract(... scorer=DamerauLevenshtein.distance, score_cutoff=2, limit=None)`는 전체 aliases에서 거리≤2를 반환한다. top-K 절단이 없고 OSA를 쓰지 않는다.
- 삭제·중복·인접 교환의 각 event는 unrestricted Damerau-Levenshtein 편집 1회 이하로 나타낼 수 있다. 따라서 0/1/2-event 모델의 양수 확률 출력은 반드시 이 필터 안에 있다. **모델 support에 대해서는 완전한 prefilter**다.
- 필터가 통과시킨 후보 중 일반 삽입·치환처럼 채널이 허용하지 않는 변형은 `path_coefficients` 결과가 비어 질량 0이 되어 제거된다. 이 동작은 의도한 모델 정의와 같다.
- `top[:5]`는 보고용으로만 잘리며 winner와 denominator는 모든 양수 후보를 사용한다.
- 독립 확인: `가/나`로 만든 길이 0~4의 전체 31개 clean 문자열에서 모든 support 출력의 DL 거리가≤2였다. toy table의 18개 질의에서 전체 aliases 무필터 brute-force와 winner/share가 모두 일치했다.

## 확률 질량

- `event_distribution()`은 같은 결과를 만드는 여러 위치의 질량을 합한다. 동일 문자 swap, 빈 문자열, 길이1 swap의 identity 처리도 포함한다.
- 2회 event는 첫 event의 실제 중간 문자열에서 둘째 event를 적용한다. 따라서 겹치는 swap이나 복구 경로를 잃는 OSA 제약이 없다.
- zero/one/two event를 서로 다른 생성 경로로 보존하므로 같은 최종 문자열을 여러 경로로 얻었다고 해서 이중 계산 오류인 것은 아니다. 경로 확률을 주변화한 정상 동작이다.
- core 총질량을 31개 문자열의 완전한 support에서 합산: 최대 `|sum(P)-1|=2.220446049250313e-16`.
- 한 선박의 **정규화 후 중복 제거된** aliases에 uniform prior를 두고 `p / alias_counts[vid]`로 합한다. 거리 필터를 통과한 alias 수로 다시 나누지 않으므로 alias 선택 확률을 잘못 재정규화하는 오류가 없다.
- 동일 한글 alias를 여러 선박이 공유하면 각 선박이 별도로 경쟁한다. vessel ID 오름차순 tie-break로 결정적이다.
- `total_channel_mass`는 여러 후보의 likelihood 합이므로 그 자체가 [0,1]의 정답 확률이라는 뜻이 아니다.

## 0.99 score_mass_share의 정확한 해석

이 값은 **카탈로그의 제공된 alias mixture와 최대 2회 오류라는 모델 아래**, 동일한 선박 prior를 가정했을 때의 상대 질량이다. 다음 불확실성을 포함하지 않는다.

- 실제 정답의 적절한 한글 표기가 표에 없을 가능성.
- 3회 이상 오류, 다른 음역, 자모 수준의 혼동, 모델 밖의 수정.
- 영문에서 추정한 오류율이 한글에 그대로 적용되지 않는 차이.
- alias prior가 실제 언어사용 빈도와 다른 차이.

따라서 다른 후보들의 질량이 모두 0이면 잘못된 후보 하나도 share 1을 받을 수 있다. 0.99는 정답 보장, 1% 이하 오답률, calibrated confidence 또는 실제 C_vote 개선 확률이 아니다. 현재 `decide()` 주석은 이를 올바르게 명시한다. 현재 판정에서도 통계 개선 숫자를 확률 보장으로 해석하지 않았다.

실제 데이터 검사: catalog 78,595행/고유 ID 78,595개, 고유 이름 78,594개. KoCat 표는 78,594행/고유 이름 78,594개로 이름 기반 join이 일관된다. fit IDs 제외 후 74,253척과 120,812개 aliases가 남으며, **24척은 유효 한글 alias가 0개**다. 이들은 질량 denominator에서 사실상 제외된다. 또한 alias가 있어도 관측 질의에 대한 모든 0/1/2-event likelihood가 0이면 그 질의에서는 경쟁하지 않는다.

## 행 독립성 및 라우터 경계

- `query()`의 `q, mass, best_alias, alias_hits`는 매 호출마다 생성된다. 캐시, 이전 행 변수, batch-fitting은 없다.
- toy 18개 질의를 정순/역순으로 실행하여 전체 반환값이 동일했다. 기존 `test_normalized_channel.py`의 네 테스트도 모두 통과했다.
- non-Hangul은 baseline 유지, 한글 gap≥4도 baseline 유지, unsupported/불확실 share도 baseline 유지한다. English 경로에서는 gap을 읽지 않는다.
- 함수에 전달된 `gap`이 실제 **110 gap**인지는 이 두 파일만으로 확인할 수 없다. `decide()`는 출처를 검증할 수 없으며 caller trace hook 확인이 필요하다.
- `baseline`이 실제 C_vote 답인지, winner가 공식 평가 후보 공간에 포함되는지, 최종 CSV 순서가 입력과 같은지 역시 caller의 책임이다. 이 두 파일은 행 순서를 변경하거나 파일을 출력하지 않는다.
- `self.params`는 외부 dict 참조를 보관하므로 caller가 추론 중 이를 변경하면 이후 행의 결과가 달라진다. 현재 확인한 코드에서는 그러한 변경을 찾지 못했다. immutable frozen parameters로 사용할 계약이 필요하다.

## 실행 범위

코드 읽기, 정상 파라미터로 소형 exhaustive check, 실제 스키마/제외 목록 검사, 기존 4개 단위 테스트만 실행했다. GPU, Public 제출, 실제 C_vote inference, 모델 재학습, 대상 코드 수정은 하지 않았다. 이 보고서는 새 실행기 전체의 3,000행/600행 검증을 대체하지 않는다.
