# 동결 C_vote 하위 모델 schema 감사

**결론: 현재 관측 데이터에서 광범위한 정답 손실을 설명할 feature 순서·모델 호출·한국어 metadata 매핑 버그를 찾지 못했다.** 0.98을 노릴 수 있는 수정 후보를 새로 확보한 감사는 아니다. 재현 가능한 입력 경계 문제 세 가지는 있지만 공식 train에서 해당 입력은 관측되지 않았다.

대상은 `astra/arena_20261101/baseline/model/{en,ko,hs}/`의 실제 동결112/v20/HSW다. 이미 검증한105~108류 라우터 들여쓰기·이전 행 누수 감사는 반복하지 않았다. 파일 수정, 모델 학습, GPU, 제출은 없다. 이 문서만 새로 저장했다.

## 1. 112 / en65 / s62 feature 순서

`en/model/opus_lib.py`의 FEATURES는60개, `opus_lib_65.py`는65개다. **65열의 앞60열 이름과 순서는60열 라이브러리와 정확히 같다.**

실제 LightGBM text header의 `feature_names`와 `max_feature_idx`를 전부 비교했다.

| family | 파일 수 | 실제 입력 | header 일치 |
|---|---:|---|---:|
| ranker | 6 | L60 | 6/6 |
| s62 | 6 | L65[:, :60] | 6/6 |
| en65 | 6 | L65 | 6/6 |
| kc113 | 6 | L60 + kc_sim + kc_margin | 6/6 |

`en/script.py:96`의 영어 branch에서 en65에는65열, s62에는그 앞60열을 전달하는 방식은 저장 모델 schema와 맞다. `s62`라는 이름 때문에62열을 기대하면 안 된다. 저장 모델은 실제60열이다.

짧은 실행 검사에서도 동일 후보3개에 대해 영문 이름, 한글 이름, NaN 이름, 숫자 문자열 네 종류의 `L.query_features()`와 `L65.query_features()[:, :60]`가 NaN 동등 비교까지 전부 일치했다. 생성 shape는각(3,60)/(3,65), infinity0이었다. 이는 feature parity 검사이지 정답률 평가가 아니다.

KoCat는62열 순서로 덧붙이며6개 kc113 파일이 모두 같은 이름을 기록한다. en65의 마지막5열은 한글의 톤수 창 상대 특징이므로 영어 입력에서0으로 남는 것은 코드 정의에 맞다.

## 2. v20 schema와 호출

- `ko/model/src/features.py`의 FEATURE_NAMES는78개다.
- 저장 `mappers.joblib['feature_names']`, `encoder_views.json['feature_names']`는현재78열 목록과 정확히 같다.
- 실행 시 `script.py` 및 `EncoderViewsEnsemble.load()`가 schema 불일치를 거절한다.
- 기본 앙상블의 두 LightGBM 파일도78열 이름·순서가 일치했다.
- 신경망 입력은현재 view 설정대로 Qwen/evidence와 encoder/phonetic을 사용한다. 기본 점수 상위20개를 동일 후보 순서로 전달하고 반환 점수를 그 위치에 넣는다.
- 기본 ensemble은질의별 확률을 합하고, encoder는top20 내부에서 다시 정규화한 뒤혼합한다. 기본 점수의 단순 합이 확률1이 아니어서 그대로 가중치를 잘못 적용하는 오류는 확인되지 않았다.

**미검증:** 세 CatBoost binary 모델을 직접 로드한 schema 확인은 하지 못했다. 현재 python3.11.9에 catboost가 없고 기존 설치 runtime도 확인되지 않았다. 새 설치는 하지 않았다. mapper/encoder 계약과두 LightGBM의 일치를 확인한 것을 세 CatBoost의 직접검사 완료로 확대하지 않는다.

## 3. HSW schema와 호출

현재 `matcher.FEATURES`는 59개다. 원래 모델과 rename model의 TreeEnsemble/RawAverage 구성원은 모두 `n_features_in_=59`이며, 최대 split index는 58 이하다. rank·한글 rank·rename rank의 LightGBM dump도 59개 열이다.

이 dump의 이름은 `Column_0..Column_58`이다. 따라서 저장 이름만으로 의미 순서까지 독립적으로 입증할 수는 없다. 현재 feature builder의 순서, 모델 호출 및 입력 폭에서는 불일치를 발견하지 못했다.

공식 카탈로그 3척과 한글·영문 질의 각 1개로 `QueryBatch→retrieve→features`를 실행했다. 두 행 모두 (3,59)를 생성했고 infinity는 0이었다. 영문행의 NaN 30개는 음역 관련 10열×3후보로, 코드가 의도적으로 비한글에 NaN을 넣는 구간과 일치한다. TreeEnsemble은 NaN을 `missing_left`로 처리한다.

## 4. metadata 매핑: 동결 모델과 합성 평가를 구분

국적·선종은 train의 한국어 범주→catalog의 영어 범주를 학습한 테이블이다. 입력 언어를 자동 번역하는 코드가 아니다. 실제 train 한글 1,191행으로 동결 테이블의 지원 여부를 확인했다.

| 모델 | 선종 지원 | 국적 지원 | 결측 해석 |
|---|---:|---:|---|
| 112 | 1,191/1,191 | 1,191/1,191 | pandas NaN을 문자열 `nan`으로 취급한 학습 키 포함 |
| v20 | 1,191/1,191 | 1,075/1,191 | 나머지 116행은 실제 국적 결측; mapper seen=0 |
| HSW | 1,191/1,191 | 1,191/1,191 | 학습 시 결측을 빈 문자열 키로 처리 |

`General Cargo`/`Panama`를 질의 쪽 범주로 넣으면 v20 mapper seen=0, HSW prob=-1이며 112에도 해당 query key가 없다. 따라서 catalog의 영어 metadata를 그대로 붙인 합성 질의는 실제 입항기록의 한국어 metadata와 같은 입력이 아니다. 이전 합성 성능을 철회한 판단과 일치한다. **이를 동결 C_vote의 광범위한 매핑 버그라고 부를 근거는 없다.**

## 5. 재현 가능한 입력 경계 문제

### 5.1 [P2, 관측 영향 없음] 112의 숫자 전용 이름 dtype이 배치에 따라 달라질 수 있다

위치: `en/script.py:30`의 `pd.read_csv(test_path)`는 텍스트 dtype을 고정하지 않는다. 이후 feature 함수는 문자열로 바꾸지만 파싱 때 잃은 앞자리 0은 복원하지 못한다.

재현: CSV에서 선명이 `00123`인 행만 읽으면 정수 123이고, 같은 행과 `OTHER` 행을 함께 읽으면 문자열 `00123`이다. 따라서 numeric-only batch에서 표현이 달라질 수 있다. 일반적인 숫자 123은 이후 `str()`로 같아지므로 모든 숫자 이름이 오류라는 뜻은 아니다.

공식 train의 숫자만으로 된 선명은 0행이다. 한글 손실 14개를 설명하는 증거가 없으며 새 고점 후보로 추천하지 않는다. v20/HSW는 텍스트 dtype을 명시한다.

### 5.2 [P2, 관측 영향 없음] NaN 선명이 112에서 문자열 `nan`으로 바뀐다

위치: `en/model/opus_lib.py:410`, `opus_lib_65.py:410`, `en/script.py:65`.

`str(float('nan') or '') == 'nan'`이므로 선명이 없을 때 빈 문자열 대신 영문 NAN 유사도를 계산할 수 있다. 정상 숫자형 톤수 NaN은 `positive()`에서 걸러지고 missing feature로 처리되므로 이 문제와 구분해야 한다. 공식 train의 빈 선명은 0행이다.

### 5.3 [P2, 관측 영향 없음] v20의 한글 판정은 NFD 입력을 직접 정규화하지 않는다

위치: `ko/model/src/common.py:23`.

`is_hangul(NFD('모닝'))`은 false지만 NFC로 복원하면 true다. 외부 root는 NFC로 판정하므로 v20로 전달된 NFD 한글이 내부에서는 비한글 경로를 탈 수 있다. v20의 다른 normalize 함수는 NFKC를 쓰지만 is_hangul보다 앞에서 항상 실행되는 것은 아니다. 공식 train에는 non-NFC 선명 0행, standalone jamo 0행이다.

## 6. 제외·동점 처리와 한계

앞선 구조 감사에서 세 모델의 제외 목록은 모두 공식 train 5,851척과 일치했다. 이번 읽기에서도 열 번호를 선박 ID로 잘못 읽거나 마스크를 다른 후보 순서에 적용하는 오류는 발견하지 못했다.

112는 점수 후 catalog index로 동점을 정렬한다. v20은 stable 후보 순서 및 argmax, HSW는 현재 후보 순서에서 argmax를 쓴다. 방식은 서로 다르지만 고정 catalog·단일 질의에서는 결정적이며 그 자체가 정답 손실 버그는 아니다. 세 모델 모두 모든 후보가 제외되는 경우 후보 전체로 복귀하는 방어 로직이 있다. 이것은 잠재적인 정답 불가능 조건이지만 현재 관측 자료에서 대규모 발생 증거는 없다.

이번 감사는 전체 GPU 추론·Public 재현·세 CatBoost binary schema의 직접 검사를 포함하지 않는다. 모델 형식, 실제 feature 값, metadata 테이블, 입력 경계 재현을 확인한 범위에서 큰 구조 결함을 고쳐 즉시 0.98에 갈 수 있다는 근거는 확보하지 못했다.
