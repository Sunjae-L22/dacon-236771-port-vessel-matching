**최종 상태:** A2만 QA 완료·인계. A1/A3 clean 손해4로탈락, C1/C2 이득 미재현으로보류. 자세한 인계는 `../msg/009_20261031-2050_A2-ready-final-handoff.md`, 상태는 `FINAL_STATUS.json`.

# Astra noise113 (실험 폴더 번호; 제출 번호 아님)

원본 110을 변경하지 않는 단일 개입 실험. 사용자 요청 (a) 음절 노이즈 재학습과 (c) hard 규칙을 별도 ZIP/평가로 분리한다. Opus kocat·combo는 담당 범위 밖이다.

- 사전 계획: `../msg/005_20261031-1935_audit-and-preregister.md`.
- C 규칙 결과·제출 보류: `../msg/006_20261031-1955_second-name-hold.md`.
- `build_second_name.json`의 상태가 `QA_PASS_BUT_HOLD_NO_SUBMISSION`이다. 파일 존재를 제출 권고로 해석하지 않는다.

## 데이터·학습

`common.py`는110의 라이브러리를 바이트 검증해 스냅샷하고, 원래72/79 레시피와 동일한 기존 학습 캐시를 사용한다. fold별 사전·LLM 선택은 fit 선박으로 제한한 incumbent97 컨텍스트를 쓴다. full은110 stats의 ko_pairs와 동일함을 확인한다.

기존 학습블록·6개 증강시드·600트리·60피처를 보존한다. 원본 한글당3사본을 추가한다. 각 사본은1/2회 편집(.7/.3), 음절삭제/중복/인접교환(.35/.35/.30), 양쪽톤수결측(.5)이다. 새 블록에는 국적결측·자모치환·연속톤수왜곡을 추가하지 않는다. 원래 블록의 증강은 삭제하지 않는다. 새 블록과 추론에는110의 넓은 후보를 쓰고, 기존 캐시 블록은 원본72 후보 그대로다.

## 평가·채택

`eval_cache.py`는 native heldout 행별 clean/delete/duplicate/swap/mixed2 × tonnage keep/missing을 만든다. fold2는6모델,fold4는1모델 대조군이다. 둘 다 이전에 사용한 개발 fold이므로 새로운 봉인 테스트로 표현하지 않는다. 반복행과 고유 선박 수를 같이 보고한다.

`evaluate.py`는 clean 훼손0·scope밖변경0·noise 순이득>0을 다음 단계 기준으로 기록한다. 반복 확인에서 반대 결과가 나오면 추천을 보류한다. 합성 평가만으로 Public 개선을 확인했다고 쓰지 않는다. Public 연결 근거와 사전예측은005에 있다.

`decision.py`는 row-local 순수 함수이다. baseline 평균margin>=4는 보호하며, C1/C2는 이름점수 .15 차이 고정이다. `build.py`는 fold2와 fold4의 채택 조건을 모두 확인한 뒤110 기존payload를 보존하고 trace를 추가한다. C2는 확인평가 전 먼저 QA한 뒤 보류했으며, 현재 빌더로 재생성하려 하면 음성 확인결과 때문에 중단된다. `qa.py`는local_check와 실제baseline출력대조,변경집합포함관계,역순/교차배열/단독행을 검증한다. clean0은 숨은Public clean0을 보장하지 않는다.

## 실행

Mac TCC 문제를 피하려고 모든 긴 Python 프로세스는 `/tmp`에서 실행한다. 아래 ROOT는 실제 프로젝트 절대경로로 설정한다. 최대4개 학습/준비 작업을 유지한다. `run_jobs.py` 한 개는2개 작업을 동시에 사용하므로 여러 coordinator를 무심코 겹치지 않는다.

```sh
python "$ROOT/astra/noise113/run_jobs.py" fold2
python "$ROOT/astra/noise113/eval_cache.py" 2
python "$ROOT/astra/noise113/evaluate.py" 2 noise
# 평가 조건을 통과한 경우에만 제출용 채택
python "$ROOT/astra/noise113/run_jobs.py" full
python "$ROOT/astra/noise113/build.py" noise
python "$ROOT/astra/noise113/qa.py" noise
```

`prepare`는 자체 extra chunk를 재사용한다. 학습은600트리 모델·manifest를 확인한다. 초기 seed0 준비 중 공통JSON 임시파일 이름 충돌이 있었으며 PID별 임시파일로 수정 후 seed0을 재실행했다. 최초C ZIP 검증에서는ZipInfo 재사용이 메모리내 source offset을 바꾸는 문제가 있었으며,copy.copy로 수정하고 ZIP을 다시 빌드해25개기존payload의바이트동일을 확인했다. 원본 ZIP에는 쓰지 않았다. 현재 QA를 통과한 결과만 인계하며 초기 unverified ZIP은 이 실험 폴더 안에 격리했다.

local_check의corrupted-train정답률은train선박제외 때문에0일 수 있어성능판정에사용하지 않는다. 런타임과행독립성검사용이며 성능은선박분리holdout에서비교한다.

## 현재 후보 분기

- A1 (`noise`,3사본): fold2 noise+45지만 clean 손해4. 빌드 금지.
- A2 (`noise_with_tonnage`,3사본): hard이면서유효톤수있는행만교체. fold2 noise+8/clean훼손0,fold4단일모델 noise+4/clean훼손0. `007` 사전등록 후 확인. 전체6모델 패키징 및QA 완료.
- A3 (`--variant light`,1사본,rule=noise): `008` 사전등록. 미리생성된각seed의첫사본만선택해원학습블록과합친다. A2의톤수 gate와결합하지 않는다. `train_light.py`/`run_light.py`는별도light/ 경로에만가중치를쓴다. 평가·빌드·QA의`--variant light`도별도파일을사용한다.

A3는6모델 clean손해4가남아기각했다. 추가규칙/학습량 grid는하지 않는다. 사용자확인대로G4는Opus사용중이며이번작업은(a)·(c)에집중한다.

A3 seed6000의중복실행은동일한모델SHA를만들었으나중복matrix정리에서오류를냈다.6개모델해시는별도로확인했고A3는인계하지않는다. 재현시scope/seed마다coordinator하나만사용하고동일seed를직접중복실행하지않는다. A2실행QA와이실험실행문제는구분한다.
