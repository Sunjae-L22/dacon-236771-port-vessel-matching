# 리드 인수인계 (Opus → Sol, 2026-10-27 16:15)

사용자의 사용량 사정으로, 10/27 16시부터 **Sol(Codex 대화 창)이 리드**를 맡습니다. Opus는 relay를 통한 자동 검토자로 남습니다(하루 3차례).

## 1. 지금 상태

- **최고점과 최종 선택:** 36번 Public **0.9403** (LLM v1 방법, 모델 12개 합본)
- **10/27 제출:** 하루 10회 중 36·39·42·55번으로 4회를 썼고, **6회가 남았습니다.** 55번은 채점 대기 중입니다. 55번은 36번에 Argon v4 매처만 바꾼 진단용이고, 재학습은 하지 않았습니다.
- 전체 기록은 `submissions/README.md`(제출 이력), `shared/experiments/LOG.md`(실험 로그), `shared/experiments/manifest.md`(zip 해시와 설정)에 있습니다.

## 2. 자동으로 진행 중인 빌드

이 빌드들은 Claude 세션과 무관하게 돌아갑니다(`nohup`). 진행 로그는 `shared/experiments/pipeline_47_54.log`입니다.

| 제출 | 구성 (모두 KO_DICT=1, 33과 같은 난수 쌍 0/5000) | 33(0.9397)과 비교하면 알 수 있는 것 | 예상 완료 |
| --- | --- | --- | --- |
| 47 = 43+44 | 기존 매처, 사전 v1+v2, 학습 v1+v2 | v2 데이터 전체의 효과 (42 재검증) | 17:30쯤 |
| 48 = 45+46 | 기존 매처, 사전 v1, 학습 v1+v2 | 학습 데이터만 늘린 효과 | 17:30쯤 |
| 53 = 49+50 | **v4-lite**(ch 수정 + LLM 사전 위생), 사전·학습 v1 | 두 수정을 합친 효과 | 21~22시 |
| 54 = 51+52 | 기존 매처, 사전 v1+v2, 학습 v1 (C) | 사전에만 v2를 넣은 효과 | 21~22시 |

- 47·48은 43~46 로컬 점검이 끝나면 자동으로 합쳐지고 점검까지 됩니다(`shared/experiments/merge_47_48.sh`).
- 49~52는 그 직후 자동으로 시작합니다(`shared/experiments/queue_49_54.sh`).
- **완료 확인:** `check_47.log`, `check_48.log`, `check_53.log`, `check_54.log`에 `zip size`와 `row independence ... mismatches 0`이 찍히면 사용자에게 제출을 안내하세요.
- **v4-lite가 이기면 다음 후보:** v4-lite에 사전·학습 모두 v1+v2를 넣은 버전입니다. v4-lite의 사전 위생이 v2 사전 오염을 막는 것은 확인했습니다(`글로리아→gloria` 유지, LLM이 기존 키를 덮어쓴 경우 0건).
- **ByT5 기준 랭커:** `shared/src/byt5_baseline_opus.py`가 실행 중입니다. 결과는 `shared/experiments/byt5_baseline/`에 나옵니다.

## 3. 빌드 방법 (반드시 지킬 것)

1. **코드는 고정 폴더에서만 빌드합니다.** `shared/experiments/pinned/buildNN/`을 복사해 고칩니다. 실행 중인 빌드가 쓰는 폴더(build43, build49)는 건드리지 않습니다.
2. **KO_DICT=1을 꼭 지정합니다.** 빠뜨려서 42번이 망가졌습니다. 그래서 build43 이후의 `make_submit.py`는 이 값이 없으면 멈춥니다.
3. **템플릿:** `shared/experiments/queue_49_54.sh`의 `launch` 함수를 참고하세요.
```sh
cd shared/experiments/pinned/buildNN && KO_DICT=1 LLM_DICT_CSV=llm_korean_scored.csv LLM_TRAIN_CSV=llm_korean_scored.csv \
  SUBMIT_NUM=NN SEEDS=3 SEED_OFFSET=0 OUT_DIR=submitNN CACHE_DIR=<root>/shared/experiments/cache \
  nohup sh -c "python3 make_submit.py && cd <root> && python3 shared/src/add_exclusion.py submissions/NN_submit.zip submissions/NN_submit.zip && python3 shared/src/local_check.py submissions/NN_submit.zip > shared/experiments/check_NN.log 2>&1" > shared/experiments/makeNN.log 2>&1 &
```
4. **같은 방법 두 묶음 합치기:** `python3 shared/src/merge_rankers.py out.zip a.zip b.zip`. 코드와 stats가 다르면 스스로 멈춥니다. 합친 뒤에는 `local_check.py`로 다시 점검합니다.
5. **기록:** `python3 shared/src/zip_manifest.py shared/experiments/manifest.md <zip들>`을 실행하고, README와 LOG에 한 줄씩 추가합니다.
6. **CPU 한도:** 맥은 10코어, 24GB입니다. 빌드는 동시에 4개까지만 돌립니다. 묶음 하나(모델 3개)는 4개 동시 실행 기준 약 2시간 걸립니다.

## 4. LB에서 확인한 교훈

- 같은 방법을 다시 학습하면 1쌍에서 0.004 차이가 났습니다. 그래서 비교는 **같은 난수 쌍(SEED_OFFSET 0/5000)의 6모델 합본끼리** 합니다.
- **train 배 제외는 사후에만 합니다.** 후보 검색 단계에서 빼면 −0.008이었습니다.
- 개명 증강은 15%가 맞습니다. 30%는 −0.018이었습니다.
- 같은 방법 모델 합치기는 6개까지는 크게 이득이고(+0.0085), 12개는 +0.0005로 거의 포화입니다.
- **로컬 스트레스 테스트는 LB를 잘 예측하지 못합니다.** 로컬에서 좋아진 변경이 LB에서 여러 번 졌습니다.

## 5. 자동 토론 relay 운영

- **실행:** `python3 shared/collab/relay.py`. 지금 백그라운드에서 돌고 있습니다. 재부팅 후에는 아래 명령으로 다시 띄웁니다. Codex 샌드박스 밖(일반 터미널 또는 전체 권한)에서 띄워야 Argon과 Opus 호출이 네트워크를 쓸 수 있습니다.
```sh
nohup python3 shared/collab/relay.py >> shared/collab/relay.log 2>&1 &
```
- **멈춤:** `shared/collab/PAUSE` 파일을 만듭니다. **Argon 끄기:** `shared/collab/ARGON_READY`를 지웁니다. **Opus 자동 호출 켜기:** `shared/collab/OPUS_HEADLESS`를 만듭니다. 사용자가 `claude` CLI 로그인을 마친 뒤에만 켭니다.
- **흐름 확인:** `shared/collab/timeline.md`(누가 누구에게 보냈는지), `shared/collab/relay.log`, `shared/collab/logs/`
- 규칙은 `shared/collab/AGENT_RULES.md`에 있습니다. 리드가 아닌 자동 차례는 분석과 답장만 합니다.

## 6. 아직 답하지 않은 메시지

- `argon/007_20261027-1522_reply-opus021-v4lite-oof-and-type-d-features.md`: Argon의 v4-lite OOF와 (d)유형 피처 제안. 자동 차례의 Sol이 지금 읽고 있습니다.
- `terra/005_20261027-1612_byt5-query-manifest-and-top20-policy.md`: ByT5 query manifest 제안. Opus 앞으로 온 메시지인데, 이제 리드가 판단하면 됩니다.
