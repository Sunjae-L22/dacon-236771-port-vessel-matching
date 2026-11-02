# 재현 가이드

대회 종료 후 기록용 문서입니다. 최종 선택 제출본은 **V5_S1S3**(Public 0.968471206, 한글 512/532, 영문 3506/3568)입니다.

이 레포에는 코드·노트북·문서만 있습니다. 대회 규칙상 데이터와 그 파생물은 공개하지 않으므로, 아래 **비공개 파일**이 따로 필요합니다.

## 필요한 비공개 파일

경로의 `<Drive>`는 소유자의 Google Drive `Projects/항만문제해결을위한AI기술공모전`입니다.

| 파일 | 위치 | 크기(바이트) | SHA-256 | 없으면 |
|---|---|---:|---|---|
| 대회 데이터 `open.zip` | `<Drive>/data/raw/` 또는 DACON | 1,873,016 | `2445fc26ade4e8b3bf4110d76296f7f36c6d907f554faa2b1308a30fdd8b5cc2` | 아무것도 재현 불가 |
| 파생 파일 묶음 `repro_private_20261102.tar.gz` | `<Drive>/repro_private/` | 64.7MB | `0f9645b709fe61138ced0443df56f522185457c8120d96a8f8c8b149097eadca` | LLM 생성물을 GPU로 다시 만들어야 하고, 같은 값은 보장 안 됨 |
| `112_submit.zip` | `<Drive>/combo_kit/` | 46,423,599 | `d8dd29d27b95429b7b949471ef19435fdf6eaed6c5b84eac689466f2210fce3f` | 아래 2단계로 재학습 |
| `v20.zip` | `<Drive>/combo_kit/` | 6,976,909,920 | 미기록 | **재학습 코드가 이 Mac과 레포에 없음** (아래 한계 참고) |
| `1031_08_hyb_w.zip` (HanSunwook) | `<Drive>/combo_kit/` | 7,237,515 | `e8361118ef075f87d1a4749d6a2b3d416fc1f6c78bdbd92fc104cb36187b2eab` | 외부 레포에서 재구성 필요 |
| `er_submit.zip` (Erde v42b) | `<Drive>/combo_kit/` | 19,562,903 | `8249b87d9b4ca5fcad4f72fc7d98492422b34ec7a27f9e72b2fa88efd719e585` | 외부 레포 `submissions/submit_v42b_fuzzy500`를 ZIP으로 묶음 |
| 완성본 `V5_S1S3.zip` | `<Drive>/combo_kit/out/` | 7,185,900,528 | 미기록 | 아래 1단계로 재조립 |

비공개 묶음에는 LLM 한글 표기·증강(`shared/llm_aug/`, `shared/kocat/out/catalog_ko.csv`), 고정 빌드별 CSV(`pinned/*/*.csv`), 3MB 이하의 실험 결과·로그가 있습니다. 대회 데이터 파생물이 들어 있으니 공개하지 마세요.

외부 레포 커밋:

| 이름 | 레포 | 커밋 |
|---|---|---|
| Erde | `ErdeKosmotale/dacon-port-vessel-matching` | `34e18eac3f3f762fcd8fa779209f1610241af4b5` |
| HanSunwook | `HanSunwook/vessel-analysis-project` | `6a96b62a3089b7be02f730a5bb51c88bb7be413a` |
| ksh (v18까지) | `ksh1004/DACON-202610-port-vessel-matching` | `f4cca66b229efc73e1fbe4def11ce0c586ad27e1` |

## 준비

```bash
git clone https://github.com/Sunjae-L22/dacon-236771-port-vessel-matching.git
cd dacon-236771-port-vessel-matching
mkdir -p data/open submissions
unzip open.zip -d data/open
tar xzf repro_private_20261102.tar.gz
python3 shared/experiments/relocate_paths.py
```

- `open.zip` 안의 폴더 구조에 따라 `data/open/train.csv`, `vessel_catalog.csv`, `sample_submission.csv` 위치를 맞추세요.
- 옛 스크립트에 `/Users/sunjae/Desktop/항만` 경로가 박혀 있어서 `relocate_paths.py`가 현재 위치로 바꿉니다.
- 환경: 학습은 Mac의 Python 3.11.9, 조립·점검은 Colab GPU(L4). Colab 패키지는 `lightgbm==4.6.0 catboost==1.2.8 rapidfuzz==3.14.3 transformers==4.57.6 accelerate==1.9.0 scikit-learn==1.5.2`입니다.
- `requirements-mac-snapshot.txt`는 2026-11-02 기준 Mac 환경 기록입니다. 학습 당시와 다를 수 있습니다.

## 1단계: 완성된 부품으로 V5_S1S3 조립 (약 15분)

Colab에서 `shared/experiments/combo_kit/8_V5_colab.ipynb`를 쓰거나 직접 실행합니다.

```bash
python3 shared/experiments/build_vote5.py V5_S1S3.zip v20.zip 1031_08_hyb_w.zip er_submit.zip 112_submit.zip 4 S1,S3
python3 shared/experiments/combo_kit/check_combo.py V5_S1S3.zip work_v5
```

`check_combo.py`는 ZIP과 같은 폴더의 `test_local3000.csv`, `vessel_catalog.csv`, `expected_93_local.csv`를 읽습니다(`<Drive>/combo_kit/`). 마지막 줄이 `OK`여야 합니다.

라우터 규칙: 영문과 110 점수차 4 이상인 한글 → 112. 나머지 한글 → v20. 단, HanSunwook == 112 == Erde ≠ v20(S1) 또는 112 == Erde ≠ v20이고 HanSunwook은 다름(S3)이면 112.

## 2단계: 데이터에서 112 재학습

각 단계의 명령은 당시 실제로 쓴 것입니다. **이번 정리에서 전체를 처음부터 다시 돌려 보지는 않았습니다.** LightGBM 학습은 환경이 다르면 바이트 단위로 같지 않을 수 있습니다.

| 순서 | 만드는 것 | 명령 또는 스크립트 |
|---|---|---|
| 1 | 37, 38 → 62 (영문 6랭커) | `pinned/build31/make_submit.py`, `KO_DICT=1 SEEDS=3 SEED_OFFSET=10000 / 15000`, `shared/src/merge_rankers.py` |
| 2 | 63, 64 → 65 (영문 6랭커) | `sh run_build63.sh` |
| 3 | 70, 71 → 72 (한글 6랭커) | `sh shared/experiments/run_70_72.sh` |
| 4 | 79 | `python3 shared/experiments/build_79_submit.py` |
| 5 | 92 (Super98·Clean18 포함) | `python3 astra/super98/build_92.py` (93의 답에는 Super98·Clean18이 쓰이지 않지만 ZIP 구조상 입력) |
| 6 | 93 | `python3 shared/experiments/build_93_decomp.py` |
| 7 | 110 | `python3 shared/experiments/build_110_fuzzy.py 110` |
| 8 | 한글 kc 랭커 6개 | 아래 명령 |
| 9 | 112 | 아래 명령 |

8·9번(`R`은 레포 경로):

```bash
for off in 0 5000; do num=$([ $off = 0 ] && echo 9113 || echo 9114); PYTHONHASHSEED=0 KO_DICT=1 SEEDS=3 SEED_OFFSET=$off KOCAT_TABLE=$R/shared/kocat/out/catalog_ko.csv OUT_DIR=shared/experiments/jamo_builds/out113_$off SUBMIT_NUM=$num python3 $R/shared/experiments/pinned/build113/make_submit.py; done
```

```bash
J=shared/experiments/jamo_builds; python3 shared/experiments/build_113_kocat_rank.py 112 shared/kocat/out/catalog_ko.csv $PWD/$J/out113_0 $PWD/$J/out113_5000 4
```

LLM 표를 새로 만들려면 `shared/kocat/4_catalog_ko.ipynb`(Qwen2.5-72B, GPU)와 `shared/llm_aug/*.ipynb`를 씁니다. 결과가 달라지므로 Public 점수 재현에는 비공개 묶음의 표를 쓰세요.

각 제출의 내용과 점수는 `submissions/README.md`, 실험 경위는 `shared/experiments/LOG.md`에 있습니다.

## 재현의 한계

- **v20은 데이터만으로 재학습할 수 없습니다.** 추론 코드는 `astra/new_direction_20261101/local_cvote/payload_mps/model/ko/`에 있지만 학습 코드와 가중치는 `v20.zip`뿐입니다. ksh 레포의 공개 커밋은 v18까지입니다. `v20.zip`을 지우면 V5_S1S3는 재현할 수 없습니다.
- **HanSunwook 1031_08과 Erde v42b는 제출 ZIP을 그대로 씁니다.** 외부 레포 커밋은 기록해 두었지만 이번에 재학습으로 검증하지 않았습니다.
- 서버 실행 시간은 조건에 따라 달라지고, V5에는 F1 같은 시간 가드가 없습니다.
- 모델은 GPU, 학습은 Mac 기준이라 다른 환경에서는 점수가 조금 다를 수 있습니다.
