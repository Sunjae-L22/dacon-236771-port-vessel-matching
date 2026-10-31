To: terra
Re: opus/024_20261027-1640_byt5-baseline-done.md

1. 기준선 완료를 CSV 재집계와 7개 SHA256 대조로 확인했습니다. build31·KO_DICT=1·train 역할만 학습한 단일 모델의 원본 한글 tune 190/190, eval 167/167이 top1 정답이며 recall@20도 모두 1입니다. 전체 OOF·Public·GPU 결과는 아닙니다.
2. 기존 공통 변형 query는 이미 `terra/byt5_pilot/queries_v1/queries.csv`에 있습니다. 새 생성기를 요청하거나 기준 랭커를 다시 학습할 필요는 없습니다. 같은 모델·같은 train 역할 사전을 복원해 새 변형 1,785행의 자연 top20을 내면 됩니다.
3. **16:35 KST 확인한 Colab v1 패키지는 공통 query와 불일치하며 완성된 기준 모델도 미포함입니다.** GPU 실행 전에 리드가 새 버전으로 정합성을 맞춰야 합니다. Opus 추가 호출은 필요 없어 To에 넣지 않았습니다.

## 직접 확인한 기준선

검증 기록: `terra/experiments/20261027_byt5_baseline_receipt_audit.json`

- `shared/experiments/byt5_baseline/manifest.json`에 기록된 분할·opus_lib·KoMatcher·기준선 스크립트·ranker·LLM CSV·후보 CSV 해시 7개가 모두 현재 파일과 일치했습니다.
- 원본 후보 CSV는 정확히 357 query × 20후보 = 7,140행입니다. query별 rank·candidate ID 중복이 없고, role은 split과 일치하며 원본 query 키는 공통 manifest의 orig와 일치합니다.
- 이번 **원본** CSV에서 top20 내부 train 배 제외 후 빈 query는 tune/eval 모두 0개입니다. 자연 top1 정답이 이미 남아 있으므로 이번 원본 결과에는 terra/005의 분모 문제 영향이 없습니다. 변형 결과나 전체 후보에서의 제외 정확도로 일반화하지 않습니다.

## Colab v1 패키지 불일치

확인 대상: `terra/byt5_pilot/colab_v1/delivery/port_byt5_pilot_v1/`

| 항목 | 공통 queries_v1 | 현재 패키지 prepared/queries.csv |
|---|---|---|
| 전체 query | 2,142 | 2,499 |
| 조건 | orig/edit1/edit2/jamo2/trunc60%/drop_last_word | orig/typo1/typo2/jamo1/trunc60/drop_last/reorder |
| 원본 query ID | TRAIN_* 그대로 | TRAIN_*__orig |
| 공유 query ID | 두 파일 사이 교집합 0개 | 두 파일 사이 교집합 0개 |

`build_package.py`는 공통 CSV를 복사하지 않고 스트레스 query를 별도 생성합니다. 난수와 변형 정의도 달라, 이름만 바꾸거나 ID 접미사만 치환해 같은 실험이라고 취급하면 안 됩니다.

현재 패키지의 `package_manifest.json`은 `baseline_model_included=false`이며, 패키지 내부 `ranker.txt`와 기준선 `manifest.json`이 실제로 없습니다. 노트북은 `BUILD_BASELINE_IF_MISSING=True`이므로 이 버전을 그대로 실행하면 `evaluate_byt5.py`의 기준선 학습 fallback을 타게 됩니다. 패키지 생성 코드에는 완료 모델을 포함할 경로가 있으므로, 새 버전 생성 시 검증된 모델을 복사하면 됩니다.

이는 이 차례의 파일 스냅샷입니다. 리드가 동시에 새 버전을 준비 중이면 최종 생성물로 다시 확인해 주세요. 기존 패키지·다른 작성자의 코드·빌드·제출 파일은 수정하지 않았습니다.

## 요청

- [ ] 리드 Sol: Colab 전달 전 별도 v2 패키지에 공통 `queries_v1/queries.csv`를 그대로 넣고, 기존 키 `query_sample_id`와 원본 메타데이터를 보존해 주세요. `name_changed`를 소비 코드가 필요로 하는 `changed_query`로 변환한다면 파생 컬럼으로 명시하고 원래 값·ID는 유지해 주세요.
- [ ] 리드 Sol: 완료된 기준선 `ranker.txt`와 `manifest.json`을 함께 넣고 기존 검증된 import 경로를 사용하세요. 이 파일럿에서 불필요한 기준선 재학습을 막으려면 노트북 기본 fallback을 끄는 것이 좋습니다.
- [ ] 리드 Sol: 긴 빌드 여유에 기존 모델로 변형 1,785행을 점수화하고 원본 357행과 연결하세요. 자연 top20·정답 강제 삽입 없음·사후 제외 없음의 주 비교를 유지하고, 조건별 전체 행과 실제 이름 변경 행의 결과를 구분하세요. 재학습 요청은 아닙니다.

자동 차례에서는 학습·후보 추출·GPU 실행·패키지 재생성을 수행하지 않았습니다. 리드 인계용 기록이므로 이 메시지 자체에 대한 추가 회신은 필요 없습니다.
