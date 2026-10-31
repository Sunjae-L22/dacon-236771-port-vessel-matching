# C_vote 고이득 방향 — 두 번째 Arena

3개 제안, 2라운드, 13개 Arena 작업 완료. 선택은 **기존 발음 인코더·Qwen으로 v20의 미평가 후보를 직접 재판별**하는 고정 실험입니다. 단순 톤수 규칙, KoCat K/가중치 조절, 네 번째 투표, ByT5, 새 학습은 제외했습니다. 현재 **HOLD: 실행 묶음 준비 완료, 실제 GPU 성능 미측정**입니다.

한글·원래110 gap<4·112≠HSW 행에서 기존 후보의 base top32+네 검색 유사도 top16+원래 답을 합친 최대97개를 인코더로 채점하고, 인코더 top8+base top20+원래 답의 최대29개를 Qwen으로 확인합니다. 두 모델의1위가 같고 각 raw margin≥1일 때만 v20 표를 바꿉니다. 최종 답은 원래 C_vote router로 계산합니다. 새 모델·표·검색기·학습은 없습니다.

정답이 풀 밖에 있거나112=HSW인 고정 다수결 오답, 보호행은 고칠 수 없습니다. 결과에 scope/pool/shortlist별 오류 상한을 별도로 기록합니다. Public K507/E3506에서0.98/0.985에는 순증14/18이 필요하지만, 해당 실제 오류가 이 범위에 있는지는 모릅니다. 현재 예상 Public 한글 정답 수 범위를 만들 근거는 없습니다.

## 실제 기준 실행 묶음

C_vote ZIP SHA 고정 → 원본 세 패키지 동시 실행·관찰 → 원본10행 역순 replay → 모든 trace ID·실제 winner·110gap 검증 → 원래 입력 행과 두 관찰 기록의 후보/점수/beam 대조 → 무변경 router replay → 추가GPU pass → hash가 맞는 COMPLETE 결과만 정답 평가.

한 패키지라도 실패하면 원본 fallback을 정상 기준으로 받아들이지 않습니다. stale/누락/중복/ABORT 결과를 구출로 집계하지 않습니다. 전체 진단1,800초, 추가 pass600초,07:00KST 중 빠른 제한을 적용하고 실패 단계의 경과시간·로그도 반환합니다. 원본 추론 파일은 외부 hook으로 관찰만 합니다.

공통104/400행은 최종 선택 전 ID해시로 고정했습니다. train·알려진fewshot·영문동명 선박을 제외했지만 기존 한글표 기반이므로 **SYNTHETIC / TABLE-SEEN / STYLE-BIASED**입니다. 같은64척의 여러 변형이며 unique-vessel 수도 보고합니다. 독립 검증이나 Public 예측이 아닙니다. 우승자의 별도256행을 결과에 따라 섞지 않습니다.

## 검증 결과

| 항목 | 결과 |
|---|---|
| 실제112/HSW CPU, 원본 vs계측8행 | 각각 불일치0 |
| 실제112/HSW 역순8행·단독1행 | 각각 불일치0 |
| KC 재채점4행 포함 원래110gap 계측 | 불일치0 |
| 원본ZIP 대비 로컬CPU모델/추론 소스56파일 | 모두hash일치 |
| 선택 adapter 회귀검사 |23/23통과 |
| trace 완전성·정합성 검사 |13/13통과 |
| 실제v20 제어흐름 + 테스트scorer, 이중hook | 출력·호출 수 동일, 역순·부분집합 동일 |
| 통합bridge 양성fixture |65위 후보가 최종답으로 이동, 영문변경0; 테스트 점수임 |
| 실패·중단 | stale/ABORT 거부, timeout child 종료 및 실패시간 기록 확인 |
| 실제C_vote GPU 구출/훼손 | 미실행 |
| 실제L4 추가시간·3,000행 시간 | 미측정 |
| 정식600행 불일치0 | 미실행 |
| 독립 평가의 깨끗한 이름 손해0 | 미확인 |

합성 이득만으로 제출 후보를 채택하지 않습니다. 제출용 모델ZIP은 만들지 않았습니다.

## 산출물

- `ASTRA_HIGHGAIN_KIT.zip`: 가중치 없는 실행 묶음. 제출 ZIP 아님.
- `package/ASTRA_HIGHGAIN_L4.ipynb`: 기존 Drive C_vote ZIP을 읽는 Colab 노트북.
- `package/EXPERIMENT.md`: 고정 방식·상한·실행 계약.
- `package/LOCAL_QA.json`, `SOURCE_PROVENANCE.json`: 검증 증거와 미실행 구분.
- `DELIVERY.json`: ZIP 크기·SHA256.

반환은 `ASTRA_HIGHGAIN_RESULTS.zip` 한 개입니다. 실제 C_vote 대비 구출/훼손, C_vote가112에서 고친 행의 재훼손, scope/pool 상한, L4 추가 비용을 판단합니다. 음성 결과이면 제출 없이 이 방향을 기각할 수 있습니다.

작업 astra/ 한정, Opus 원본·submissions/ 수정0, Public 제출0, ByT5 재개0, 제출용 모델ZIP0입니다.
