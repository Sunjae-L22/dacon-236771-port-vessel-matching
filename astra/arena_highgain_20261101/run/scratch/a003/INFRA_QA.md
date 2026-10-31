# 공통 GPU 실행기 QA

검토 범위: package/trace_hooks.py, capture_baseline.py, evaluate.py, run_all.py, ASTRA_HIGHGAIN_L4.ipynb, README.md와 실제 baseline의 ko/en/hs 호출부. GPU와 무거운 테스트는 실행하지 않았으며 main 파일을 수정하지 않았다. run_candidate.py·manifest·EXPERIMENT.md는 선택 이후 완성 예정이므로 부재를 문제로 세지 않았다.

## 1. [P1] COMPLETE 이전에 HSW/v20 trace 완전성과 component winner 정합성을 강제해야 한다

- 위치: evaluate.py:24,34–40; trace_hooks.py:40,73,94.
- 현재 en trace 존재 및 gap 일치만 강제한다. jl()은 trace 파일이 없으면 빈 dict를 반환하며, v20 행이 없으면 오류 대신 trace_missing 카테고리가 된다. HSW 파일이 아예 없어도 현재 evaluator는 계속 진행한다. 정답을 맞힌 행은 trace가 없어도 baseline_correct로 분류되어 누락이 카테고리에조차 나타나지 않는다.
- 구체적인 반례: 정상 baseline 예측 CSV는 유지한 채 internal/hs_candidates.jsonl만 빠진 반환 묶음은 evaluate를 통과한다. union oracle은 HSW 후보를 누락한 채 계산되고, 뒤 단계가 끝나면 run_all은 COMPLETE를 쓸 수 있다. internal/v20_candidates.jsonl에서 baseline-correct 행 하나가 빠져도 마찬가지다.
- 수정: 평가 전에 expected Korean sample_id 집합과 en/hs/v20 trace의 집합이 정확히 일치하는지 검사한다. 원본의 무후보 fallback을 지원할 필요가 있으면 명시적 fallback record를 만들어 이유와 실제 출력 ID를 남기고, silent absence는 금지한다. 각 component CSV의 sample_id 집합도 기대 집합과 정확히 비교한다.
- 추가 정합성: trace의 후보/점수 길이, top20 부분집합, 유한 점수를 검사하고, v20는 원본 outer train mask를 적용한 argmax가 실제 ko_pred와 같은지, HSW는 allowed mask 후 argmax가 hs_pred와 같은지,112는 top20 첫 ID가 en_pred와 같은지 검사한다. 지금 기록된 input_fingerprint도 사용하지 않으므로 잘못된 입력 trace 결합을 검출하려면 동일한 모델별 정규화로 입력 fingerprint를 재계산해 비교하거나 허용 필드 원문을 함께 보관한다.

## 2. [P1 if total30min required / otherwise explicit limitation] 현재는 전체1,800초 예산이 아니라 단계마다 새로운 예산이다

- 위치: run_all.py:18–25,63,66,72; README의 시간 설명.
- run()은 호출마다 min(remaining_to_07, cap)으로 timeout을 새로 잡는다. baseline1,800초 + reverse replay1,800초 + candidate900초가 허용되어 총4,500초(+해시·추출·평가)까지 가능하다. 07:00 절대 마감은 model subprocess에 적용되지만 전체30분 watchdog은 없다. README는 단계별 제한을 정확히 설명하므로 문서 내부 모순은 아니다.
- 결정 필요: task의30분을 이 paired 연구 실행 전체의 상한으로 해석한다면 run_all 시작 시 monotonic deadline을 고정하고 모든 run timeout을 min(절대마감 잔여, 전역 예산 잔여, 단계cap)으로 제한해야 한다. baseline 검증만 예산을 소모했으면 후보를 부분 적용하지 않고 시간 부족으로 종료한다. 연구 진단 여러 패스는 별도시간을 허용한다면 README/STATUS에 total_experiment_budget_seconds=null, production_runtime_verified=false를 명시하고 전체 실제 소요를 별도로 기록한다. 현재 전체 실험을 서버30분 합격으로 해석해서는 안 된다.

## 3. [P2] 실패한 단계의 실제 소요시간과 시작시각이 반환물에서 빠진다

- 위치: run_all.py:63,66,72,79; capture_baseline.py:41.
- timings[key]=run(...)은 성공 반환 후에만 대입된다. timeout/nonzero이면 실패한 단계 timing은 없다. capture_baseline도 run_meta.json을 정상 완료 때만 쓰므로 실패할 때 launch PID/start 정보와 end-to-end 시간이 함께 사라진다. STATUS에는 finished_utc만 있어 실패시 실제 얼마나 기다렸는지 복원하기 어렵다.
- 수정: run()의 finally에서 stage name/start UTC/elapsed/returncode/timed_out를 stage receipt로 저장하고, run_all도 started_utc와 total_seconds를 finally에서 항상 기록한다. capture_baseline의 process launch 정보는 프로세스 시작 직후 또는 finally에서 기록한다. 이렇게 해야 결과 ZIP이 실패 원인뿐 아니라 GPU 재실행 비용을 알려 준다.

## 확인된 보존 동작과 남은 검증 한계

- capture_baseline.checked_finish는 원본 optional voter fallback이 반환한 빈 결과를 정상 baseline으로 받아들이지 않는다. exit code, 행수, 중복·누락·빈 target을 확인하고 finally에서 남은 자식을 정리한다. en의 SystemExit도 finally를 거친다.
- v20 hook은 원본 score_records를 그대로 호출하고 baseline score 반환값을 복사하여 관찰한다. source_hash 대상 파일을 수정하지 않고, scorer.score 및 baseline.score_groups를 finally에서 복구한다. train-mask/KoCat 적용된 closure를 감싸므로 점수 처리순서는 보존된다. 원본 top20 stable 순서도 맞다.
-112 hook의 preds.append 두 위치 및 locals(row_gap,allowed,use_kc)는 실제 en/script.py와 일치한다. HSW hook의 score_candidates와 allowed도 실제 pipeline API와 일치한다. 수치적 no-change 근거는 main이 이미 실행한 실제 CPU8행 plain/traced·역순·singleton 및 v20 scorer-double 검증에 한정되며 새 GPU 성능으로 확대하지 않는다.
- 프로세스별 start_new_session과 timeout 시 killpg는 baseline의 세 자식까지 종료하기 위한 적절한 구조다. 실패 후 finally의 STATUS/ZIP 생성 경로는 정상 파이썬 예외에서 작동한다. 런타임 자체가 소멸하거나 노트북 준비 셀(pip/업로드)에서 실패한 경우에는 WORK/결과ZIP이 아직 없어 마지막 다운로드 셀로 복구되지 않는다. README의 “실패했더라도 결과ZIP”은 run_all이 시작된 이후의 예외에 해당한다고 좁혀 쓰면 정확하다.
- 현재 no-change GPU 검증은 사전 고정10행 subset이다. 전체 원본 plain 재실행을 수행한 것으로 보고하지 않으며, actual production3,000행/600행 검증 미실시 표시는 맞다.
