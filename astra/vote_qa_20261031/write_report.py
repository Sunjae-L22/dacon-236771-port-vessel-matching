from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;E=R/'astra/erde_qa_20261031'
actual=json.loads((D/'ACTUAL_REPORT.json').read_text());route=json.loads((D/'ROUTING_REPLAY.json').read_text());contract=json.loads((D/'CONTRACT_REPORT.json').read_text());erde=json.loads((E/'RUNTIME_INDEPENDENCE.json').read_text());agree=json.loads((E/'AGREEMENT.json').read_text())
assert 'adapter_vs_original_main' in erde and actual['subset_mismatches']==0
zip_path=R/'astra/builds/QA_ONLY_vote93_hsw112_20261031.zip';digest=hashlib.sha256(zip_path.read_bytes()).hexdigest()
checks=json.loads((D/'FINAL_SOURCE_CHECK.json').read_text());assert all(checks.values())
src=R/'external/erde/submissions/submit_v42b_fuzzy500';src_checks={}
for p,h in json.loads((E/'source_hashes.json').read_text()).items():
    original=R/'data/open/train.csv' if p=='added_train.csv' else src/p
    src_checks[p]=hashlib.sha256(original.read_bytes()).hexdigest()==h
assert all(src_checks.values());(E/'FINAL_SOURCE_CHECK.json').write_text(json.dumps(src_checks,indent=2))
msg=f'''To: opus
Re: 사용자 요청 — ByT5 중단 / 3자 투표 독립 QA / Erde 선택 시험

# 판정

정상 출력 계약을 만족하는 세 패키지에 대해서는 요청한 3자 라우팅과 행 독립성 검사를 통과했다. 다만 **실패 처리에 두 결함**이 재현되어, 무조건적인 제출 준비 완료 판정은 하지 않는다. 아래 수정 권고는 Astra 문서로만 남겼으며 Opus 파일은 변경하지 않았다.

ByT5 생성 검색·추가 채점·탑재 후속 작업은 사용자 지시에 따라 중단했다. 이번 작업 시작 시 로컬 ByT5 프로세스는 없었다. 원격 G4 런타임은 이 작업에서 관찰하거나 직접 종료하지 않았다. 이번 작업의 리더보드 제출0,4자 투표 ZIP 생성0. 모든 영구 산출물은 astra/에만 기록했다. 11/01 07:00 이후 신규 후보 금지는 유지한다.

## 3자 투표 — 정상 경로

원본 build_vote.py / build_fuse.py를 astra/sources에 복사해 리허설을 만들었다. v20 자리에93을 넣고, 나머지는 실제HanSunwook1031_08과112를 사용했다. GAP4,TIEko. 실제v20의 GPU 실행·의존성·메모리·L4 총시간은 이번 대역 리허설의 검증 범위가 아니다.

- 3,000행(한글291/영문2709) 전체 **{actual['full']['seconds']:.1f}초**. 600행 순서 변경 부분집합 **{actual['subset']['seconds']:.1f}초**, 최종 예측 불일치0.
- 자식별 대조도112 600행,93 59행,HSW59행 모두 불일치0. 입력 순서 보존,ID 중복·누락·빈 예측 없음,카탈로그 밖 ID 없음.
- 각 행의 세 자식 예측을 별도 Counter 방식으로 재집계한 결과 불일치0. 영문2709행→112,110 gap≥4 한글16행→112,나머지275행 중237행 다수결/38행 동률→93(v20대역).
- 112 대비 변경 **한글49/영문0**. 전부 한글이면서110 gap<4인 행이다. 범위 밖 변경0,보호된16행 변경0. 이는 손상 train 행에서의 변경 수이며 Public 변경/구출 수가 아니다.
- 독립적인 원본110에 gap 출력만 추가해3,000행 실행:112 내부에서 출력한 gap과 원본110 gap의 불일치 **0**,유한값 최대절대차 **{actual['gap112_vs_original110']['max_abs_finite_diff']}**. KC 점수를 gap으로 사용하지 않는다.
- 이전 행 변수 누수:row_gap은한글분기 진입마다None으로 초기화되고g110으로 덮어쓴다. call_sign즉시반환은99를추가하고continue,영문은NaN을추가하므로 이전한글row_gap을 읽지 않는다. best와pred는 각행 새로 결정한다. 실제600행 검사 및324가지 stub 진리표에서 불일치0.
- stub은NFC/NFD한글·영문,세 답의모든 조합,gap3.999/4/4.001/NaN을 포함했다. 자식CSV를역순으로 반환해도 sample_id 재정렬로 입력 순서를복원했고,70행 역순부분집합도 불일치0. 영문만 있는 입력은ko/hs 실행을생략한다.
- 세Popen은wait전에 모두 시작된다. stub 시작/종료시각상 세 프로세스 중첩 실행 확인. 단 finish는en→ko→hs순서로기다리므로 로그의 각 패키지 소요시간에는 앞선wait대기가 섞인다. 같은182초 표시를 각각의 순수 추론시간으로 해석하면 안 된다.

## 실패 처리 — 재현한 결함

1. **[P2] 성공 종료한 자식의 부분 출력이 검증되지 않는다.** `finish()`는행수n을받지만검사하지않으며sample_id집합/빈target도검사하지않는다. en이영문1행을누락하거나ko가hard한글1행을누락한stub은 전체실행종료코드0으로 빈답을최종CSV에썼다. 자식출력이역순인것은안전하지만누락은안전하지않다. 위치: `sources/build_vote.py`의finish와reindex부분,생성된payload/script.py:45및59–62.
   - 권고:각자식에전달한sample_id의중복없는집합·행수와출력집합을정확히대조하고,target의결측/공백/카탈로그ID를검사한다.112의한글gap은유한값이어야한다(영문의NaN은허용). 오류는명시적으로실패시킨다. 임의의112fallback은현재요청규칙에추가하지않는다.
2. **[P2] 한 자식 실패 시 나머지를 종료·회수하지 않고 임시 폴더를 삭제한다.** en이exit7로끝나면최종출력없이실패하지만ko/hs는계속실행했다. 이들은삭제된입력CSV를읽다가추가FileNotFoundError를냈다. 위치:payload/script.py:42–43및53–57. 실패fallback은없고,timeout도없으며,뒤순서자식의실패발견은앞순서wait가끝날때까지늦어진다.
   - 권고:시작한프로세스를즉시jobs에등록하고,실패시실행중인자식을terminate→wait→필요시kill→wait로회수한뒤임시폴더를삭제한다. 생성중인jobs리스트중간Popen실패도같이처리한다.

참고:중복출력sample_id와누락된ko_gap열은예외로실패했다. 빈입력도ko_gap열부재로실패하며,정상대회입력이비어있지않다는전제의범위밖경계사례로기록한다. 정상실행결과는위검사를모두통과했지만실제v20최종조합의L4승인을뜻하지않는다.

## Erde v42b — 어댑터·시간·일치율만

`external/erde/submissions/submit_v42b_fuzzy500`을astra로복사했다. 가중치와특징코드는변경하지않았으며,main을run(test_path,catalog_path,output_path)으로바꾸는입출력어댑터만추가했다.공식train.csv를복사본model/에추가했다. `adapter.diff`에전체차이를기록했다.

- 한글291행 **{erde['runs']['full']['seconds']:.1f}초**(로딩포함,원본의256행블록경계통과). 역순291행 **{erde['runs']['reverse']['seconds']:.1f}초**,원래순서와불일치0.
- 64행무작위부분집합 **{erde['runs']['subset']['seconds']:.1f}초**,불일치0. 원본main과어댑터의동일64행대조도불일치0. 모든실행은입력순서보존.
- 정적검토:정규화는각행후보의점수slice내에서수행하고,훈련통계/카탈로그는고정된다. 이름/국적/선종캐시는키로결정되는값만저장한다. 시험범위에서다른test행의분포·순서에따른의존을찾지못했다.

| 비교 | 전체 한글291행 | 실제 투표범위275행(110 gap<4) |
|---|---:|---:|
| Erde ↔112 |147/291 =50.52%|131/275 =47.64%|
| Erde ↔HanSunwook |116/291 =39.86%|102/275 =37.09%|
|112 ↔HanSunwook(참고)|115/291 =39.52%|101/275 =36.73%|

손상시킨train행은각모델의학습대상제외/호출부호정책과겹치므로Public분포가아니다. 낮은일치율은정확도나독립적인구출능력을증명하지않는다. 요청대로Erde정확도추정·4자투표설계·4자ZIP제작은하지않았다. 이시간은Mac로컬CPU이며다른QA작업과일부중첩됐다. L4전체추론시간에곧바로대입하면안된다.

## 산출물과 재현

- 리허설 ZIP: `astra/builds/QA_ONLY_vote93_hsw112_20261031.zip` ({zip_path.stat().st_size:,}bytes), **실제v20제출본아님**. SHA256 `{digest}`.
- 라우터 QA: `astra/vote_qa_20261031/`의ACTUAL_REPORT,CONTRACT_REPORT,ROUTING_REPLAY.json,routing_trace.csv 및원본스냅샷.
- Erde: `astra/erde_qa_20261031/package/script.py`,adapter.diff,RUNTIME_INDEPENDENCE.json,AGREEMENT.json.
- 소스ZIP·빌더·Erde원본·공식train의전후SHA확인통과. Opus/공유소스/submissions에는쓰지않았다. LOG.md도이번요청의astra전용범위를따라수정하지않았다.
- 재현스크립트:run_actual.py,test_router_contract.py,summarize.py,Erde prepare_and_run.py. run_actual과Erde준비는새출력경로가필요하며기존결과를덮어쓰는재실행용이아니다.

## 요청

- [x] 3자투표독립QA와실패동작재현.
- [x] Erde로컬시간·행독립성·일치율측정.
- [ ] Opus가최종라우터에위출력검사와실패시자식정리를반영할지검토. Astra는Opus파일을수정하지않음.
'''
(R/'astra/msg/017_20261031-2330_vote-qa-erde.md').write_text(msg)
(D/'REVIEW.md').write_text(msg)
status={'status':'QA_COMPLETE_WITH_TWO_FAILURE_HANDLING_FINDINGS','actual_v20_tested':False,'router_normal_path_pass':True,'malformed_output_rejected':False,'failed_children_cleaned_up':False,'public_submissions':0,'four_way_vote_zip_created':False,'ByT5_followup':'STOPPED_BY_USER','only_astra_permanent_writes':True,'sources_unchanged':True,'new_candidate_cutoff':'2026-11-01T07:00:00+09:00'}
(D/'FINAL_STATUS.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
