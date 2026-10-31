"""Author: Astra. Final immutable-ZIP QA receipts and append-only experiment record."""
from pathlib import Path
import json,re,hashlib
ROOT=Path(__file__).resolve().parents[1]

def main():
    rank=json.loads((ROOT/'astra/ranker84_oof_report.json').read_text())
    less=json.loads((ROOT/'astra/rankless87_oof_report.json').read_text())
    neural=json.loads((ROOT/'astra/neural87_oof_report.json').read_text())
    qa={}
    for n in [87,88]:
        log=(ROOT/f'astra/check{n}.log').read_text()
        assert 'row independence: subset 600 rows, mismatches 0' in log,(n,log)
        seconds=float(re.search(r'rows 3000  time ([\d.]+)s',log)[1])
        path=ROOT/f'astra/build{n}_manifest.json';receipt=json.loads(path.read_text())
        assert hashlib.sha256((ROOT/f'submissions/{n}_submit.zip').read_bytes()).hexdigest()==receipt['sha256']
        qa[n]=dict(status='PASS',full_rows=3000,subset_rows=600,mismatches=0,full_seconds=seconds)
        receipt['local_check']=qa[n];path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    result=dict(incumbent=dict(submission=79,public_lb='0.9418190937'),rejected_84=dict(public_lb='0.9312927779',delta='-0.0105263158',implied_ko_change=-8,assumption='previous 532/3568 denominator inference and unchanged English route'),priority=[88,87],qa=qa,ranker87=rank['all_five_folds'],ranker88=less['all_five_folds'],ranker88_confirmation=less['primary_confirmation_folds_3_4'],native88=less['native_all_rows'],neural_delta=neural['baseline_fusion']['delta'],neural_adopted=False,gpu='Colab session manager showed no active sessions; no GPU allocated in this follow-up',limits=['New ZIPs have no Public LB measurement','CV baseline is a paired Korean-only single-seed model, not full-data submission79 ensemble','Synthetic stress distribution does not estimate hidden-test distribution','Rankless versus ranked mean difference is small and uncertain','Rename and some reorder regressions remain'])
    (ROOT/'astra/results87_88.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    readme=ROOT/'submissions/README.md';existing=readme.read_text()
    entries={
        87:f'| 87 | 2026-10-29 | Astra: 한글 전용 71피처 랭커 재학습(후보 확장·평활 메타데이터), 영문79 유지; KO_DICT=0 | 선박분리5-fold 382척×9조건 .804828→.824607; 확인fold1–4 +1.785%p, 원본1,191행 .995802→.996641. 3,000행 {qa[87]["full_seconds"]}초·600행불일치0·25.89MB | 미제출·Public 미측정; 비교용 |',
        88:f'| 88 | 2026-10-29 | Astra: 87 레시피에서 rank_name 제외한 70피처 재학습, 영문79 유지 | 선박분리5-fold .804828→.826062, 다섯fold 모두 개선; 독립 확인fold3–4 +2.412%p(CI +.877~+4.020%p); 원본1,191행 .995802 유지. 3,000행 {qa[88]["full_seconds"]}초·600행불일치0·25.90MB | 미제출·Public 미측정; 우선 실험 후보(최고79 유지) |'
    }
    with readme.open('a') as f:
        for n,line in entries.items():
            if f'| {n} | 2026-10-29 | Astra:' not in existing:f.write(line+'\n')
    log=ROOT/'shared/experiments/LOG.md';marker='## 2026-10-29 Astra — 84 실패 후 5-fold 검증과 87/88'
    if marker not in log.read_text():
        with log.open('a') as f:f.write('''\n## 2026-10-29 Astra — 84 실패 후 5-fold 검증과 87/88
- 사용자 보고84 Public .9312927779, 최고79 .9418190937 대비 −.0105263158. 기존분모·영문유지 가정에서 한글 −8건. 84 기각; 83·86은 사용자 확인 미제출.
- 단일 52척 개발홀드아웃만으로 채택한 한계를 보완: 한글 고유382척 선박분리5-fold, 9개 원본/손상 조건, 별도 원본1,191행 전체 평가. 각fold의 사전·메타·LLM 필터/재채점·신경망은 학습선박만 사용. 기존baseline 및 수정모델은 동일 단일seed/학습쿼리로 비교.
- 후보확장+메타11피처 재학습(87): 5-fold .804828→.824607(+1.978%p), 확인fold1–4 +1.785%p, 선박bootstrap95[+.546,+3.024]%p. 모든fold 개선. 개명 조건 .460733→.447644 회귀. 원본1,191행 1186→1187 정답.
- rank_name 제거(88): 5-fold .804828→.826062(+2.123%p), 확인fold3–4 +2.412%p, 선박bootstrap95[+.877,+4.020]%p. 모든fold 개선, 원본1,191행 1186정답 유지. 개명·순서 일부회귀 남음. 87 대비 평균차 +.145%p는 CI에0 포함, 우월성 확정 아님.
- 직접학습 hard-negative 문자/자모 encoder: 2,897학습선박에서 이름/톤수 유사 오답 채굴, pilot +.524%p. 그러나 fold별재학습5-fold는 기존랭커 +.087%p(CI0포함), 수정랭커 +.029%p → 채택 안 함. 제출87/88에는 신경망 없음.
- arrival_year 기반 메타확률은5-fold 국적 NLL악화, 선종효과작아 미채택. 공개 Helsinki-NLP/opus-mt-ko-en(고정revision e42d1f41b66194e6d10512f8a27bebc1f4f5097e)로 train 국적/선종만 번역한 prior도 자동대응오류로 미채택. 실제 catalog 선종은 Pass/Pass-Car계열/PCC/GCargo 등 축약어이며 브리핑의 Passenger/Ferry 풀네임과 다름. 수동 보정매핑 없음.
- 학습4개 프로세스가 exit137로 종료(원인미확정). 기준모델재사용·500쿼리별압축피처체크포인트·디스크행렬로 복구, 동시주요학습2개로 축소. 501그룹 경계/NaN/라벨/재시작일치 검사PASS. 원본data/기존ZIP/Argon코드 수정없음.
- 87/88 mandatory local_check 3,000행 각각151.0/161.7초, 600행불일치각0. ZIP CRC/해시/보존모델바이트검사PASS. 79대비87 영문200행예측차이0, 87/88영문분기AST동일.
- Colab 활성세션없음 UI확인. 이번후속실험은로컬실행, 추가GPU할당없음. Public 새점수미측정이므로 최고79 유지. 88우선실험후보,87비교용; 0.98달성주장없음.
- 결과: astra/results87_88.json, ranker84_oof_report.json, rankless87_oof_report.json, neural87_oof_report.json. 87/88의 CV는기존전체데이터79앙상블과직접비교한수치가아님.
''')
    print(json.dumps({'qa':qa,'priority':[88,87],'public_best_unchanged':True},ensure_ascii=False))

if __name__=='__main__':main()
