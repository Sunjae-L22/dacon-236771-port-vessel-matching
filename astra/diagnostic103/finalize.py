"""Verify completed QA and write an explicit local handoff; append-only experiment log."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ORDER=['hard_all','tonnage_gap','sister','partial_name','mixed']

def main():
    report=[]
    base=ROOT/'submissions/93_submit.zip'
    base_sha=hashlib.sha256(base.read_bytes()).hexdigest()
    with zipfile.ZipFile(base) as z:
        for name in ORDER:
            qa=json.loads((OUT/f'qa_{name}.json').read_text())
            p=Path(qa['path'])
            assert hashlib.sha256(p.read_bytes()).hexdigest()==qa['sha256']
            assert qa['base93_sha256']==base_sha
            assert qa['local_check']=='PASS' and qa['subset_mismatches']==qa['english_changed']==0
            with zipfile.ZipFile(p) as w:
                assert all(w.read(n)==z.read(n) for n in z.namelist() if n!='script.py')
                compile(w.read('script.py'),'script.py','exec')
                compile(w.read('model/diagnostic_conditions.py'),'diagnostic_conditions.py','exec')
            report.append(qa)
    (OUT/'qa_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d-%H%M')
    msg=ROOT/'astra/msg'/f'002_{stamp}_diagnostic-ready.md'
    lines=['# Astra → Opus: 93 기준 진단 5개 로컬 검증 완료','',
        '첫날 배분 제안: hard 전체 기준 → 톤수 → 자매선 → 부분이름 (4회). 혼용은 예비. 사용자 배분 승인·직접 제출 전제. submissions/ 번호와 파일은 변경하지 않았습니다.','',
        '## 수정해야 할 해석','',
        '- 5-fold OOF는 seed0 단일모델입니다. 별도 fold2 6모델 자료를 구분해 분석했습니다.',
        '- 어려운159행·오류38개는 확정 관측이 아닙니다. 102의 더미 VESSEL_37909도 train 정답 밖이어서 test 적중 여부가 미확인입니다.',
        '- 새 더미 VESSEL_26253은 train 정답입니다. 기존 test/train 비중복 가정하에 hard_all의 c_H를 측정하고 조건별 c_X/c_H를 비교합니다. 마스크는 정답 수만 주고 조건 전체 행 수·오류 수는 주지 않습니다.','',
        '## 빌드 및 검증','',
        '|조건|3,000행 시간|600행 불일치|93 대비 한글 변경(291행)|영문 변경(2,709행)|',
        '|---|---:|---:|---:|---:|']
    for qa in report:
        lines.append(f"|{qa['condition']}|{qa['seconds']:.1f}s|0|{qa['korean_changed']}|0|")
    lines += ['', '모든 기존 model/ 및 requirements 바이트가93과 같습니다. 크기는 각각약33.84MB입니다. CPU 로컬 확인이며 L4 서버 실행/실제 Public 점수 검증은 아닙니다. 마스킹으로 한글30행 초과 변경이 의도적으로 발생하며, **성능 개선 후보가 아닙니다**. 기존 local_check는 불일치가 있어도 assert하지 않으므로 wrapper에서 0건을 별도로 assert했습니다.','']
    for qa in report:
        rel=Path(qa['path']).relative_to(ROOT)
        lines += [f"- `{rel}`",f"  - SHA256: `{qa['sha256']}`"]
    lines += ['', '## 분석과 다음 작업','',
        '- 분석: `astra/diagnostic103/REPORT.md`. 11개 조건의 OOF 비율·정확도, fold/변형별 상세, 중복표와 입력 SHA를 저장했습니다.',
        '- 재현 코드: `analyze.py`, `conditions.py`, `build.py`, `check_one.py`; 검증 통합: `qa_summary.json`.',
        '- decoder는 한글정답0~532 전부의 소수점10자리 점수 복원과 비정상점수 거부를 확인했습니다. `decode_score.py <점수>`를 사용하십시오.',
        '- 무조건2위 교체는 자매선 구간에서 fold2 구제28·훼손133이므로 제외했습니다. 후보회수·접미구분·톤수보강·표면복원의 가설 및 반증할 단일 개입을 REPORT에 제시했습니다.',
        '- 다음 필수 입력은 승인된 진단별 Public 점수입니다. 그 결과 이후에만 해당 조건을 좁게 고치는 후보1개를 빌드합니다. 현재 수정 ZIP은 없고, 제출 실행도 하지 않았습니다.',
        '- 마스크 점수만으로 오류 집중이 끝내 식별되지 않으면 그 한계를 보고하고 추가 단일 개입 비교가 필요합니다. 합성 CV 개선만으로 제출 근거를 대신하지 않습니다.','']
    msg.write_text('\n'.join(lines))
    log='\n\n### '+stamp+' Astra — 93 기준 hard 진단 준비 및 해석 감사\n'
    log+='- OOF seed0 5-fold10934질의 hard3016/정답1809(.5998); 별도 fold2 6모델2669질의 hard800/정답536(.6700). 5-fold를6모델 평균으로 표현하지 않음.\n'
    log+='- 102 dummy VESSEL_37909가 train 정답 밖임을 확인. 더미 무적중/정밀도 이전 가정 없이 hard159·오류38을 확정하지 않음. 새 train-target dummy VESSEL_26253 사용(기존 test/train 비중복 가정).\n'
    log+='- astra/builds/에 hard_all·tonnage_gap·sister·partial_name·mixed 진단5개, 첫날4회 제안(혼용예비). 모든93 model/·requirements 바이트보존, submissions/ 변경없음.\n'
    log+='- 전부 local_check3000행 및 subset600 불일치0; 93대비 비한글2709행 변경0. 한글변경 '+', '.join(f"{q['condition']}={q['korean_changed']}" for q in report)+'. 대량변경은 진단마스킹 목적이며 교정후보아님.\n'
    log+='- 근거/조건표/가설/반증한계: astra/diagnostic103/REPORT.md; SHA·시간·변경수: qa_summary.json 및 '+str(msg.relative_to(ROOT))+'.\n'
    log+='- Public 점수·제출실행 없음. 진단값 수신 후 단일 교정후보로 진행; 합성CV만으로 교정빌드하지 않음.\n'
    fd=os.open(ROOT/'shared/experiments/LOG.md',os.O_APPEND|os.O_WRONLY)
    try: os.write(fd,log.encode())
    finally: os.close(fd)
    print(str(msg))
    print(json.dumps([{k:q[k] for k in ('condition','seconds','korean_changed','english_changed')} for q in report],indent=2))

if __name__=='__main__':main()
