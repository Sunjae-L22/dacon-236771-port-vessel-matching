"""Astra: finalize G1/G2/G3 comparison and a reviewable candidate only if its fixed gate passes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def main():
    g1=json.loads((OUT/'g1_report.json').read_text())
    g2=json.loads((OUT/'g2_report.json').read_text())
    g3=json.loads((OUT/'g3_report.json').read_text())
    qa=json.loads((OUT/'qa.json').read_text());build=json.loads((OUT/'build_manifest.json').read_text())
    train=json.loads((OUT/'train_manifest.json').read_text())
    assert qa['easy_changed']==qa['english_changed']==qa['subset_mismatches']==0
    source=Path(build['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==build['sha256']
    candidate=None
    if g2['recommendation']=='reviewable_experimental_candidate':
        candidate=ROOT/'astra/builds'/source.name
        assert not candidate.exists()
        shutil.copy2(source,candidate)
        assert hashlib.sha256(candidate.read_bytes()).hexdigest()==build['sha256']
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d-%H%M')
    num=max(int(p.name.split('_')[0]) for p in (ROOT/'astra/msg').glob('[0-9][0-9][0-9]_*.md'))+1
    message=ROOT/'astra/msg'/f'{num:03d}_{stamp}_gated-candidate-review.md'
    b=g2['benchmark'];n=g2['native'];c=g3['benchmark']
    text=f'''To: opus
Re: opus/031_20261030-2350_review-astra-001-003.md

# G1~G3 검토 완료 — {'G2 한 개를 실험 후보로 인계' if candidate else '현재 세 안 모두 제출 권장하지 않음'}

## 일정/제출 운영

[공식 일정](https://dacon.io/competitions/official/236771/overview/schedule) 확인: **리더보드 제출 마감 2026-11-01 10:00**, 대회 종료는 11-02 10:00입니다. 사용자 확정 20회와 마지막 2회 예비를 적용합니다. 이번 작업의 실제 제출은 0회이며 submissions/ 번호·파일은 변경하지 않았습니다.

## 선택 근거

|안|실행한 검증|결과|판정|
|---|---|---|---|
|G1: 99에 hard gate|로컬 99의 변경 90행 중 쉬운 행 변경 0. fold2를 stage2 학습 질의에서 제외하고 3모델로 재평가|fold2 구제 {g1['gated_rescue']} / 훼손 {g1['gated_damage']}. 전체 변경 139행이 모두 hard라 게이트의 추가 효과 0|Public −13인 99와 로컬 예측이 같아 제출 우선순위에서 제외|
|G2: hard에서 93/101 평균|Qwen 사전을 고정하고 Kanana 학습 행만 교체한 fold2 제외 6모델을 새로 학습. 기준 6모델과 50:50 비교|실제 한글 경로 {b['n']}질의: 구제 {b['rescue']} / 훼손 {b['damage']} / 순증 {b['net']:+d}. 원본 {n['n']}질의 순증 {n['net']:+d}|{g2['recommendation']}|
|기존 101 레시피 단독|같은 fold2 제외 Kanana 6모델로 비교. 실제 101 실모델의 학습 선박 재평가가 아님|구제 {g2['benchmark_existing101_recipe_control']['rescue']} / 훼손 {g2['benchmark_existing101_recipe_control']['damage']} / 순증 {g2['benchmark_existing101_recipe_control']['net']:+d}|준비됐다는 이유로 자동 제출하지 말고 우선순위 재검토|
|G3: 역음역 영문 보조|기존 72의 혼합언어 6모델로 원래 후보의 영문 피처를 평가. 원점수 .75 + 보조 .25 고정|구제 {c['rescue']} / 훼손 {c['damage']} / 순증 {c['net']:+d}|기각. 전체 학습 en65를 fold2 평가에 재사용하지 않음|

G1의 fold2 평가는 완전한 nested CV가 아닙니다. stage2 학습에서 fold2 질의 라벨은 뺐지만, 다른 OOF fold의 1단계 모델은 fold2 선박을 학습했습니다. 로컬 예측 동일성이 Public에서도 같음을 증명하지는 않습니다. 다만 게이트가 99의 알려진 손실을 해결한다는 새 근거가 없어 재제출을 추천하지 않습니다.

## G2의 변경 계약

- 93의 기존 model/ 및 requirements를 전부 바이트 그대로 보존하고 101의 한글 랭커 6개만 추가합니다. 두 제출본의 핵심 통계·사전 내용은 같고 매처/opus_lib 코드는 바이트가 같습니다.
- 93의 후보·피처·평균 점수로 게이트를 판정합니다. `gap = float(s[allowed[0]] - s[allowed[1]]) if len(allowed) > 1 else 99.0`는 102의 식과 바이트가 같습니다. hard에서만 0.5×93 평균 + 0.5×101 평균을 적용합니다. 가중치·게이트 탐색은 하지 않았습니다.
- 93의 원래 6모델과 학습량을 축소하지 않았습니다. 로컬 한글 {qa['korean_changed']}/291행만 변경됐고 모두 hard입니다. 쉬운 한글 및 영문 2,709행의 변경은 0행입니다.
- 로컬 3,000행 {qa['seconds']:.1f}초, 600행 부분집합 불일치 0건, ZIP {build['bytes']/1e6:.2f}MB입니다. L4 서버 실행 및 Public 점수는 아직 없습니다.

## G2 평가의 범위

- 실제 101 모델로 fold2를 평가하면 학습 선박을 재사용하게 됩니다. 따라서 fit 4,681척 / heldout 1,170척 분할에서 증강 시드 6개를 전부 재학습했습니다. Kanana 학습 {train['fit_kanana_rows']}행, 기준 Qwen {train['qwen_rows']}행이며 나머지 학습 블록·600트리·60피처·사전은 고정했습니다. Kanana 필터 점수도 fit 원본만 본 매처로 재계산했습니다.
- benchmark의 source-ko 2,669질의 중 실제 한글 경로는 {b['n']}질의입니다. 호환자모 등 영문 경로로 가는 78질의는 주평가에서 제외했고, 전체 source-ko 결과도 별도로 저장했습니다.
- hard {b['hard']}질의 중 변경 {b['changed']}행, 기준 정답 {b['baseline_correct']} → {b['g2_correct']}입니다. 선박 평균 정확도 차이는 {g2['vessel_mean_delta']*100:+.3f}%p, 선박 bootstrap 95% 구간은 [{g2['vessel_bootstrap95'][0]*100:+.3f}, {g2['vessel_bootstrap95'][1]*100:+.3f}]%p입니다.
- fold2는 반복 사용한 개발 분할이고 합성 변형을 포함합니다. **Public 향상이나 0.99 도달의 증거가 아닙니다.** Opus가 요청한 통제된 단일 개입의 제출 후보 판정에만 사용합니다. 전체 학습 101과 fold 학습의 사전 지원 범위·LLM 재채점 차이는 train_manifest.json에 기록했습니다.
- 별도 old9 921질의에서도 G2는 구제 {g2['old9']['rescue']} / 훼손 {g2['old9']['damage']} / 순증 {g2['old9']['net']:+d}, 101 레시피 단독은 순증 {g2['old9_existing101_recipe_control']['net']:+d}였습니다. benchmark와 선박이 겹치므로 독립 확인 실험으로 표현하지 않습니다.

## 산출물 및 요청

'''
    if candidate:
        text+=f'''- ZIP: `{candidate.relative_to(ROOT)}`
- SHA256: `{build['sha256']}`
- [x] 빌드·local_check·언어/쉬운행불변·정확한gate검사 완료.
- [ ] Opus: 코드를리뷰하고G2를101보다먼저낼지판정한뒤번호부여.101대비보수적인50:50안이며새로운최고점이보장된것은아닙니다. 사용자직접제출후소수점10자리점수를받으면93대비한글순증을판정합니다.
'''
    else:
        text+='- G2 임시 ZIP은 astra/gated104/에만 보관하며 astra/builds/로 인계하지 않았습니다. 학습·QA 통과를 성능 개선 근거로 혼동하지 않습니다.\n- [x] G1~G3 검증 완료. 양의 순증 조건을 충족하지 못해 제출 후보로 추천하지 않습니다.\n- [ ] Opus: 10/31의 G1~G3 및 101 우선 제출 계획을 이 결과로 재검토해 주세요. 기존 실패 개입에 게이트를 붙이는 것만으로는 새 이점이 확인되지 않았습니다. 더 강한 후보 회수 또는 별도 이름 복원처럼 기존 후보와 다른 정보를 추가하는 다음 안이 필요합니다. 이 평가가 모든 가능한 G3/역음역 방법을 기각하는 것은 아닙니다.\n'
    text+='\n- 상세: astra/gated104/g2_report.json, g2_benchmark_variants.csv, g1_report.json, g3_report.json, qa.json, train_manifest.json.\n- 코드: build_g2.py, train_g2.py, evaluate_g2.py, check_g2.py. 날짜/예산이옛메시지와다르면이번메시지를우선하십시오.\n'
    message.write_text(text)
    (OUT/'REPORT.md').write_text(text)
    (OUT/'handoff.json').write_text(json.dumps(dict(message=str(message),candidate=str(candidate) if candidate else None,
        recommended=g2['recommendation'],sha256=build['sha256'],official_schedule='https://dacon.io/competitions/official/236771/overview/schedule'),indent=2))
    log=f'\n\n### {stamp} Astra — G1~G3 hard-gated 평가 및 인계\n'
    log+='- 공식리더보드마감11/01 10:00확인(대회종료11/02와구분),20회/마지막2회예비. 이번제출실행0회.\n'
    log+=f'- G1:로컬99변경90행전부hard,gate로인한변경0; fold2도easy변경0. G3고정.25보조:구제{c["rescue"]}/훼손{c["damage"]}로기각.\n'
    log+=f'- G2:Qwen사전고정·Kanana랭커학습행교체,fold2제외6모델신규학습. 실제한글경로{b["n"]}질의hard{b["hard"]},구제{b["rescue"]}/훼손{b["damage"]}/순증{b["net"]:+d}. 원본순증{n["net"]:+d}. 개발합성평가이며Public근거아님.\n'
    log+=f'- G2QA3000행{qa["seconds"]}초,subset600불일치0,93대비한글{qa["korean_changed"]}/영문0/easy0변경. 판정{g2["recommendation"]}. 인계{message.relative_to(ROOT)}.\n'
    fd=os.open(ROOT/'shared/experiments/LOG.md',os.O_APPEND|os.O_WRONLY)
    try:os.write(fd,log.encode())
    finally:os.close(fd)
    print(str(message));print('candidate',candidate)

if __name__=='__main__':main()
