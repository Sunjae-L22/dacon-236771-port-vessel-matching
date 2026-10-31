from pathlib import Path
import json
R=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
md('''# Astra: C_vote 고이득 방향 검증 (제출 없음)
실제 C_vote ZIP의 세 모델을 실행하고 고정된 실험 하나만 비교합니다. 새로운 제출 ZIP을 만들지 않습니다.

**L4 권장.** 원본 C_vote.zip은 Drive의 기존 경로를 사용합니다. 먼저 `ASTRA_HIGHGAIN_KIT.zip`을 준비하세요.
검증 입력은 104행 pilot / 400행 full인 **합성·카탈로그 표기 노출 데이터**입니다. 독립 검증이나 Public 점수 예측이 아닙니다.
07:00 KST(2026-11-01) 이후 실행을 시작하거나 계속하지 않습니다. GPU 실행 결과가 없으면 개선은 미확인입니다.
''')
code('''import datetime, pathlib, subprocess, sys, time
CUTOFF = datetime.datetime(2026, 10, 31, 22, tzinfo=datetime.timezone.utc).timestamp()
assert time.time() < CUTOFF, "07:00 KST 실험 마감이 지났습니다."
from google.colab import drive, files
drive.mount('/content/drive')
import torch
assert torch.cuda.is_available(), "런타임을 GPU로 바꾸세요."
print(torch.cuda.get_device_name(0))
''')
code('''# 이 대화에서 받은 작은 실행 묶음 ZIP을 업로드합니다 (C_vote 7GB를 다시 업로드하지 않습니다).
import zipfile, os
BASE = pathlib.Path('/content/astra_highgain')
BASE.mkdir(parents=True, exist_ok=True)
os.chdir(BASE)
uploaded = files.upload()
kit_names = [n for n in uploaded if n.endswith('.zip') and 'HIGHGAIN_KIT' in n]
assert len(kit_names) == 1, "ASTRA_HIGHGAIN_KIT.zip 한 개를 선택하세요."
KIT = pathlib.Path('/content/astra_highgain/kit')
KIT.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(kit_names[0]) as z:
    for info in z.infolist():
        target = (KIT / info.filename).resolve()
        assert target.is_relative_to(KIT.resolve())
        assert ((info.external_attr >> 16) & 0o170000) != 0o120000
        z.extract(info, KIT)
assert (KIT / 'run_all.py').exists()
''')
code('''assert time.time() < CUTOFF
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'lightgbm==4.6.0', 'catboost==1.2.8', 'rapidfuzz==3.14.3',
    'transformers==4.57.6', 'accelerate==1.9.0', 'scikit-learn==1.5.2', 'sentencepiece'], check=True)
''')
code('''# 기본 경로에 없다면 C_VOTE만 실제 Drive 경로로 고치세요.
C_VOTE = pathlib.Path('/content/drive/MyDrive/Projects/항만문제해결을위한AI기술공모전/combo_kit/out/C_vote.zip')
SPLIT = 'pilot'  # 기본 104행. 사전 고정 400행 전체는 'full'. 결과를 보고 파라미터를 바꾸지 않습니다.
assert C_VOTE.exists(), f"파일 없음: {C_VOTE}"
assert time.time() < CUTOFF
WORK = pathlib.Path('/content/astra_highgain') / ('run_' + time.strftime('%Y%m%d_%H%M%S'))
p = subprocess.run([sys.executable, str(KIT/'run_all.py'), '--cvote', str(C_VOTE), '--work', str(WORK), '--split', SPLIT])
print('종료 코드:', p.returncode)
print('실패했더라도 다음 셀에서 결과 ZIP을 받아 전달하세요.')
''')
code('''result = WORK/'ASTRA_HIGHGAIN_RESULTS.zip'
assert result.exists(), f"결과 파일 없음. 마지막 오류를 확인하세요: {WORK}"
print((WORK/'results/STATUS.json').read_text())
files.download(str(result))
''')
md('''## 반환
`ASTRA_HIGHGAIN_RESULTS.zip` 한 개를 이 대화에 첨부하세요. 세 모델 예측, 후보/점수, 원본 재실행 일치 검사, 변경행, 구출·훼손, GPU 종류와 시간이 들어 있습니다.

`COMPLETE`는 실험 실행 완료만 뜻합니다. **배포/제출 적격은 항상 false**입니다. 실제 3,000행 제한 시간 및 600행 독립성 검증을 대체하지 않습니다. L4가 아니면 L4 시간 측정으로 해석하지 않습니다.
''')
nb=dict(cells=cells,metadata=dict(colab=dict(name='ASTRA_HIGHGAIN_L4.ipynb',provenance=[]),kernelspec=dict(display_name='Python 3',name='python3'),language_info=dict(name='python'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
(R/'package/ASTRA_HIGHGAIN_L4.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
