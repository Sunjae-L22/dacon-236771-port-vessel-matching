from pathlib import Path
import json,hashlib,zipfile,ast
ROOT=Path(__file__).resolve().parent;PKG=ROOT/'package'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    # Input manifest contains only immutable data/baseline snapshots and unchanged neural helper.
    p=PKG/'input_manifest.json';m=json.loads(p.read_text());m['files']={str(f.relative_to(PKG)):sha(f) for f in sorted(PKG.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and (f.is_relative_to(PKG/'data') or f.is_relative_to(PKG/'baseline') or f.name=='neural.py')};p.write_text(json.dumps(m,ensure_ascii=False,indent=2))
    cells=[]
    def add(kind,s):
        c=dict(id=f'cell-{len(cells)}',cell_type=kind,metadata={},source=s.splitlines(True))
        if kind=='code':c.update(execution_count=None,outputs=[]);ast.parse(s)
        cells.append(c)
    add('markdown','''# Astra ByT5 기존 가중치 재사용 — G4

**새 학습0스텝, 대회 제출0회.** 이전 fold2/fold4 모델을 Drive에서 찾아 카탈로그 전체 검색을 검증합니다. 이전 학습 폴더를 그대로 보관하세요.

위부터 모두 실행하고, 업로드 창에 `Astra_ByT5_Reuse_G4_20261031.zip`을 넣으세요. 끝나면 `REUSE_RESULTS_RETURN_TO_ASTRA.zip`만 전달하면 됩니다. 새 제출ZIP은 만들지 않습니다.
''')
    add('code', '''from google.colab import drive, files
from pathlib import Path
import sys, subprocess, json, zipfile, datetime
drive.mount('/content/drive')
subprocess.run(['nvidia-smi'], check=True)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.6', 'lightgbm==4.6.0', 'rapidfuzz', 'pandas', 'numpy<3', 'scipy'])
''')
    add('code', '''uploaded = files.upload()
names = [n for n in uploaded if n.endswith('.zip')]
assert len(names) == 1, '재사용 실행 묶음 ZIP 하나를 선택하세요.'
stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
work = Path('/content') / ('astra_byt5_reuse_' + stamp)
work.mkdir(exist_ok=False)
with zipfile.ZipFile(names[0]) as archive:
    for n in archive.namelist():
        assert not Path(n).is_absolute() and '..' not in Path(n).parts
    archive.extractall(work)
package = work / 'astra_reuse'
previous_root = Path('/content/drive/MyDrive/astra_byt5_20261031')
run_dir = Path('/content/drive/MyDrive/astra_byt5_reuse_20261031') / ('run_' + stamp)
assert previous_root.is_dir(), '이전 ByT5 실행 결과가 있는 Drive 계정인지 확인하세요.'
del uploaded
print('기존 모델 탐색:', previous_root)
print('새 검증 결과:', run_dir)
''')
    add('code', '''proc = subprocess.Popen([sys.executable, '-u', str(package/'run_reuse.py'), str(previous_root), str(run_dir)],
                        cwd='/tmp', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end='', flush=True)
print('process returncode:', proc.wait())
''')
    add('code', '''status_path = run_dir / 'FINAL_STATUS.json'
if status_path.exists():
    result = json.loads(status_path.read_text())
    print('상태:', result['status'])
    for fold, report in result.get('reports', {}).items():
        print(fold, '기존 누락 정답 추가검색:', report['newly_recalled'], '/', report['original_missing_hard'])
        for name, metric in report['methods'].items():
            print(name, 'clean 훼손=', metric['clean']['harm'], 'noise 순증=', metric['noise']['net'])
return_zip = run_dir / 'REUSE_RESULTS_RETURN_TO_ASTRA.zip'
assert return_zip.exists(), f'로그와 {run_dir}를 확인하세요.'
files.download(str(return_zip))
''')
    nb=dict(cells=cells,metadata=dict(colab=dict(name='Astra_ByT5_Reuse_G4_20261031.ipynb'),kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
    n=ROOT/'Astra_ByT5_Reuse_G4_20261031.ipynb';n.write_text(json.dumps(nb,ensure_ascii=False,indent=2))
    for f in PKG.glob('*.py'):ast.parse(f.read_text())
    fs=[f for f in sorted(PKG.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='package_manifest.json' and 'reused_weights' not in f.parts]
    manifest=dict(files={str(f.relative_to(PKG)):sha(f) for f in fs});(PKG/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    zpath=ROOT/'Astra_ByT5_Reuse_G4_20261031.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for f in fs+[PKG/'package_manifest.json']:z.write(f,'astra_reuse/'+str(f.relative_to(PKG)))
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        for name,digest in manifest['files'].items():assert hashlib.sha256(z.read('astra_reuse/'+name)).hexdigest()==digest
    result=dict(status='REUSE_ONLY_EXECUTION_KIT',zip=str(zpath),sha256=sha(zpath),bytes=zpath.stat().st_size,notebook=str(n),files=len(fs)+1,new_training_steps=0,public_submissions=0)
    (ROOT/'BUNDLE_REPORT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
