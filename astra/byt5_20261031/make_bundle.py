from pathlib import Path
import json,hashlib,zipfile,ast,shutil
ROOT=Path(__file__).resolve().parent
PKG=ROOT/'package'

def main():
    cells=[]
    def md(s):cells.append(dict(id=f'cell-{len(cells)}',cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):cells.append(dict(id=f'cell-{len(cells)}',cell_type='code',execution_count=None,metadata={},outputs=[],source=s.splitlines(True)))
    md('''# Astra ByT5 — G4 검증·학습 묶음 (2026-10-31)

**대회 제출 파일이 아닙니다. 제출 횟수 0회로 검증합니다.** Opus의72B/112/113/v18 작업을 사용하지 않습니다.

비어 있는 GPU 런타임에서 위부터 모두 실행하세요. 업로드 창에는 같은 이름의 ZIP을 넣습니다. 두 fold 검증에 통과해야 전체 학습·ZIP·검사를 진행합니다. 현재 설정은 fold별1500스텝이며 실제 속도/남은 시간은50스텝마다 표시합니다. 고정 마지막 모델만 평가합니다.

완료 후 `RESULTS_RETURN_TO_ASTRA.zip`을 전달하세요. 생성된 UNREVIEWED 후보는 검토·L4 시간 확인 전 제출하지 마세요. 111의 Public 실패(한글−3) 때문에 합성검증 점수를 Public 예상값으로 사용하지 않습니다.
''')
    code('''from google.colab import drive, files
from pathlib import Path
import sys, subprocess, json, zipfile, datetime
drive.mount('/content/drive')
subprocess.run(['nvidia-smi'], check=True)
''')
    code('''# torch는 현재 G4 런타임 것을 유지합니다. 별도 프로세스가 설치된 transformers를 읽습니다.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'transformers==4.57.6', 'lightgbm==4.6.0',
                       'rapidfuzz', 'pandas', 'numpy<3', 'scipy'])
''')
    code('''uploaded = files.upload()  # Astra_ByT5_G4_20261031.zip 하나 선택
zip_names = [n for n in uploaded if n.endswith('.zip')]
assert len(zip_names) == 1, '실행 묶음 ZIP 하나만 업로드하세요.'
stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
work = Path('/content') / ('astra_byt5_work_' + stamp)
work.mkdir(exist_ok=False)
with zipfile.ZipFile(zip_names[0]) as archive:
    for name in archive.namelist():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    archive.extractall(work)
package = work / 'astra_byt5'
run_dir = Path('/content/drive/MyDrive/astra_byt5_20261031') / ('run_' + stamp)
assert (package / 'run_all.py').exists()
print('결과 저장:', run_dir)
del uploaded
''')
    code('''# 한 번만 실행. 중단 후 다시 실행하면 새 run 폴더를 만들어야 합니다.
proc = subprocess.Popen([sys.executable, '-u', str(package/'run_all.py'), str(run_dir)],
                        cwd=package, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
for line in proc.stdout:
    print(line, end='', flush=True)
print('process returncode:', proc.wait())
''')
    code('''status_file = run_dir / 'FINAL_STATUS.json'
if status_file.exists():
    status = json.loads(status_file.read_text())
    print('최종 상태:', status['status'])
    for fold, result in status.get('fold_reports', {}).items():
        print(fold, 'clean harm=', result['clean']['harm'],
              'noise rescue/harm=', result['noise']['rescue'], result['noise']['harm'],
              'error reduction=', result['noise_error_reduction'])
result_zip = run_dir / 'RESULTS_RETURN_TO_ASTRA.zip'
assert result_zip.exists(), f'실행 로그와 {run_dir} 내용을 확인하세요.'
print('이 ZIP만 Astra에 전달하세요. 모델은 Drive에 남습니다:', result_zip)
files.download(str(result_zip))
''')
    nb=dict(cells=cells,metadata=dict(colab=dict(name='Astra_ByT5_G4_20261031.ipynb'),kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
    notebook=ROOT/'Astra_ByT5_G4_20261031.ipynb';notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=2))
    for c in cells:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    for p in PKG.glob('*.py'):ast.parse(p.read_text())
    files=[p for p in sorted(PKG.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='package_manifest.json']
    manifest=dict(files={str(p.relative_to(PKG)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (PKG/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    bundle=ROOT/'Astra_ByT5_G4_20261031.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[PKG/'package_manifest.json']:z.write(p,'astra_byt5/'+str(p.relative_to(PKG)))
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        for name,digest in manifest['files'].items():assert hashlib.sha256(z.read('astra_byt5/'+name)).hexdigest()==digest
    report=dict(bundle=str(bundle),sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),bytes=bundle.stat().st_size,notebook=str(notebook),files=len(files)+1,status='GPU_EXECUTION_KIT_NOT_SUBMISSION')
    (ROOT/'BUNDLE_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
