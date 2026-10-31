"""Create the bounded Colab run after package CPU QA, without embedded data."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    payload = ROOT / 'neural97_package.zip'
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    code = '''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, threading, time, traceback, zipfile
from google.colab import drive, runtime

TASK = Path('/content/drive/MyDrive/Colab Notebooks/Astra97_항만_20261029')
EXPECTED_SHA256 = '__SHA__'
FIXED_WEIGHT = 0.25  # Frozen before any outer-fold neural result.
mounted = False
out = None
started = time.time()
status = {'status': 'STARTED', 'experiment': 'neural97', 'fixed_weight': FIXED_WEIGHT}
try:
    drive.mount('/content/drive')
    mounted = True
    assert TASK.is_dir(), str(TASK)
    stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    out = TASK / ('run_' + stamp)
    out.mkdir(exist_ok=False)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip()
    status['gpu'] = gpu
    print('GPU:', gpu, flush=True)
    assert 'L4' in gpu, 'This bounded experiment requires the selected L4 runtime.'
    archive = TASK / 'neural97_package.zip'
    raw = archive.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256, 'Package SHA256 mismatch'
    work = Path('/content/astra97')
    work.mkdir(exist_ok=False)
    local_zip = work / 'package.zip'
    local_zip.write_bytes(raw)
    with zipfile.ZipFile(local_zip) as z:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(work)
    env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    (out / 'launch.json').write_text(json.dumps(status, indent=2))
    with (out / 'train.log').open('w') as log:
        cmd = [sys.executable, '-u', 'train.py', '--data', 'data', '--config', 'config.json', '--out', str(out), '--device', 'cuda']
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        watchdog = threading.Timer(2400, proc.kill)
        watchdog.daemon = True
        watchdog.start()
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line); log.flush()
            if time.time() - started > 2400:
                proc.kill()
                raise TimeoutError('40 minute run budget exceeded')
        assert proc.wait() == 0, 'Training failed; see saved train.log'
        watchdog.cancel()
    receipt = json.loads((out / 'complete.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['steps'] == 2000
    assert (out / 'best.pt').stat().st_size > 100000
    # The baseline export is produced independently with fold2 ships excluded.
    candidates = TASK / 'matched79_outer2.jsonl'
    if candidates.exists():
        shutil.copyfile(candidates, work / candidates.name)
        result = subprocess.run([sys.executable, '-u', 'evaluate.py', '--checkpoint', str(out / 'best.pt'), '--candidates', candidates.name, '--data', 'data', '--out', str(out / 'matched79_outer2.json'), '--device', 'cuda', '--split', 'outer2', '--weight', str(FIXED_WEIGHT), '--top-k', '100'], cwd=work, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
        (out / 'evaluation.log').write_text(result.stdout)
        print(result.stdout, flush=True)
        assert result.returncode == 0, 'Evaluation failed; checkpoint is saved'
        status['matched_baseline_evaluation'] = 'COMPLETE'
    else:
        status['matched_baseline_evaluation'] = 'PENDING_CANDIDATE_EXPORT'
    status.update(status='COMPLETE', seconds=time.time()-started, best_sha256=hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest())
except BaseException as error:
    status.update(status='FAILED', error=repr(error), traceback=traceback.format_exc(), seconds=time.time()-started)
    print(status['traceback'], flush=True)
finally:
    if out is not None:
        (out / 'launch_complete.json').write_text(json.dumps(status, indent=2))
        print('SAVED:', out, json.dumps(status), flush=True)
    try:
        if mounted:
            drive.flush_and_unmount(timeout_ms=120000)
            print('Drive writes flushed; releasing this runtime.', flush=True)
    finally:
        runtime.unassign()
'''.replace('__SHA__', digest)
    ast.parse(code)
    note = '''# Astra97 문자·자모 순서 모델 — 고정된 1회 실험

허용된 train/catalog와 train-only LLM 증강만 사용합니다. 지도학습 fold 0/1/3, 체크포인트 선택 fold 4, 최종 비교 fold 2를 분리합니다. 학습 2,000단계 이후 결과·가중치를 이 노트북 폴더에 저장하고 런타임을 자동 삭제합니다. 실제 제출이나 Public 점수 개선을 의미하지 않습니다.

L4를 선택한 뒤 이 셀을 실행합니다. 이전 ByT5 실험 파일은 읽지 않습니다. 이 실험의 결과는 새 run_날짜 폴더에 보존합니다.
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'accelerator': 'GPU', 'colab': {'name': 'Astra97_Character_Jamo_L4.ipynb', 'provenance': [], 'gpuType': 'L4'}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}, 'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': note.splitlines(True)}, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(True)}]}
    path = ROOT / 'Astra97_Character_Jamo_L4.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2))
    print(json.dumps({'notebook': str(path), 'bytes': path.stat().st_size, 'package_sha256': digest}))


if __name__ == '__main__':
    main()
