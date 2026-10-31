"""Author: Astra. Reproduce 88 without pyarrow, validate 89 and prediction parity."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'shared/src'))
import pandas as pd
import local_check

BLOCKER = '''import sys
class NoPyarrow:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ModuleNotFoundError("No module named 'pyarrow' (deliberately unavailable in portability QA)", name="pyarrow")
sys.meta_path.insert(0, NoPyarrow())
'''


def run_script(work, env):
    started = time.time()
    result = subprocess.run([sys.executable, 'script.py'], cwd=work, env=env, text=True, capture_output=True)
    return result, time.time() - started


def main():
    env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('KO_DICT', None)
    env.pop('PYTHONPATH', None)
    report = {}
    with tempfile.TemporaryDirectory(prefix='astra89_') as tmp:
        tmp = Path(tmp)
        blocker = tmp / 'no_pyarrow'; blocker.mkdir()
        (blocker / 'sitecustomize.py').write_text(BLOCKER)
        blocked = dict(env, PYTHONPATH=str(blocker))
        probe = subprocess.run([sys.executable, '-c', 'import pyarrow'], cwd='/tmp', env=blocked, text=True, capture_output=True)
        assert probe.returncode and "No module named 'pyarrow'" in probe.stderr
        report['pyarrow_import_blocker_verified'] = True
        examples = local_check.corrupted_test(n=3000)
        ko = examples.vessel_name.map(local_check.L.is_hangul)
        examples = pd.concat([examples[ko].head(250), examples[~ko].head(350)]).sample(frac=1, random_state=89)
        report['parity_rows'] = len(examples)
        report['parity_korean_rows'] = int(examples.vessel_name.map(local_check.L.is_hangul).sum())
        workspaces = {}
        for num in (88, 89):
            work = tmp / str(num); work.mkdir(); workspaces[num] = work
            shutil.unpack_archive(ROOT / f'submissions/{num}_submit.zip', work)
            (work / 'data').mkdir()
            shutil.copy(ROOT / 'data/open/vessel_catalog.csv', work / 'data')
            examples.drop(columns='target_vessel_id').to_csv(work / 'data/test.csv', index=False)
        failing, _ = run_script(workspaces[88], blocked)
        assert failing.returncode and "No module named 'pyarrow'" in failing.stderr, failing.stderr
        report['88_missing_pyarrow_reproduced'] = True
        report['88_failure_tail'] = failing.stderr[-1800:]
        print('88 missing pyarrow reproduced', flush=True)
        a, seconds_a = run_script(workspaces[88], env)
        assert a.returncode == 0, a.stderr
        b, seconds_b = run_script(workspaces[89], blocked)
        assert b.returncode == 0, b.stderr
        old = pd.read_csv(workspaces[88] / 'output/submission.csv').set_index('sample_id')
        new = pd.read_csv(workspaces[89] / 'output/submission.csv').set_index('sample_id')
        pd.testing.assert_frame_equal(old, new)
        report['prediction_mismatches'] = 0
        report['parity_seconds'] = {'88_with_pyarrow': seconds_a, '89_without_pyarrow': seconds_b}
        print(f'88/89 parity {len(new)} rows: 0 mismatches', flush=True)
        # Invoke the exact shared checker, with pyarrow unavailable in checker and children.
        command = [sys.executable, str(ROOT / 'shared/src/local_check.py'), str(ROOT / 'submissions/89_submit.zip')]
        result = subprocess.run(command, cwd='/tmp', env=blocked, text=True, capture_output=True)
        (ROOT / 'astra/check89.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr
        assert 'subset 600 rows, mismatches 0' in result.stdout, result.stdout
        report['local_check_without_pyarrow'] = result.stdout
        print(result.stdout, flush=True)
    report['status'] = 'PASS'
    (ROOT / 'astra/check89_portability.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest_path = ROOT / 'astra/build89_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(local_check='PASS', pyarrow_absent_qa='PASS', parity_88_rows=report['parity_rows'], parity_88_mismatches=0)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('BUILD89_QA_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
