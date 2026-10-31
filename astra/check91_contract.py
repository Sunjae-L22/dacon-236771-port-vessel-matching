"""Author: Astra. Whole-script NFC preservation/NFD invariance and no-Arrow QA."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata

import pandas as pd
from check89_portability import BLOCKER

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'astra'))
import ranker84 as R
sys.path.insert(0, str(ROOT / 'shared/src'))
import local_check


def execute(command, env, cwd):
    result = subprocess.run(command, env=env, cwd=cwd, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def main():
    env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('KO_DICT', None); env.pop('PYTHONPATH', None)
    report = {}
    with tempfile.TemporaryDirectory(prefix='astra91_') as tmp:
        tmp = Path(tmp)
        blocker = tmp / 'no_pyarrow'; blocker.mkdir()
        (blocker / 'sitecustomize.py').write_text(BLOCKER)
        env['PYTHONPATH'] = str(blocker)
        train = pd.read_csv(ROOT / 'data/open/train.csv')
        selected = train[train.vessel_name.map(R.B.is_hangul)].drop_duplicates('target_vessel_id').sample(50, random_state=91)
        korean = pd.DataFrame([r for r in R.queries(selected, train) if r['variant'] in ('orig', 'trunc_noise')])
        korean['sample_id'] = korean['variant'] + '_' + korean['sample_id']
        english = local_check.corrupted_test(n=3000)
        english = english[~english.vessel_name.map(lambda s: R.B.is_hangul(unicodedata.normalize('NFC', str(s))))].head(100).copy()
        english['sample_id'] = 'english_' + english['sample_id']
        probes = pd.concat([korean, english], ignore_index=True)
        probes['call_sign'] = 'UNSEEN'
        probes['vessel_name'] = probes.vessel_name.map(lambda s: unicodedata.normalize('NFC', str(s)))
        probes = probes.drop(columns=['target_vessel_id', 'variant'])
        assert len(probes) == probes.sample_id.nunique() == 200
        folders = {}
        for num in (90, 91):
            folder = tmp / str(num); folder.mkdir(); folders[num] = folder
            shutil.unpack_archive(ROOT / f'submissions/{num}_submit.zip', folder)
            (folder / 'data').mkdir(); shutil.copy(ROOT / 'data/open/vessel_catalog.csv', folder / 'data')
            probes.to_csv(folder / 'data/test.csv', index=False)
            execute([sys.executable, 'script.py'], env, folder)
        baseline = pd.read_csv(folders[90] / 'output/submission.csv').set_index('sample_id')
        fixed_nfc = pd.read_csv(folders[91] / 'output/submission.csv').set_index('sample_id')
        pd.testing.assert_frame_equal(baseline, fixed_nfc)
        report['nfc_prediction_parity'] = {'baseline_submission': 90, 'rows': 200, 'korean_rows': 100, 'english_rows': 100, 'mismatches': 0}
        print('90/91 NFC prediction parity: 200 mixed rows, mismatches 0', flush=True)
        decomposed = probes[probes.sample_id.isin(korean.sample_id)].copy()
        assert len(decomposed) == 100
        decomposed['vessel_name'] = decomposed.vessel_name.map(lambda s: unicodedata.normalize('NFD', s))
        decomposed.to_csv(folders[91] / 'data/test.csv', index=False)
        execute([sys.executable, 'script.py'], env, folders[91])
        fixed_nfd = pd.read_csv(folders[91] / 'output/submission.csv').set_index('sample_id')
        pd.testing.assert_frame_equal(fixed_nfc.loc[fixed_nfd.index], fixed_nfd)
        report['nfd_prediction_invariance'] = {'rows': 100, 'orig_rows': 50, 'trunc_noise_rows': 50, 'call_sign': 'UNSEEN', 'whole_script_with_final_train_exclusion': True, 'mismatches': 0}
        print('91 NFC/NFD whole-script invariance: 100 Korean rows, mismatches 0', flush=True)
        done = execute([sys.executable, str(ROOT / 'shared/src/local_check.py'), str(ROOT / 'submissions/91_submit.zip')], env, '/tmp')
        (ROOT / 'astra/check91.log').write_text(done.stdout + done.stderr)
        assert 'subset 600 rows, mismatches 0' in done.stdout, done.stdout
        report['local_check_without_pyarrow'] = done.stdout
        print(done.stdout, flush=True)
    report['status'] = 'PASS'
    (ROOT / 'astra/check91_contract.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest_path = ROOT / 'astra/build91_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(nfc_prediction_parity='PASS', nfd_prediction_invariance='PASS', local_check='PASS', pyarrow_absent_qa='PASS')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('BUILD91_QA_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
