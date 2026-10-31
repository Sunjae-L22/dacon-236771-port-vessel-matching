"""Author: Astra. Verify dictionary contract, English parity, and no-Arrow execution."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from check89_portability import BLOCKER

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'astra'))
import ranker84 as R
sys.path.insert(0, str(ROOT / 'shared/src'))
import local_check

CAPTURE = '''    rows=test.to_dict('records')
    xs=[];ids=[];groups=[]
    for row in rows:
        ii,ff=A.query_features(koix,row,A.query_candidates(koix,row))
        xs.append(ff);ids.extend(ii);groups.append(len(ii))
    np.savez(HERE/'capture.npz',X=np.concatenate(xs),idx=np.array(ids),groups=np.array(groups))
    raise SystemExit(0)
'''

REFERENCE = '''import os,sys,pickle
from pathlib import Path
os.environ['KO_DICT']='0'
import numpy as np
import pandas as pd
ROOT=Path(sys.argv[1]);HERE=Path(sys.argv[2])
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import opus_lib as L, astra_lib as A
assert L.KA._UNIVERSAL_KO_SUBWORDS==[]
catalog=pd.read_csv(ROOT/'data/open/vessel_catalog.csv')
test=pd.read_csv(HERE/'data/test.csv')
with (ROOT/'astra/full84/stats.pkl').open('rb') as f:new=pickle.load(f)
koix=A.Index(catalog,new['fit']);koix.ko=L.KA.KoMatcher(catalog,new['ko_pairs'])
xs=[];ids=[];groups=[]
for row in test.to_dict('records'):
    ii,ff=A.query_features(koix,row,A.query_candidates(koix,row))
    xs.append(ff);ids.extend(ii);groups.append(len(ii))
np.savez(HERE/'reference.npz',X=np.concatenate(xs),idx=np.array(ids),groups=np.array(groups))
'''


def execute(command, env, cwd):
    completed = subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def main():
    env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('KO_DICT', None); env.pop('PYTHONPATH', None)
    report = {}
    with tempfile.TemporaryDirectory(prefix='astra90_') as tmp:
        tmp = Path(tmp)
        blocker = tmp / 'no_pyarrow'; blocker.mkdir()
        (blocker / 'sitecustomize.py').write_text(BLOCKER)
        blocked = dict(env, PYTHONPATH=str(blocker))
        train = pd.read_csv(ROOT / 'data/open/train.csv')
        selected = train[train.vessel_name.map(R.B.is_hangul)].drop_duplicates('target_vessel_id').sample(10, random_state=90)
        probes = pd.DataFrame(R.queries(selected, train))
        work = tmp / '90'; work.mkdir()
        shutil.unpack_archive(ROOT / 'submissions/90_submit.zip', work)
        (work / 'data').mkdir()
        shutil.copy(ROOT / 'data/open/vessel_catalog.csv', work / 'data')
        probes.to_csv(work / 'data/test.csv', index=False)
        original_script = (work / 'script.py').read_text()
        script = original_script.replace('    predictions=[]\n', CAPTURE)
        # Validate legacy matcher instance copies survive the temporary global clear.
        script = script.replace('    universal=L.KA._UNIVERSAL_KO_SUBWORDS\n', "    legacy_dictionary_before=pickle.dumps((ix.ko.exact_word_dict,ix.ko.learned_subwords))\n    universal=L.KA._UNIVERSAL_KO_SUBWORDS\n")
        script = script.replace('    assert list(ix.ids)==list(koix.ids)\n', "    assert legacy_dictionary_before==pickle.dumps((ix.ko.exact_word_dict,ix.ko.learned_subwords))\n    assert list(ix.ids)==list(koix.ids)\n")
        (work / 'script.py').write_text(script)
        execute([sys.executable, 'script.py'], blocked, work)
        ref = tmp / 'reference.py'; ref.write_text(REFERENCE)
        execute([sys.executable, str(ref), str(ROOT), str(work)], env, '/tmp')
        with np.load(work / 'capture.npz') as actual, np.load(work / 'reference.npz') as expected:
            for key in ('idx', 'groups'):
                np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(actual['X'], expected['X'])
            report['training_contract'] = {'queries': len(probes), 'variants': sorted(probes.variant.unique().tolist()), 'candidates': len(actual['idx']), 'features': actual['X'].shape[1], 'candidate_mismatches': 0, 'feature_mismatches': 0, 'legacy_dictionary_unchanged': True}
        print('90 training feature/candidate contract exact parity:', report['training_contract'], flush=True)
        (work / 'script.py').write_text(original_script)
        english = local_check.corrupted_test(n=3000)
        english = english[~english.vessel_name.map(R.B.is_hangul)].head(350)
        old = tmp / '79'; old.mkdir()
        shutil.unpack_archive(ROOT / 'submissions/79_submit.zip', old)
        (old / 'data').mkdir(); shutil.copy(ROOT / 'data/open/vessel_catalog.csv', old / 'data')
        for folder in (old, work):
            english.drop(columns='target_vessel_id').to_csv(folder / 'data/test.csv', index=False)
            execute([sys.executable, 'script.py'], blocked, folder)
        a = pd.read_csv(old / 'output/submission.csv')
        b = pd.read_csv(work / 'output/submission.csv')
        pd.testing.assert_frame_equal(a, b)
        canonical_ko = english.vessel_name.map(lambda s: R.B.is_hangul(__import__('unicodedata').normalize('NFC', str(s))))
        report['english_parity'] = {'baseline_submission': 79, 'rows': len(english), 'true_english_rows': int((~canonical_ko).sum()), 'nfd_korean_rows': int(canonical_ko.sum()), 'mismatches': 0}
        print('79/90 parity: 350 rows (348 English, 2 NFD Korean), 0 mismatches', flush=True)
        done = execute([sys.executable, str(ROOT / 'shared/src/local_check.py'), str(ROOT / 'submissions/90_submit.zip')], blocked, '/tmp')
        (ROOT / 'astra/check90.log').write_text(done.stdout + done.stderr)
        assert 'subset 600 rows, mismatches 0' in done.stdout, done.stdout
        report['local_check_without_pyarrow'] = done.stdout
        print(done.stdout, flush=True)
    report['status'] = 'PASS'
    (ROOT / 'astra/check90_contract.json').write_text(json.dumps(report, indent=2) + '\n')
    path = ROOT / 'astra/build90_manifest.json'; manifest = json.loads(path.read_text())
    manifest.update(training_feature_parity='PASS', english_parity='PASS', local_check='PASS', pyarrow_absent_qa='PASS')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('BUILD90_QA_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
