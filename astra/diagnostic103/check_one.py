"""Run repository local_check with hard assertions, preserved predictions and language diffs."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def main(condition):
    entry=next(x for x in json.loads((OUT/'build_manifest.json').read_text()) if x['condition']==condition)
    pred=OUT/f'pred_{condition}.csv'
    env=dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
             LOCAL_CHECK_SAVE=str(pred))
    env.pop('KO_DICT',None)
    p=subprocess.run([sys.executable,str(ROOT/'shared/src/local_check.py'),entry['path']],
                     cwd='/tmp',env=env,capture_output=True,text=True)
    (OUT/f'check_{condition}.log').write_text(p.stdout+p.stderr)
    assert p.returncode == 0, p.stderr[-3000:]
    assert 'subset 600 rows, mismatches 0' in p.stdout, p.stdout
    t=pd.read_csv(ROOT/'data/open/train.csv').set_index('sample_id')
    b=pd.read_csv(ROOT/'shared/experiments/decomp93/pred_93.csv').set_index('sample_id')
    a=pd.read_csv(pred).set_index('sample_id')
    assert a.index.is_unique and set(a.index)==set(b.index) and len(a)==3000
    # Use exactly local_check's corrupted names, including NFC routing.
    sys.path.insert(0,str(ROOT/'shared/src'))
    from local_check import corrupted_test
    q=corrupted_test().set_index('sample_id').loc[a.index]
    ko=q.vessel_name.map(lambda s:bool(re.search('[가-힣]',unicodedata.normalize('NFC',str(s)))))
    changed=a.target_vessel_id.ne(b.loc[a.index].target_vessel_id)
    assert int(changed[~ko].sum())==0
    catalog=set(pd.read_csv(ROOT/'data/open/vessel_catalog.csv').vessel_id)
    assert set(a.target_vessel_id)<=catalog
    entry.update(local_check='PASS',rows=len(a),subset_rows=600,subset_mismatches=0,
                 seconds=float(re.search(r'time ([\d.]+)s',p.stdout)[1]),
                 korean_rows=int(ko.sum()),english_rows=int((~ko).sum()),
                 korean_changed=int(changed[ko].sum()),english_changed=int(changed[~ko].sum()),
                 diagnostic_change_explanation='Mask intentionally removes all easy Korean rows and hard rows outside X. >30 changes are expected; not a correction candidate.')
    (OUT/f'qa_{condition}.json').write_text(json.dumps(entry,indent=2,ensure_ascii=False))
    print(json.dumps(entry,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main(sys.argv[1])
