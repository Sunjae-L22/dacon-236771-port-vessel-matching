"""Astra: local_check plus English/easy invariance against saved93 and hard gate."""
from pathlib import Path
import json,os,sys,subprocess,re,unicodedata
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((OUT/'build_manifest.json').read_text())
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',LOCAL_CHECK_SAVE=str(OUT/'pred_g2.csv'))
    env.pop('KO_DICT',None)
    p=subprocess.run([sys.executable,str(ROOT/'shared/src/local_check.py'),manifest['path']],cwd='/tmp',env=env,capture_output=True,text=True)
    (OUT/'local_check.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stderr[-3000:]
    assert 'subset 600 rows, mismatches 0' in p.stdout,p.stdout
    b=pd.read_csv(ROOT/'shared/experiments/decomp93/pred_93.csv').set_index('sample_id').target_vessel_id
    a=pd.read_csv(OUT/'pred_g2.csv').set_index('sample_id').target_vessel_id
    gate=pd.read_csv(ROOT/'astra/diagnostic103/pred_hard_all.csv').set_index('sample_id').target_vessel_id
    assert a.index.is_unique and set(a.index)==set(b.index)
    a=a.reindex(b.index);gate=gate.reindex(b.index);changed=a.ne(b)
    sys.path.insert(0,str(ROOT/'shared/src'))
    from local_check import corrupted_test
    t=corrupted_test().set_index('sample_id').loc[b.index]
    ko=t.vessel_name.map(lambda s:bool(re.search('[가-힣]',unicodedata.normalize('NFC',str(s)))))
    hard=ko&gate.ne('VESSEL_26253');easy=ko&~hard
    assert not changed[~ko].any() and not changed[easy].any()
    result=dict(local_check='PASS',rows=3000,subset_rows=600,subset_mismatches=0,
        seconds=float(re.search(r'time ([\d.]+)s',p.stdout)[1]),
        korean_rows=int(ko.sum()),korean_changed=int(changed[ko].sum()),english_changed=int(changed[~ko].sum()),
        easy_rows=int(easy.sum()),easy_changed=int(changed[easy].sum()),hard_rows=int(hard.sum()),hard_changed=int(changed[hard].sum()))
    (OUT/'qa.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
