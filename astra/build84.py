"""Author: Astra. Build additive Korean retrieval on immutable submission 79.

No model re-training, no English route changes, no test-derived rules.
Must be evaluated with frozen_ranker84.py and local_check.py before submission.
"""
from pathlib import Path
import hashlib,json,zipfile,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'shared/experiments/pinned/build84_astra'

def main():
    dst=ROOT/'submissions/84_submit.zip'
    if dst.exists(): raise FileExistsError(dst)
    summary=ROOT/'astra/frozen84_ranked/predictions.csv'
    completion=json.loads((summary.parent/'manifest.json').read_text())
    d=pd.read_csv(summary)
    assert d.query_sample_id.nunique()==completion['queries']==3213
    # Development holdout gate: positive tune AND eval mean, no orig regression.
    for role in ['tune','eval']:
        q=d[(d.role==role)&d.exclude_train]
        means=q.groupby(['variant','method']).correct.mean().unstack()
        assert (means.expanded-means.baseline).mean()>0, (role,means)
        assert means.loc['orig','expanded']>=means.loc['orig','baseline'], (role,means.loc['orig'])
    with zipfile.ZipFile(ROOT/'submissions/79_submit.zip') as z:
        items={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
    # Original inference feature code must match pilot aside from augmentation and determinism.
    script=items['script.py'].decode()
    script=script.replace('if os.environ.get("PYTHONHASHSEED") is None:', 'if os.environ.get("PYTHONHASHSEED") != "0":')
    script=script.replace('import lightgbm as lgb','os.environ["OMP_NUM_THREADS"]="1"\nos.environ["OPENBLAS_NUM_THREADS"]="1"\nimport lightgbm as lgb')
    script=script.replace('import opus_lib_65 as L65','import opus_lib_65 as L65\nimport astra_lib as A\nimport json')
    script=script.replace('ix = L.Index(catalog, None)','ix = A.Index(catalog, None)\n    ix.meta = json.loads((HERE / "model" / "astra_meta.json").read_text())')
    old='idx, F = L.query_features(ix, row, L.query_candidates(ix, row))'
    assert script.count(old)==1
    script=script.replace(old,'idx, F = L.query_features(ix, row, A.query_candidates(ix, row))')
    script=script.replace('m.predict(F)', 'm.predict(F, num_threads=1)').replace('m.predict(F65)', 'm.predict(F65, num_threads=1)')
    items['script.py']=script.encode()
    items['model/astra_lib.py']=(CODE/'astra_lib.py').read_bytes()
    train=pd.read_csv(ROOT/'data/open/train.csv');cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv').set_index('vessel_id')
    meta={}
    for field,source in [('flag_name','flag_name'),('vessel_type','vessel_type'),('unit','vessel_type')]:
        counts={}
        for r in train.drop_duplicates(['target_vessel_id',source]).to_dict('records'):
            key=str(r[source]);value=str(cat.loc[r['target_vessel_id'],field]);c=counts.setdefault(key,{})
            c[value]=c.get(value,0)+1
        meta[field]=counts
    items['model/astra_meta.json']=json.dumps(meta,ensure_ascii=False).encode()
    with zipfile.ZipFile(dst,'x',zipfile.ZIP_DEFLATED) as z:
        for n,data in sorted(items.items()): z.writestr(n,data)
    receipt=dict(base='79_submit.zip',change='Korean candidate expansion plus retrieval-rank feature correction; frozen rankers',
                 neural_model_included=False,local_check='PENDING',public_lb=None,
                 sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size,
                 source_sha256={n:hashlib.sha256(v).hexdigest() for n,v in items.items()})
    (ROOT/'astra/build84_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(dst,receipt['bytes'],receipt['sha256'])

if __name__=='__main__':main()
