"""Author: Astra. Build a fully retrained Korean route only after OOF gates pass."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]

SCRIPT='''"""Astra87: re-trained Korean ranker; original 79 English route."""
from pathlib import Path
import os,sys,pickle,unicodedata,json
if os.environ.get('PYTHONHASHSEED')!='0':
    os.execve(sys.executable,[sys.executable]+sys.argv,dict(os.environ,PYTHONHASHSEED='0'))
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import lightgbm as lgb
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'model'))
import opus_lib as L
import opus_lib_65 as L65
import astra_lib as A

def run():
    catalog=pd.read_csv(HERE/'data/vessel_catalog.csv',encoding='utf-8-sig')
    test=pd.read_csv(HERE/'data/test.csv',encoding='utf-8-sig')
    with (HERE/'model/stats.pkl').open('rb') as f:stats=pickle.load(f)
    os.environ['KO_DICT']='1'
    ix=L.Index(catalog,None)
    ix.cs_map={k:ix.id2i[v] for k,v in stats['cs_map'].items() if v in ix.id2i}
    ix.flag_p,ix.type_p=stats['flag_p'],stats['type_p']
    ix.train_targets=set(ix.cs_map.values())
    ix.ko=L.KA.KoMatcher(catalog,stats['ko_pairs'])
    ix.fit_alias(stats['alias_rows'])
    with (HERE/'model/ko_stats87.pkl').open('rb') as f:new=pickle.load(f)
    os.environ['KO_DICT']='0'
    koix=A.Index(catalog,new['fit'])
    koix.ko=L.KA.KoMatcher(catalog,new['ko_pairs'])
    assert list(ix.ids)==list(koix.ids)
    ko_model=lgb.Booster(model_file=str(HERE/'model/ko_ranker87.txt'))
    assert ko_model.feature_name()==A.FEATURES
    en_models=[lgb.Booster(model_file=str(p)) for p in sorted((HERE/'model').glob('en65_*.txt'))]
    predictions=[]
    for row in test.to_dict('records'):
        known=ix.cs_map.get(row.get('call_sign'))
        if known is not None:
            predictions.append(ix.ids[known]);continue
        q=unicodedata.normalize('NFC',str(row.get('vessel_name') or ''))
        if L.is_hangul(q):
            idx,F=A.query_features(koix,row,A.query_candidates(koix,row))
            scores=ko_model.predict(F,num_threads=1)
        else:
            idx,F=L65.query_features(ix,row,L65.query_candidates(ix,row))
            scores=sum(m.predict(F,num_threads=1) for m in en_models)
        order=sorted(range(len(idx)),key=lambda j:(-float(scores[j]),idx[j]))
        win=next((idx[j] for j in order if idx[j] not in ix.train_targets),idx[order[0]])
        predictions.append(ix.ids[win])
    (HERE/'output').mkdir(exist_ok=True)
    pd.DataFrame({'sample_id':test.sample_id,'target_vessel_id':predictions}).to_csv(HERE/'output/submission.csv',index=False)
    print('Saved',len(predictions),'independent predictions')
if __name__=='__main__':run()
'''

def main():
    report=json.loads((ROOT/'astra/ranker84_oof_report.json').read_text())
    assert report['primary_confirmation_folds_1_to_4']['eligible_for_full_training'],'OOF gate rejected this candidate'
    native=json.loads((ROOT/'astra/neural87_oof_report.json').read_text())['native_ranker']
    assert native['delta']>=0,'All native Korean rows regressed; do not package'
    dst=ROOT/'submissions/87_submit.zip'
    if dst.exists():raise FileExistsError(dst)
    with zipfile.ZipFile(ROOT/'submissions/79_submit.zip') as z:items={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
    compile(SCRIPT,'script.py','exec')
    items['script.py']=SCRIPT.encode()
    items['model/astra_lib.py']=(ROOT/'shared/experiments/pinned/build84_astra/astra_lib.py').read_bytes()
    items['model/ko_ranker87.txt']=(ROOT/'astra/full84/enhanced.txt').read_bytes()
    items['model/ko_stats87.pkl']=(ROOT/'astra/full84/stats.pkl').read_bytes()
    with zipfile.ZipFile(dst,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(items.items()):z.writestr(n,b)
    receipt=dict(base='79 English route, new Korean-only ranker',ko_dict=0,features=71,neural=False,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size,local_check='PENDING',public_lb=None,validation='paired single-seed Korean-only ranker 5-fold; primary confirmation folds1-4; not a direct heldout comparison against full-data submission79 ensemble')
    (ROOT/'astra/build87_manifest.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
