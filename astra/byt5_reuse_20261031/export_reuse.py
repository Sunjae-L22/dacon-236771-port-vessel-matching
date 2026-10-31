from pathlib import Path
import sys,json,pickle,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'package'
sys.path.insert(0,str(ROOT/'astra/noise113'))
from common import I,B,SEEDS,SRC
import numpy as np,pandas as pd

def main():
    (OUT/'data').mkdir(parents=True,exist_ok=True);(OUT/'baseline').mkdir(exist_ok=True)
    cat=None;manifest={}
    for fold in [2,4]:
        tr,cat,fit,va,ix,llm,base=I.context(fold)
        assert not set(fit.target_vessel_id)&set(va.target_vessel_id)
        assert Path(B.KA.__file__).read_bytes()==(SRC/'ko_translit_argon.py').read_bytes()
        pairs=pd.concat([fit[fit.vessel_name.map(B.is_hangul)],llm])[['vessel_name','target_vessel_id']].to_dict('records')
        stats=dict(cs_map={k:str(ix.ids[v]) for k,v in ix.cs_map.items()},flag_p=ix.flag_p,type_p=ix.type_p,ko_pairs=pairs,alias_rows=fit[['sample_id','vessel_name','target_vessel_id']].to_dict('records'),universal=list(B.KA._UNIVERSAL_KO_SUBWORDS))
        d=OUT/'baseline'/f'fold{fold}';d.mkdir(exist_ok=True)
        (d/'stats.pkl').write_bytes(pickle.dumps(stats,protocol=4))
        for seed in (SEEDS if fold==2 else [0]):shutil.copy2(base/f'seed{seed}/ranker.txt',d/f'ranker_{seed}.txt')
        src=ROOT/'astra/byt5_20261031'
        for old,new in [(src/f'package/data/eval_fold{fold}.jsonl',f'eval_fold{fold}.jsonl'),(src/f'package/data/train_fold{fold}.jsonl',f'train_fold{fold}.jsonl'),(ROOT/f'astra/noise113/fold{fold}/eval/queries.json',f'queries_fold{fold}.json'),(src/f'gpu_return_20261031/fold{fold}/eval/paired_rows.jsonl',f'old_scores_fold{fold}.jsonl')]:shutil.copy2(old,OUT/'data'/new)
        (OUT/'data'/f'excluded_fold{fold}.json').write_text(json.dumps(sorted(set(stats['cs_map'].values()))))
        manifest[f'fold{fold}']=dict(models=len(list(d.glob('ranker_*.txt'))),fit_vessels=len(set(fit.target_vessel_id)),held_vessels=len(set(va.target_vessel_id)),fit_validation_overlap=0,excluded=len(set(stats['cs_map'].values())))
        print(fold,manifest[f'fold{fold}'],flush=True)
    for name in ['opus_lib.py','ko_translit_argon.py','fuzzy_ext.py','ksh_hangul.py']:shutil.copy2(SRC/name,OUT/'baseline'/name)
    shutil.copy2(ROOT/'data/open/vessel_catalog.csv',OUT/'data/vessel_catalog.csv')
    shutil.copy2(ROOT/'astra/byt5_20261031/package/neural.py',OUT/'neural.py')
    previous=ROOT/'astra/byt5_20261031/gpu_return_20261031/FINAL_STATUS.json'
    manifest['expected_previous_status_sha256']=hashlib.sha256(previous.read_bytes()).hexdigest()
    manifest['files']={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
