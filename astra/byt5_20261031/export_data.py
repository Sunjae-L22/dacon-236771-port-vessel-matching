"""Export fixed, fit-only inputs. Does not run neural models or touch other agents' files."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'package'
sys.path.insert(0,str(ROOT/'astra/noise113'))
from common import I,B

def write(name,rows):
    p=OUT/'data'/name
    with p.open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def pairs(df,cat,source):
    names=cat.set_index('vessel_id').vessel_name.to_dict();seen=set();out=[]
    for r in df.to_dict('records'):
        name=str(r['vessel_name']);v=str(r['target_vessel_id'])
        if not B.is_hangul(name) or (v,name) in seen:continue
        seen.add((v,name));out.append(dict(name=name,target=str(names[v]),vessel=v,source=source))
    return out

def main():
    manifest={}
    for fold in [2,4]:
        tr,cat,fit,va,ix,llm,base=I.context(fold)
        assert list(map(str,cat.vessel_id))==list(map(str,ix.ids))
        catalog_names=cat.vessel_name.astype(str).tolist();fit_targets=set(fit.target_vessel_id)
        rows=pairs(fit,cat,'native')+pairs(llm,cat,'llm')
        assert not set(x['vessel'] for x in rows)&set(va.target_vessel_id)
        write(f'train_fold{fold}.jsonl',rows)
        ev=ROOT/f'astra/noise113/fold{fold}/eval';qs=json.loads((ev/'queries.json').read_text());packets=[]
        for path in sorted(ev.glob('[0-9][0-9][0-9][0-9][0-9][0-9].npz')):
            lo=int(path.stem)
            with np.load(path) as z:
                idx_all=z['idx'];scores_all=z['base_scores'];allowed_all=z['allowed'];groups=z['groups']
                off=0
                for j,n in enumerate(groups):
                    q=qs[lo+j];idx=idx_all[off:off+n];scores=scores_all[off:off+n]
                    allowed=np.flatnonzero(allowed_all[off:off+n]).tolist() or list(range(n));off+=n
                    order=sorted(allowed,key=lambda k:(-float(scores[k]),int(idx[k])))[:100]
                    cs=[dict(index=int(idx[k]),id=str(ix.ids[idx[k]]),name=catalog_names[idx[k]],score=float(scores[k])) for k in order]
                    assert all(c['id'] not in fit_targets for c in cs)
                    packets.append(dict(query_id=q['query_id'],name=str(q['vessel_name']),variant=q['variant'],vessel=str(q['target_vessel_id']),target=str(q['target_vessel_id']),baseline=cs[0]['id'],gap=cs[0]['score']-cs[1]['score'] if len(cs)>1 else 99.,candidates=cs))
        write(f'eval_fold{fold}.jsonl',packets)
        manifest[f'fold{fold}']=dict(train_pairs=len(rows),fit_vessels=len(set(fit.target_vessel_id)),heldout_vessels=len(set(va.target_vessel_id)),overlap=0,eval_queries=len(packets),baseline_models=6 if fold==2 else 1,top100_recall=sum(any(c['id']==p['target'] for c in p['candidates']) for p in packets),hard_errors=sum(p['gap']<4 and p['baseline']!=p['target'] for p in packets))
        print(fold,manifest[f'fold{fold}'],flush=True)
    tr,cat=I.cv.load();llm=B.llm_korean_rows(tr,ROOT/'shared/experiments/pinned/build70/llm_korean_scored.csv')
    write('train_full.jsonl',pairs(tr,cat,'native')+pairs(llm,cat,'llm'))
    (OUT/'data/open').mkdir(exist_ok=True)
    for n in ['train.csv','vessel_catalog.csv']:shutil.copy2(ROOT/'data/open'/n,OUT/'data/open'/n)
    shutil.copy2(ROOT/'submissions/110_submit.zip',OUT/'baseline110.zip')
    # Freeze official local_check test input now; no reimplementation of its corruption.
    sys.path.insert(0,str(ROOT/'shared/src'))
    import local_check
    t=local_check.corrupted_test();t.to_csv(OUT/'data/local_check3000_with_truth.csv',index=False)
    for n in ['local_check.py','cv.py','opus_lib.py','ko_translit_argon.py']:
        shutil.copy2(ROOT/'shared/src'/n,OUT/n)
    manifest['files']={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file()}
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
