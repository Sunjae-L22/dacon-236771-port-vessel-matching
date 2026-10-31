"""Separate CPU process. Reconstruct frozen110 fold reference, add generated candidates."""
import os
os.environ['KO_DICT']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,pickle,time
import numpy as np,pandas as pd,lightgbm as lgb
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'baseline'))
import opus_lib as B
from fuzzy_ext import FuzzyExt

def context(fold):
    cat=pd.read_csv(HERE/'data/vessel_catalog.csv');path=HERE/f'baseline/fold{fold}'
    stats=pickle.loads((path/'stats.pkl').read_bytes());B.KA._UNIVERSAL_KO_SUBWORDS=stats['universal']
    ix=B.Index(cat,None);ix.cs_map={k:ix.id2i[v] for k,v in stats['cs_map'].items()};ix.flag_p=stats['flag_p'];ix.type_p=stats['type_p'];ix.ko=B.KA.KoMatcher(cat,stats['ko_pairs']);ix.fit_alias(stats['alias_rows'])
    models=[lgb.Booster(model_file=str(p)) for p in sorted(path.glob('ranker_*.txt'))]
    return cat,ix,FuzzyExt(ix,cat),models

def run(fold,out):
    out=Path(out);cat,ix,fx,models=context(fold);excluded=set(ix.cs_map.values());names=cat.vessel_name.astype(str).tolist();start=time.time()
    assert not B.USE_ALIAS,'Cache requires the unchanged110 alias-disabled setting'
    generated=json.loads((out/'generated.json').read_text());packets=[json.loads(s) for s in (HERE/f'data/eval_fold{fold}.jsonl').read_text().splitlines()];queries=json.loads((HERE/f'data/queries_fold{fold}.json').read_text())
    cache={};checked=0;max_diff=0.
    def rank(row,cand):
        idx,F=B.query_features(ix,row,cand);s=np.mean([m.predict(F,num_threads=2) for m in models],axis=0)
        allowed=[j for j,i in enumerate(idx) if i not in excluded] or list(range(len(idx)))
        order=sorted(allowed,key=lambda j:(-float(s[j]),idx[j]))
        return [dict(index=int(idx[j]),id=str(ix.ids[idx[j]]),name=names[idx[j]],score=float(s[j])) for j in order]
    with (out/'expanded_packets.jsonl').open('w') as f:
        for q,p in zip(queries,packets):
            assert q['query_id']==p['query_id'];r=dict(p);r['original_top100_n']=len(p['candidates'])
            if p['gap']>=4:
                r.update(retrieval_pred=p['baseline'],beam_ids=[],original_full_ids=[],expanded_full_ids=[],new_candidates=0)
            else:
                assert q.get('call_sign') not in ix.cs_map
                key=tuple(str(q.get(k)) for k in ['vessel_name','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year'])
                if key not in cache:
                    cand=fx.add(B.query_candidates(ix,q),q);before=rank(q,cand);added=set();beam=[]
                    for item in generated[p['name']]:
                        for vid in item['ids']:
                            i=ix.id2i[vid]
                            assert i not in excluded
                            beam.append(vid)
                            if i not in cand:cand[i]=[999,999,999];added.add(i)
                    after=rank(q,cand) if added else before
                    cache[key]=(before,after,sorted(set(beam)),len(added))
                before,after,beam,added=cache[key]
                assert [c['id'] for c in before[:100]]==[c['id'] for c in p['candidates']],p['query_id']
                diff=max(abs(a['score']-b['score']) for a,b in zip(before,p['candidates']));max_diff=max(max_diff,diff);assert diff<1e-6
                checked+=1
                amap={c['id']:c for c in after};pool=[amap[c['id']] for c in p['candidates']];seen={c['id'] for c in pool}
                for c in after[:100]+[amap[v] for v in beam]:
                    if c['id'] not in seen:pool.append(c);seen.add(c['id'])
                r.update(candidates=pool,retrieval_pred=after[0]['id'],beam_ids=beam,original_full_ids=[c['id'] for c in before],expanded_full_ids=[c['id'] for c in after],new_candidates=added)
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            if checked and checked%100==0:print('BASELINE',fold,checked,'seconds',round(time.time()-start),flush=True)
    (out/'baseline_report.json').write_text(json.dumps(dict(seconds=time.time()-start,baseline_top100_checked=checked,baseline_score_max_diff=max_diff,models=len(models)),indent=2))

if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2])
