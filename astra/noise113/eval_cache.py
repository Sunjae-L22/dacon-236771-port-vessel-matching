"""Astra: paired110 wide-candidate clean/three-edit holdout cache, fixed before evaluation."""
import argparse,time
from common import *
VARIANTS=['clean_keep','clean_missing','delete_keep','delete_missing','duplicate_keep','duplicate_missing','swap_keep','swap_missing','mixed2_keep','mixed2_missing']
def main(fold):
    scope=f'fold{fold}';fit,va,cat,ix,fx,base,cache,plan,key=context(scope)
    out=OUT/scope/'eval';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261031+fold);rows=[]
    for r in va[va.vessel_name.map(B.is_hangul)].to_dict('records'):
        names={'clean':r['vessel_name']}
        for op in ['delete','duplicate','swap']:names[op]=edit(r['vessel_name'],rng,op)
        names['mixed2']=edit(edit(r['vessel_name'],rng),rng)
        for v in VARIANTS:
            op,tone=v.split('_');q={k:r.get(k) for k in I.MODEL_FIELDS+['target_vessel_id']}
            q['call_sign']='UNSEEN';q['vessel_name']=names[op];q['variant']=v
            q['query_id']=r['sample_id']+'::'+v
            if tone=='missing':q['gross_tonnage']=q['international_gross_tonnage']=np.nan
            rows.append(q)
    dump(out/'queries.json',rows)
    models=[lgb.Booster(model_file=str(base/f'seed{s}/ranker.txt')) for s in (SEEDS if fold==2 else [0])]
    fit_targets=set(ix.cs_map.values());start=time.time()
    for lo in range(0,len(rows),100):
        p=out/f'{lo:06d}.npz'
        if p.exists():continue
        X=[];ids=[];groups=[];targets=[];allow=[]
        for r in rows[lo:lo+100]:
            idx,F=B.query_features(ix,r,fx.add(B.query_candidates(ix,r),r))
            X.append(F);ids.extend(idx);groups.append(len(idx));targets.append(ix.id2i[r['target_vessel_id']]);allow.extend(i not in fit_targets for i in idx)
        X=np.concatenate(X);scores=sum(m.predict(X,num_threads=2) for m in models)/len(models)
        temp=p.with_suffix('.tmp.npz');np.savez_compressed(temp,X=X,idx=np.array(ids,np.int32),groups=np.array(groups,np.int32),
            target=np.array(targets,np.int32),allowed=np.array(allow,bool),base_scores=scores);temp.replace(p)
        print('EVAL_CACHE',fold,lo+len(groups),len(rows),round(time.time()-start),flush=True)
    dump(out/'COMPLETE.json',dict(fold=fold,n=len(rows),native_rows=len(rows)//len(VARIANTS),variants=VARIANTS,
        baseline_models=len(models),baseline='110 wide candidates, fold-safe72 recipe',
        limitations=['Development folds reused','Synthetic edit/tonnage-missing conditions are hypotheses, not test labels']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fold',type=int);a=p.parse_args();main(a.fold)
