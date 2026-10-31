"""Astra: measure clean harm counts (not just net), row scope and fixed C rule controls."""
import argparse,time
from common import *
from decision import choose

def metrics(df):
    rescue=int((~df.base_ok & df.new_ok).sum());harm=int((df.base_ok & ~df.new_ok).sum())
    return dict(n=len(df),base_correct=int(df.base_ok.sum()),new_correct=int(df.new_ok.sum()),rescue=rescue,harm=harm,net=rescue-harm,
                changed=int(df.changed.sum()),outside_scope=int((df.changed&~df.eligible).sum()),
                unique_vessels=int(df.vessel.nunique()),rescued_vessels=int(df.loc[~df.base_ok & df.new_ok,'vessel'].nunique()),
                harmed_vessels=int(df.loc[df.base_ok & ~df.new_ok,'vessel'].nunique()))
def main(fold,rule,variant=None):
    scope=f'fold{fold}';evaldir=OUT/scope/'eval'
    result_root=OUT/variant if variant else OUT
    (result_root/scope).mkdir(parents=True,exist_ok=True)
    assert (evaldir/'COMPLETE.json').exists()
    rows=json.loads((evaldir/'queries.json').read_text());res=[]
    models=[]
    if rule.startswith('noise'):
        models=[lgb.Booster(model_file=str(result_root/scope/f'seed{s}/ranker.txt')) for s in (SEEDS if fold==2 else [0])]
    started=time.time()
    for path in sorted(evaldir.glob('*.npz')):
        qlo=int(path.stem)
        with np.load(path) as z:
            X,idx,g,target,allowed,base=(z[k] for k in ['X','idx','groups','target','allowed','base_scores'])
            alt=None
            if models:
                selected=np.zeros(len(X),bool);offset=0
                for n in g:
                    valid=np.flatnonzero(allowed[offset:offset+n]).tolist() or list(range(n))
                    _,gate=choose(idx[offset:offset+n],base[offset:offset+n],valid,X[offset:offset+n],B.FEATURES,rule,base[offset:offset+n])
                    if gate['eligible']:selected[offset:offset+n]=True
                    offset+=n
                alt=np.zeros(len(X))
                if selected.any():alt[selected]=sum(m.predict(X[selected],num_threads=2) for m in models)/len(models)
            off=0
            for j,n in enumerate(g):
                ids=idx[off:off+n];s=base[off:off+n];valid=np.flatnonzero(allowed[off:off+n]).tolist() or list(range(n))
                new,info=choose(ids,s,valid,X[off:off+n],B.FEATURES,rule,alt[off:off+n] if alt is not None else None)
                r=rows[qlo+j];t=target[j];leader=info['baseline_index']
                order=sorted(valid,key=lambda k:(-float(s[k]),int(ids[k])))
                res.append(dict(query_id=r['query_id'],vessel=r['target_vessel_id'],variant=r['variant'],
                    base_ok=leader==t,new_ok=ids[new]==t,changed=ids[new]!=leader,tonnage_missing=info['tonnage_missing'],eligible=info['eligible'],hard=info['hard'],gap=info['gap'],
                    target_rank=next((k+1 for k,i in enumerate(order) if ids[i]==t),None),in_candidates=bool(t in ids)))
                off+=n
        print('EVAL_PROGRESS',scope,rule,len(res),len(rows),round(time.time()-started),flush=True)
    df=pd.DataFrame(res);df.to_csv(result_root/scope/f'{rule}_rows.csv',index=False)
    by={v:metrics(d) for v,d in df.groupby('variant')};clean=df[df.variant.str.startswith('clean')];noise=df[~df.variant.str.startswith('clean')]
    report=dict(scope=scope,rule=rule,model_variant=variant or 'three_extra_copies',all=metrics(df),clean=metrics(clean),noise=metrics(noise),by_variant=by,
        clean_harm_zero=metrics(clean)['harm']==0,eligible_scope_verified=metrics(df)['outside_scope']==0,
        models=len(models),candidate_recall=float(df.in_candidates.mean()),
        limits=['Observed clean harm0 is not a guarantee on hidden Public','Repeatedly used development fold; not a fresh final holdout',
                'Public evidence is v15-v18 retrieval trend and110 gain; retraining effect is still unverified'])
    report['eligible_for_next_stage']=bool(report['clean_harm_zero'] and report['eligible_scope_verified'] and report['noise']['net']>0)
    dump(result_root/scope/f'{rule}_report.json',report);print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fold',type=int);p.add_argument('rule',choices=['noise','noise_with_tonnage','name_top3','second_name']);p.add_argument('--variant',choices=['light']);a=p.parse_args();main(a.fold,a.rule,a.variant)
