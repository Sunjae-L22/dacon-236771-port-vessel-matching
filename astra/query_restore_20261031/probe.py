from pathlib import Path
import sys,json,time,argparse
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astra/noise113'))
from common import context,B,SEEDS,lgb,dump
from restorer import Restorer

def main(fold):
    started=time.time();fit,va,cat,ix,fx,base,cache,plan,key=context(f'fold{fold}')
    restore=Restorer(ix.ko);dest=OUT/f'fold{fold}';dest.mkdir(exist_ok=True)
    evaldir=ROOT/f'astra/noise113/fold{fold}/eval';queries=json.loads((evaldir/'queries.json').read_text());rows=[]
    models=[lgb.Booster(model_file=str(base/f'seed{s}/ranker.txt')) for s in (SEEDS if fold==2 else [0])]
    excluded=set(ix.cs_map.values());restoration_cache={};view_cache={}
    for path in sorted(evaldir.glob('[0-9][0-9][0-9][0-9][0-9][0-9].npz')):
        lo=int(path.stem)
        with np.load(path) as z:
            off=0
            for j,n in enumerate(z['groups']):
                q=queries[lo+j];ids=z['idx'][off:off+n];score=z['base_scores'][off:off+n];allow=np.flatnonzero(z['allowed'][off:off+n]).tolist() or list(range(n));order=sorted(allow,key=lambda k:(-score[k],int(ids[k])));old=int(ids[order[0]]);gap=float(score[order[0]]-score[order[1]]) if len(order)>1 else 99.;target=int(z['target'][j]);off+=n
                name=q['vessel_name'];info=restoration_cache.setdefault(name,restore.restore(name));views=[];chosen=old;eligible=False
                if gap<4 and info['views']:
                    for name2 in info['views']:
                        changed=dict(q,vessel_name=name2)
                        key2=tuple(str(changed.get(k)) for k in ['vessel_name','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year'])
                        if key2 not in view_cache:
                            inds,F=B.query_features(ix,changed,fx.add(B.query_candidates(ix,changed),changed))
                            predictions=np.asarray([m.predict(F,num_threads=2) for m in models]);mean=predictions.mean(axis=0);allowed=[k for k,x in enumerate(inds) if x not in excluded] or list(range(len(inds)));ranked=sorted(allowed,key=lambda k:(-mean[k],inds[k]));winner=int(inds[ranked[0]]);g=float(mean[ranked[0]]-mean[ranked[1]]) if len(ranked)>1 else 99.
                            votes=[int(inds[min(allowed,key=lambda k:(-s[k],inds[k]))]) for s in predictions]
                            view_cache[key2]=dict(name=name2,pred=winner,gap=g,unanimous=all(v==winner for v in votes),candidates=inds)
                        views.append(view_cache[key2])
                    eligible=bool(views and len({v['pred'] for v in views})==1 and all(v['gap']>=4 and v['unanimous'] for v in views))
                    if eligible:chosen=views[0]['pred']
                rows.append(dict(query_id=q['query_id'],sample_id=q['sample_id'],vessel=q['target_vessel_id'],variant=q['variant'],name=name,baseline=old,pred=chosen,target=target,base_ok=old==target,new_ok=chosen==target,changed=old!=chosen,hard=gap<4,eligible=eligible,gap=gap,reason=info['reason'],edits=info['edits'],views=len(views),any_view_correct=any(v['pred']==target for v in views),restored_names=[v['name'] for v in views]))
        pd.DataFrame(rows).to_csv(dest/'rows.csv',index=False);print('RESTORE',fold,len(rows),len(queries),round(time.time()-started),flush=True)
    d=pd.DataFrame(rows)
    def metric(z):return dict(n=len(z),vessels=z.vessel.nunique(),base_correct=int(z.base_ok.sum()),new_correct=int(z.new_ok.sum()),rescue=int((~z.base_ok&z.new_ok).sum()),harm=int((z.base_ok&~z.new_ok).sum()),changed=int(z.changed.sum()),outside_scope=int((z.changed&~z.eligible).sum()),any_view_oracle_rescue=int((~z.base_ok&z.any_view_correct).sum()),restored_queries=int((z.views>0).sum()))
    clean=d[d.variant.str.startswith('clean')];noise=d[~d.variant.str.startswith('clean')];report=dict(fold=fold,baseline_models=len(models),all=metric(d),clean=metric(clean),noise=metric(noise),by_variant={k:metric(z) for k,z in d.groupby('variant')},dictionary_words=len(restore.exact),reverse_keys=len(restore.rev),seconds=time.time()-started)
    dump(dest/'report.json',report);dump(dest/'view_cache.json',list(view_cache.values()));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fold',type=int);main(p.parse_args().fold)
