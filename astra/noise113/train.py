"""Astra: baseline full blocks + 3 syllable-edit/missing-tonnage copies; never shrinks baseline."""
import argparse,time,gc
from common import *

def prepare(scope,seed):
    start=time.time();fit,va,cat,ix,fx,base,cache,plan,key=context(scope)
    dest=OUT/scope;extra=dest/f'extra{seed}';extra.mkdir(parents=True,exist_ok=True)
    rows=augment(fit,seed)
    if va is not None:assert not set(fit.target_vessel_id)&set(va.target_vessel_id)
    dump(extra/'rows.json',rows)
    if not (dest/'cache_verification.json').exists():
        checked=assert_cache_samples(base,cache,plan,key,ix)
        dump(dest/'cache_verification.json',dict(block_samples=checked,first60_exact=True,baseline_plan_sha=sha(base/'plan.json')))
    for lo in range(0,len(rows),250):
        p=extra/f'{lo:06d}.npz'
        if p.exists():continue
        X=[];ys=[];groups=[]
        for r in rows[lo:lo+250]:
            idx,F=B.query_features(ix,r,fx.add(B.query_candidates(ix,r),r));t=ix.id2i[r['target_vessel_id']]
            X.append(F);ys.append(np.array([i==t for i in idx],np.int8));groups.append(len(idx))
        temp=p.with_suffix(f'.{os.getpid()}.tmp.npz');np.savez_compressed(temp,X=np.concatenate(X),y=np.concatenate(ys),groups=np.array(groups,np.int32));temp.replace(p)
        print('EXTRA',scope,seed,lo+len(groups),len(rows),round(time.time()-start),flush=True)
    dump(extra/'COMPLETE.json',dict(scope=scope,seed=seed,extra_rows=len(rows),fit_vessels=int(fit.target_vessel_id.nunique()),
        heldout_vessels=int(va.target_vessel_id.nunique()) if va is not None else 0,
        baseline_queries=plan['seeds'][str(seed)]['queries'],features=60,candidate_generator='unchanged110 FuzzyExt on new block',
        original_block_candidate_generator='original72; cached first60 preserved',
        seed_protocol='three copies, 1/2 edits .7/.3, delete/dup/swap .35/.35/.30, both tonnages missing .5, no new flag/tonnage noise'))

def train(scope,seed):
    dest=OUT/scope/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    extra=OUT/scope/f'extra{seed}';assert (extra/'COMPLETE.json').exists()
    if scope.startswith('fold'):
        base=ROOT/f'astra/incumbent97/{scope}';cache=base/'features';key='X'
    else:base=ROOT/'astra/super98/full_train';cache=base/'features84';key='X84'
    plan=json.loads((base/'plan.json').read_text())
    paths=[(p,key) for p in baseline_paths(base,cache,plan,seed)]+[(p,'X') for p in sorted(extra.glob('[0-9][0-9][0-9][0-9][0-9][0-9].npz'))]
    start=time.time();model=dest/'ranker.txt'
    if model.exists():return
    groups=[];ys=[];n=0
    for p,k in paths:
        with np.load(p) as z:groups.extend(z['groups'].tolist());ys.append(z['y']);n+=len(z['y'])
    assert len(groups)==plan['seeds'][str(seed)]['queries']+json.loads((extra/'COMPLETE.json').read_text())['extra_rows']
    mat=dest/'matrix.npy';X=np.lib.format.open_memmap(mat,mode='w+',dtype=np.float32,shape=(n,60));off=0
    for p,k in paths:
        with np.load(p) as z:count=len(z['y']);X[off:off+count]=z[k][:,:60];off+=count
    y=np.concatenate(ys);assert sum(groups)==len(y)==n
    print('TRAIN',scope,seed,len(groups),n,flush=True)
    m=lgb.train(dict(I.PARAMS,num_threads=2),lgb.Dataset(X,y,group=groups,feature_name=B.FEATURES),600)
    m.save_model(str(model));del X,y,ys;gc.collect();mat.unlink()
    dump(dest/'manifest.json',dict(scope=scope,seed=seed,model_sha256=sha(model),queries=len(groups),candidates=n,seconds=time.time()-start,trees=m.num_trees()))
    print('TRAIN_DONE',scope,seed,round(time.time()-start),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope');p.add_argument('seed',type=int);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.scope,a.seed)
    else:train(a.scope,a.seed)
