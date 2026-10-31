"""One extra copy instead of three. Existing baseline blocks and source caches are read-only."""
import argparse,time,gc,fcntl
from common import *
def main(scope,seed):
    extra=OUT/scope/f'extra{seed}';meta=json.loads((extra/'COMPLETE.json').read_text())
    assert meta['extra_rows']%3==0;extra_count=meta['extra_rows']//3
    dest=OUT/'light'/scope/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (dest/'manifest.json').exists():
        saved=json.loads((dest/'manifest.json').read_text());assert sha(dest/'ranker.txt')==saved['model_sha256'];return
    assert not (dest/'ranker.txt').exists()
    if scope.startswith('fold'):base=ROOT/f'astra/incumbent97/{scope}';cache=base/'features';key='X'
    else:base=ROOT/'astra/super98/full_train';cache=base/'features84';key='X84'
    plan=json.loads((base/'plan.json').read_text());paths=[(p,key,None) for p in baseline_paths(base,cache,plan,seed)]
    remaining=extra_count
    for p in sorted(extra.glob('[0-9][0-9][0-9][0-9][0-9][0-9].npz')):
        with np.load(p) as z:take=min(remaining,len(z['groups']))
        if take:paths.append((p,'X',take));remaining-=take
        if remaining==0:break
    assert remaining==0
    groups=[];ys=[];selected=[];total=0;start=time.time()
    for p,k,take in paths:
        with np.load(p) as z:
            g=z['groups'] if take is None else z['groups'][:take];n=int(g.sum())
            groups.extend(g.tolist());ys.append(z['y'][:n]);selected.append((p,k,n));total+=n
    assert len(groups)==plan['seeds'][str(seed)]['queries']+extra_count
    matrix=dest/'matrix.npy';X=np.lib.format.open_memmap(matrix,mode='w+',dtype=np.float32,shape=(total,60));off=0
    for p,k,n in selected:
        with np.load(p) as z:X[off:off+n]=z[k][:n,:60];off+=n
    y=np.concatenate(ys);assert sum(groups)==len(y)==total
    print('LIGHT_TRAIN',scope,seed,len(groups),total,flush=True)
    m=lgb.train(dict(I.PARAMS,num_threads=2),lgb.Dataset(X,y,group=groups,feature_name=B.FEATURES),600)
    model=dest/'ranker.txt';m.save_model(str(model));del X,y,ys;gc.collect();matrix.unlink()
    dump(dest/'manifest.json',dict(scope=scope,seed=seed,variant='one_extra_copy',extra_queries=extra_count,queries=len(groups),candidates=total,trees=m.num_trees(),features=m.num_feature(),model_sha256=sha(model),seconds=time.time()-start,selection='first augmentation copy in pre-generated order; no label-based selection'))
    print('LIGHT_DONE',scope,seed,round(time.time()-start),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope');p.add_argument('seed',type=int);a=p.parse_args();main(a.scope,a.seed)
