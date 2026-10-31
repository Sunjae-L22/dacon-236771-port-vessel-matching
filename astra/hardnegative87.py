"""Author: Astra. Train-only hard-negative contrastive pilot; tune-only selection.

Catalog names may be negatives. No holdout labels, test rows, or hand mappings
are consumed by prepare/train. Evaluation uses the pre-existing vessel split.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,random,argparse,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rapidfuzz import process,fuzz
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import ngram_encoder84 as N
import opus_lib as B
OUT=ROOT/'astra/hardnegative87'
PAIR=ROOT/'terra/byt5_pilot/colab_v1/delivery/port_byt5_pilot_v1/prepared/train_pairs.csv'

def prepare():
    OUT.mkdir(exist_ok=True)
    pairs=pd.read_csv(PAIR)
    catalog=pd.read_csv(ROOT/'data/open/vessel_catalog.csv')
    ids=catalog.vessel_id.tolist();names=catalog.vessel_name.fillna('').astype(str).tolist()
    id2i={v:i for i,v in enumerate(ids)};norm=[B.norm(s) for s in names]
    gt=catalog.gross_tonnage.map(B.positive).to_numpy()
    targets=pairs.vessel_id.drop_duplicates().tolist();hard={};start=time.time()
    for j,target in enumerate(targets):
        t=id2i[target];key=norm[t]
        near=process.extract(key,norm,scorer=fuzz.ratio,limit=70)
        lexical=[i for _,_,i in near if norm[i]!=key][:24]
        # These are catalog-side attributes of a training target, never test statistics.
        if np.isfinite(gt[t]):
            gaps=np.abs(np.log(np.maximum(gt,1)/max(gt[t],1)))
            gaps[~np.isfinite(gaps)]=np.inf
            metadata=[int(i) for i in np.argsort(gaps,kind='stable')[:100] if norm[i]!=key][:24]
        else:metadata=[]
        hard[target]=dict(lexical=lexical,metadata=metadata)
        if (j+1)%500==0:print('MINED',j+1,len(targets),round(time.time()-start),flush=True)
    (OUT/'hard_negatives.json').write_text(json.dumps(hard))
    (OUT/'data_manifest.json').write_text(json.dumps(dict(pair_sha256=hashlib.sha256(PAIR.read_bytes()).hexdigest(),pairs=len(pairs),targets=len(targets),catalog=len(catalog),seconds=time.time()-start,source='training pair English labels and allowed catalog only'),indent=2))

def main():
    random.seed(87);np.random.seed(87);torch.manual_seed(87)
    pairs=pd.read_csv(PAIR);catalog=pd.read_csv(ROOT/'data/open/vessel_catalog.csv')
    names=catalog.vessel_name.fillna('').astype(str).tolist();norm=[B.norm(s) for s in names]
    id2i={v:i for i,v in enumerate(catalog.vessel_id)}
    hard=json.loads((OUT/'hard_negatives.json').read_text())
    d=pd.read_csv(ROOT/'shared/experiments/llm_rerank/candidates_top20.csv')
    assert not set(pairs.vessel_id)&set(d.target_vessel_id)
    d=d[~d.cand_is_train_role_target].copy()
    z=lambda a:(a-a.mean())/max(float(a.std(ddof=0)),1e-6)
    d['base_z']=d.groupby('query_sample_id').ranker_score.transform(z)
    tune=d[d.role=='tune'].copy();rng=np.random.default_rng(87)
    model=N.Encoder();optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    weights=pairs.sample_weight.to_numpy(float,copy=True);weights/=weights.sum()
    best=(-1.,0);trials=[];start=time.time()
    for step in range(1,1501):
        b=pairs.iloc[rng.choice(len(pairs),size=64,p=weights)]
        queries=[];positives=[];negative_names=[]
        for row in b.to_dict('records'):
            q=str(row['korean']);t=id2i[row['vessel_id']];positive=names[t]
            if rng.random()<.4:q=B._typo(q,int(rng.integers(1,3)),rng)
            if rng.random()<.15 and len(q.split())>1:q=' '.join(q.split()[::-1])
            # Mild truncation; severe deletion makes some names unidentifiable without metadata.
            if rng.random()<.1 and len(q)>5:q=q[:max(3,int(.8*len(q)))]
            options=hard[row['vessel_id']]
            chosen=list(rng.choice(options['lexical'],size=3))
            pool=options['metadata'] or options['lexical'];chosen+=list(rng.choice(pool,size=2))
            while len(chosen)<8:
                i=int(rng.integers(len(names)))
                if norm[i]!=norm[t]:chosen.append(i)
            queries.append(q);positives.append(positive);negative_names.extend(names[i] for i in chosen)
        optimizer.zero_grad(set_to_none=True)
        qv=model(queries);pv=model(positives);nv=model(negative_names).reshape(64,8,-1)
        # Exact same catalog spellings within the batch are multi-positive, never false negatives.
        eq=torch.tensor(np.array([B.norm(s) for s in positives])[:,None]==np.array([B.norm(s) for s in positives])[None,:])
        logits=qv@pv.T/.07;neg=torch.einsum('bd,bnd->bn',qv,nv)/.07
        pos=logits.masked_fill(~eq,float('-inf'))
        loss=(torch.cat([logits,neg],1).logsumexp(1)-pos.logsumexp(1)).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0:print('TRAIN',step,float(loss.detach()),round(time.time()-start),flush=True)
        if step%300==0:
            model.eval();tune['nn']=N.score_candidates(model,tune);tune['nn_z']=tune.groupby('query_sample_id').nn.transform(z)
            base,orig,_=N.evaluation(tune,0)
            for w in [0,.05,.1,.2,.35,.5,1.]:
                sc,osc,_=N.evaluation(tune,w);trials.append(dict(step=step,weight=w,tune=sc,orig=osc))
                if osc+1e-12>=orig and (sc,-w)>best:
                    best=(sc,-w);torch.save(model.state_dict(),OUT/'best.pt')
                    (OUT/'selection.json').write_text(json.dumps(dict(step=step,weight=w,tune=sc,orig=osc,baseline=base)))
            print('TUNE',step,best,flush=True);model.train()
    model.load_state_dict(torch.load(OUT/'best.pt',weights_only=True));model.eval()
    d['nn']=N.score_candidates(model,d);d['nn_z']=d.groupby('query_sample_id').nn.transform(z)
    sel=json.loads((OUT/'selection.json').read_text());metrics=[]
    for role in ['tune','eval']:
        for method,w in [('baseline',0),('hardnegative',sel['weight'])]:
            sc,orig,p=N.evaluation(d[d.role==role],w)
            metrics.append(dict(role=role,method=method,ship_macro=sc,original=orig))
    d.to_csv(OUT/'scored_candidates.csv',index=False);pd.DataFrame(trials).to_csv(OUT/'tune_trials.csv',index=False)
    report=dict(selection=sel,metrics=metrics,seconds=time.time()-start,protocol='reused three-way vessel holdout; exploratory, not full OOF')
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare() if a.prepare else main()
