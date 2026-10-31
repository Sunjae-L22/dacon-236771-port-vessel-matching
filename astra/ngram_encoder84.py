"""Author: Astra. CPU trainable character/jamo contrastive bi-encoder pilot.
Public competition train pairs only; stable hashing; tune-only model/fusion selection.
"""
import os,sys,unicodedata,re,zlib,random,json,time
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import opus_lib as B
torch.set_num_threads(2)
N=32768; DIM=96

@lru_cache(maxsize=30000)
def features(name):
    s=' '.join(unicodedata.normalize('NFKC',str(name)).lower().split())
    roman=B.romanize(B.ko_clean(s)) if B.is_hangul(s) else B.acronym_expand(s)
    projections=[('c',unicodedata.normalize('NFD',s)),('r',roman),('p',B.phon_key(roman,ko=B.is_hangul(s))),('v',B.vowel_key(roman))]
    ids=[]
    for tag,text in projections:
        words=text.split(); strings=words+[''.join(words)]
        for word in strings:
            word='^'+word+'$'
            for n in (1,2,3,4):
                for i in range(len(word)-n+1): ids.append(zlib.crc32((tag+word[i:i+n]).encode())%N)
    return tuple(ids or [0])

def batch(names):
    arrays=[features(s) for s in names]; lengths=[len(a) for a in arrays]
    return torch.tensor([x for a in arrays for x in a],dtype=torch.long),torch.tensor(np.cumsum([0]+lengths[:-1]),dtype=torch.long)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.EmbeddingBag(N,DIM,mode='mean');self.proj=nn.Sequential(nn.Linear(DIM,DIM),nn.GELU(),nn.Linear(DIM,DIM))
        nn.init.normal_(self.emb.weight,std=.08)
    def forward(self,names):
        v=self.emb(*batch(names));return F.normalize(v+self.proj(v),dim=1)

def vectors(model,names):
    with torch.inference_mode(): return torch.cat([model(names[i:i+256]) for i in range(0,len(names),256)]).numpy()

def score_candidates(model,data):
    qnames=list(dict.fromkeys(data.vessel_name.tolist()));cnames=list(dict.fromkeys(data.candidate_vessel_name.tolist()))
    qvec=vectors(model,qnames);cvec=vectors(model,cnames);qi={s:i for i,s in enumerate(qnames)};ci={s:i for i,s in enumerate(cnames)}
    qidx=data.vessel_name.map(qi).to_numpy();cidx=data.candidate_vessel_name.map(ci).to_numpy()
    return (qvec[qidx]*cvec[cidx]).sum(1)

def evaluation(d,weight):
    d=d.assign(score=d.base_z+weight*d.nn_z)
    p=d.sort_values(['query_sample_id','score','rank','candidate_vessel_id'],ascending=[True,False,True,True]).groupby('query_sample_id').head(1)
    macro=p.groupby(['variant','target_vessel_id']).is_target.mean().groupby('variant').mean().mean()
    orig=p[p.variant=='orig'].is_target.mean()
    return float(macro),float(orig),p

def full_pairs():
    t=pd.read_csv(ROOT/'data/open/train.csv');cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv').set_index('vessel_id')
    real=t[t.vessel_name.map(B.is_hangul)]
    real=pd.DataFrame(dict(vessel_id=real.target_vessel_id,english=real.target_vessel_id.map(cat.vessel_name),korean=real.vessel_name,source='real'))
    g=pd.read_csv(ROOT/'shared/experiments/pinned/build70/llm_korean_scored.csv')
    base=t.drop_duplicates('target_vessel_id').set_index('target_vessel_id')
    g=g[(g.score>=.7)&g.vessel_id.map(base.gross_tonnage).notna()]
    synthetic=pd.DataFrame(dict(vessel_id=g.vessel_id,english=g.vessel_id.map(cat.vessel_name),korean=g.ko,source='llm_v1'))
    pairs=pd.concat([real,synthetic],ignore_index=True).drop_duplicates(['vessel_id','english','korean','source'])
    for source in pairs.source.unique():
        mask=pairs.source==source;counts=pairs[mask].groupby('vessel_id').size()
        pairs.loc[mask,'sample_weight']=.5/len(counts)/pairs.loc[mask,'vessel_id'].map(counts)
    return pairs

def main(full=False):
    random.seed(84);np.random.seed(84);torch.manual_seed(84)
    out=ROOT/'astra'/('ngram85_full' if full else 'ngram84');out.mkdir(exist_ok=True)
    pairs=full_pairs() if full else pd.read_csv(ROOT/'terra/byt5_pilot/colab_v1/delivery/port_byt5_pilot_v1/prepared/train_pairs.csv')
    d=pd.read_csv(ROOT/'shared/experiments/llm_rerank/candidates_top20.csv')
    if not full: assert not set(pairs.vessel_id)&set(d.target_vessel_id)
    # Exclusion mirrors the actual submission rule, and is based exclusively on fit vessels.
    d=d[~d.cand_is_train_role_target].copy()
    z=lambda a:(a-a.mean())/max(float(a.std(ddof=0)),1e-6)
    d['base_z']=d.groupby('query_sample_id').ranker_score.transform(z)
    tune=d[d.role=='tune'].copy()
    model=Encoder();optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    weights=pairs.sample_weight.to_numpy(float,copy=True);weights/=weights.sum()
    rng=np.random.default_rng(84);best=(-1.,0.);start=time.time();trials=[]
    for step in range(1,1201):
        b=pairs.iloc[rng.choice(len(pairs),size=128,p=weights)]
        q=[]
        for name in b.korean:
            name=str(name)
            if rng.random()<.35: name=B._typo(name,int(rng.integers(1,3)),rng)
            if rng.random()<.15 and len(name.split())>1: name=' '.join(name.split()[::-1])
            q.append(name)
        optimizer.zero_grad(set_to_none=True)
        qv=model(q);cv=model(b.english.tolist());logits=qv@cv.T/.07
        eq=torch.tensor(b.vessel_id.to_numpy()[:,None]==b.vessel_id.to_numpy()[None,:])
        # Equal catalog spellings are indistinguishable to a name-only model.
        en=np.array([B.norm(s) for s in b.english]);eq|=torch.tensor(en[:,None]==en[None,:])
        pos=logits.masked_fill(~eq,float('-inf'))
        loss=((logits.logsumexp(1)-pos.logsumexp(1)).mean()+(logits.logsumexp(0)-pos.logsumexp(0)).mean())/2
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0:print('step',step,'loss',round(float(loss.detach()),4),'seconds',round(time.time()-start),flush=True)
        if step%300==0 and not full:
            model.eval();tune['nn']=score_candidates(model,tune);tune['nn_z']=tune.groupby('query_sample_id').nn.transform(z)
            base,orig,_=evaluation(tune,0)
            for w in [0,.05,.1,.2,.35,.5,1.]:
                sc,osc,_=evaluation(tune,w);trials.append(dict(step=step,weight=w,tune=sc,orig=osc))
                if osc+1e-12>=orig and (sc,-w)>best:
                    best=(sc,-w);torch.save(model.state_dict(),out/'best.pt');(out/'selection.json').write_text(json.dumps(dict(step=step,weight=w,tune=sc,orig=osc,baseline=base)))
            print('TUNE',step,best,flush=True);model.train()
    if full:
        torch.save(model.state_dict(),out/'best.pt');pairs.to_csv(out/'training_pairs.csv',index=False)
        (out/'manifest.json').write_text(json.dumps(dict(steps=1200,seed=84,weight=.1,selection='ngram84 tune only',pairs=len(pairs),seconds=time.time()-start),indent=2));return
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True));model.eval()
    d['nn']=score_candidates(model,d);d['nn_z']=d.groupby('query_sample_id').nn.transform(z)
    sel=json.loads((out/'selection.json').read_text());res=[]
    for role in ['tune','eval']:
        for method,w in [('baseline',0),('fused',sel['weight'])]:
            sc,orig,p=evaluation(d[d.role==role],w);res.append(dict(role=role,method=method,ship_macro=sc,original=orig))
    d.to_csv(out/'scored_candidates.csv',index=False);pd.DataFrame(trials).to_csv(out/'tune_trials.csv',index=False)
    report=dict(selection=sel,metrics=res,seconds=time.time()-start,protocol='three-way pilot, not 5-fold OOF',pretrained_weights=False)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main('--full' in sys.argv)
