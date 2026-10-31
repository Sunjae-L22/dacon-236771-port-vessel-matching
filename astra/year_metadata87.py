"""Author: Astra. Train/catalog-only vessel-disjoint evaluation of arrival-year priors."""
from pathlib import Path
import sys,json
from collections import Counter
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import cv

def main():
    train,cat=cv.load();folds=cv.folds(train);cat=cat.set_index('vessel_id');out=ROOT/'astra/year_metadata87';out.mkdir(exist_ok=True)
    results=[]
    for field in ['flag_name','vessel_type']:
        labels=sorted(cat[field].fillna('').unique());label2i={s:i for i,s in enumerate(labels)}
        prior=cat[field].fillna('').value_counts().reindex(labels).to_numpy(float);prior/=prior.sum()
        d=train[['target_vessel_id',field,'arrival_year']].copy();d['fold']=folds;d['query']=d[field].fillna('').astype(str);d['truth']=d.target_vessel_id.map(cat[field]).fillna('');d['ci']=d.truth.map(label2i)
        for f in range(5):
            fit=d[d.fold!=f].drop_duplicates(['target_vessel_id','query','arrival_year']);val=d[d.fold==f].drop_duplicates(['target_vessel_id','query','arrival_year'])
            byquery={k:g for k,g in fit.groupby('query')}
            for query,rows in val.groupby('query'):
                sub=byquery.get(query,fit.iloc[:0]);basecounts=np.bincount(sub.drop_duplicates(['target_vessel_id','query']).ci.to_numpy(int),minlength=len(labels)).astype(float)
                base=(basecounts+prior)/(basecounts.sum()+1)
                for year,g in rows.groupby('arrival_year'):
                    true=g.ci.to_numpy(int)
                    for sigma in [2.,4.]:
                        temp=sub.copy();temp['w']=np.exp(-.5*((temp.arrival_year-year)/sigma)**2)
                        votes=temp.groupby(['target_vessel_id','ci']).w.max().reset_index()
                        cnt=np.bincount(votes.ci.to_numpy(int),weights=votes.w,minlength=len(labels))
                        for alpha in [5.,20.,50.]:
                            prob=(cnt+alpha*base)/(cnt.sum()+alpha)
                            for target,t in zip(g.target_vessel_id,true):
                                results.append(dict(field=field,fold=f,target=target,sigma=sigma,alpha=alpha,base_nll=-np.log(max(base[t],1e-12)),year_nll=-np.log(max(prob[t],1e-12)),base_correct=int(t==base.argmax()),year_correct=int(t==prob.argmax())))
    d=pd.DataFrame(results);d.to_csv(out/'predictions.csv',index=False)
    m=d.groupby(['field','sigma','alpha','target'])[['base_nll','year_nll','base_correct','year_correct']].mean().groupby(['field','sigma','alpha']).mean()
    m.to_csv(out/'summary.csv');print(m.to_string())
    (out/'manifest.json').write_text(json.dumps(dict(protocol='5-fold target-vessel disjoint; scores macro-averaged per vessel; exploratory grid; no test data',meaning='metadata label prediction only, not vessel matching accuracy'),indent=2))

if __name__=='__main__':main()
