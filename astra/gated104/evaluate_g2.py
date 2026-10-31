"""Astra: six-versus-six fixed50:50 gate, no validation threshold/weight search."""
from pathlib import Path
import json,re,unicodedata
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SEEDS=[0,5000,1000,2000,6000,7000]

def summarize(df):
    rescue=int((~df.baseline_correct&df.g2_correct).sum());damage=int((df.baseline_correct&~df.g2_correct).sum())
    return dict(n=len(df),hard=int(df.hard.sum()),changed=int(df.changed.sum()),easy_changed=int((df.changed&~df.hard).sum()),
        baseline_correct=int(df.baseline_correct.sum()),g2_correct=int(df.g2_correct.sum()),rescue=rescue,damage=damage,net=rescue-damage)

def main():
    base=ROOT/'astra/incumbent97/fold2';report={}
    for kind in ('benchmark','old9'):
        z=np.load(next(base.glob(f'eval_{kind}_*.npz')));rows=json.loads((base/f'eval_{kind}_queries.json').read_text())
        s=np.load(base/f'{kind}_6models_all_scores.npy')
        alt=sum(np.load(OUT/f'seed{seed}/{kind}_scores.npy') for seed in SEEDS)/6
        assert len(s)==len(alt)==len(z['idx'])
        mixed=.5*s+.5*alt;ids,allowed,targets=(z[k] for k in ('idx','allowed','target'))
        lo=0;results=[]
        for j,n in enumerate(z['groups']):
            pool=np.arange(lo,lo+n);valid=pool[allowed[pool]]
            if not len(valid):valid=pool
            ranked=sorted(valid,key=lambda k:(-float(s[k]),int(ids[k])))
            leader=ranked[0];gap=float(s[leader]-s[ranked[1]]) if len(ranked)>1 else 99.
            row=rows[j]
            routed=bool(re.search('[가-힣]',unicodedata.normalize('NFC',str(row['vessel_name']))))
            winner=min(valid,key=lambda k:(-float(mixed[k]),int(ids[k]))) if gap<4 and routed else leader
            altwinner=min(valid,key=lambda k:(-float(alt[k]),int(ids[k]))) if routed else leader
            if row.get('source_language')=='ko':
                results.append(dict(query_id=row.get('query_id'),vessel=row['target_vessel_id'],variant=row['variant'],
                    routed_ko=routed,hard=gap<4 and routed,gap=gap,baseline_correct=bool(ids[leader]==targets[j]),g2_correct=bool(ids[winner]==targets[j]),
                    changed=bool(ids[leader]!=ids[winner]),baseline_idx=int(ids[leader]),g2_idx=int(ids[winner]),
                    alt101_correct=bool(ids[altwinner]==targets[j]),alt101_changed=bool(ids[altwinner]!=ids[leader]),
                    target_rank=next((k+1 for k,i in enumerate(ranked) if ids[i]==targets[j]),None)))
            lo+=n
        df=pd.DataFrame(results);assert not (df.changed&~df.hard).any()
        df.to_csv(OUT/f'g2_{kind}_rows.csv',index=False)
        report[kind+'_all_source_ko']=summarize(df)
        df=df[df.routed_ko].copy()
        report[kind]=summarize(df)
        # The already existing101 is an external fixed comparator, not a searched blend weight.
        control=df.drop(columns=['g2_correct','changed']).rename(columns={'alt101_correct':'g2_correct','alt101_changed':'changed'})
        report[kind+'_existing101_recipe_control']=summarize(control)
        variants=pd.DataFrame([dict(variant=v,**summarize(g)) for v,g in df.groupby('variant')])
        variants.to_csv(OUT/f'g2_{kind}_variants.csv',index=False)
        if kind=='benchmark':
            clean=df[df.variant.eq('native_all_rows')];report['native']=summarize(clean)
            vessel=df.assign(delta=df.g2_correct.astype(int)-df.baseline_correct.astype(int)).groupby('vessel').delta.mean().to_numpy()
            rng=np.random.default_rng(104);boot=np.mean(rng.choice(vessel,size=(4000,len(vessel))),axis=1)
            report['vessel_mean_delta']=float(vessel.mean());report['vessel_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
    report['weight']=.5;report['margin_gate']=4.;report['seeds']=SEEDS
    report['limitations']=['One repeatedly used development fold, not a fresh final holdout','Synthetic gain is not evidence of Public gain','Full-data101 versus fold2 retrain differ in fit/dictionary support and fit-only LLM rescoring']
    report['local_qa']=json.loads((OUT/'qa.json').read_text())
    report['recommendation']='reviewable_experimental_candidate' if report['benchmark']['net']>0 and report['native']['net']>=0 else 'do_not_recommend'
    (OUT/'g2_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
