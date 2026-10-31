"""Astra G3 prototype: existing72 mixed-language rankers score an English reverse-transliteration.

Fixed .25 auxiliary weight, unchanged candidate pool; this does NOT claim en65's98% accuracy.
"""
from pathlib import Path
import os,sys,json,time,unicodedata
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astra'))
import incumbent97 as I
import numpy as np,pandas as pd,lightgbm as lgb

def main():
    start=time.time();_,cat,fit,va,ix,llm,base=I.context(2)
    z=np.load(next(base.glob('eval_benchmark_*.npz')))
    rows=json.loads((base/'eval_benchmark_queries.json').read_text())
    scores=np.load(base/'benchmark_6models_all_scores.npy')
    models=[lgb.Booster(model_file=str(base/f'seed{s}/ranker.txt')) for s in I.SEEDS]
    ids,F,allowed,targets=(z[k] for k in ('idx','X','allowed','target'))
    ci={n:i for i,n in enumerate(I.B.FEATURES)};lo=0;result=[]
    for j,n in enumerate(z['groups']):
        pool=np.arange(lo,lo+n);valid=pool[allowed[pool]]
        if not len(valid):valid=pool
        rank=sorted(valid,key=lambda k:(-float(scores[k]),int(ids[k])))
        leader=rank[0];gap=float(scores[rank[0]]-scores[rank[1]]) if len(rank)>1 else 99.
        row=rows[j];name=unicodedata.normalize('NFC',str(row['vessel_name']))
        routed=I.B.is_hangul(name);winner=leader;trans='';used=False
        if row.get('source_language')=='ko' and routed and gap<4:
            trans=ix.ko.translate_ko(name)['trans']
            if trans and not I.B.is_hangul(trans):
                q={k:row.get(k) for k in I.MODEL_FIELDS};q['vessel_name']=trans
                # Preserve the original candidate pool AND retrieval ranks. Only name-feature view changes.
                cand={int(ids[k]):[float(F[k,ci['rank_name']]),float(F[k,ci['rank_gt']]),999] for k in pool}
                idx,E=I.B.query_features(ix,q,cand);assert idx==ids[pool].tolist()
                aux=sum(m.predict(E,num_threads=2) for m in models)/6
                mixed=.75*scores[pool]+.25*aux
                winner=min(valid,key=lambda k:(-float(mixed[k-lo]),int(ids[k])));used=True
        if row.get('source_language')=='ko':
            result.append(dict(query_id=row.get('query_id'),vessel=row['target_vessel_id'],variant=row['variant'],
                routed_ko=routed,hard=routed and gap<4,used=used,trans=trans,baseline_correct=bool(ids[leader]==targets[j]),
                g3_correct=bool(ids[winner]==targets[j]),changed=bool(ids[leader]!=ids[winner])))
        lo+=n
        if j%500==0:print('G3_PROGRESS',j,len(rows),round(time.time()-start),flush=True)
    df=pd.DataFrame(result);df.to_csv(OUT/'g3_fold2_rows.csv',index=False);df=df[df.routed_ko].copy()
    def summary(g):
        rescue=int((~g.baseline_correct&g.g3_correct).sum());damage=int((g.baseline_correct&~g.g3_correct).sum())
        return dict(n=len(g),hard=int(g.hard.sum()),used=int(g.used.sum()),changed=int(g.changed.sum()),rescue=rescue,damage=damage,net=rescue-damage)
    report=dict(benchmark=summary(df),native=summary(df[df.variant.eq('native_all_rows')]),weight=.25,
        models='72 mixed-language six rankers reused as English auxiliary, not en65',
        limitations=['Existing development fold2','Synthetic gain not Public evidence','Counterfactual English feature view preserves Korean retrieval ranks',
                    'Pure compatibility-Jamo source-ko queries route as English in production and are excluded from primary metrics'])
    pd.DataFrame([dict(variant=v,**summary(g)) for v,g in df.groupby('variant')]).to_csv(OUT/'g3_variants.csv',index=False)
    report['recommendation']='reviewable_experimental_candidate' if report['benchmark']['net']>0 and report['native']['net']>=0 else 'do_not_recommend'
    (OUT/'g3_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
