"""Independent replay of outputs and pairwise agreement. Never reads labels."""
from pathlib import Path
import json,unicodedata,hashlib,platform,sys
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;E=R/'astra/erde_qa_20261031'
def read(p):return pd.read_csv(p,dtype=str,keep_default_na=False).set_index('sample_id')
t=read(D/'inputs/full.csv');out=read(D/'full/pred.csv');en=read(D/'full/en_pred.csv');ko=read(D/'full/ko_pred.csv');hs=read(D/'full/hs_pred.csv')
cat=pd.read_csv(R/'data/open/vessel_catalog.csv',dtype=str);valid=set(cat.vessel_id)
for df in [out,en,ko,hs]:assert df.index.is_unique and set(df.target_vessel_id)<=valid
assert en.index.tolist()==t.index.tolist()
hang=t.vessel_name.map(lambda x:any('가'<=c<='힣' for c in unicodedata.normalize('NFC',x)))
assert set(ko.index)==set(hs.index)==set(t[hang].index)
gap=pd.to_numeric(en.ko_gap,errors='coerce').reindex(t.index)
assert gap[hang].notna().all()
replay=[];trace=[]
for sid,row in t.iterrows():
    c=en.loc[sid,'target_vessel_id'];a=ko.loc[sid,'target_vessel_id'] if hang.loc[sid] else None;b=hs.loc[sid,'target_vessel_id'] if hang.loc[sid] else None
    if not hang.loc[sid]:p,route=c,'english'
    elif gap.loc[sid]>=4:p,route=c,'sure'
    else:
        votes=[a,b,c];w=[x for x in set(votes) if votes.count(x)>=2]
        p,route=(w[0],'majority') if w else (a,'tie_to_ko93_proxy')
    replay.append(p);trace.append(dict(sample_id=sid,is_hangul=bool(hang.loc[sid]),gap110=None if pd.isna(gap.loc[sid]) else float(gap.loc[sid]),ours112=c,ko93_proxy=a,hsw=b,pred=p,route=route))
assert replay==out.loc[t.index].target_vessel_id.tolist()
pd.DataFrame(trace).to_csv(D/'routing_trace.csv',index=False)
changed=out.loc[t.index].target_vessel_id!=en.loc[t.index].target_vessel_id
scope=hang & (gap<4)
report={'rows':len(t),'korean_rows':int(hang.sum()),'english_rows':int((~hang).sum()),'independent_routing_replay_mismatches':0,'changed_vs112_K':int((changed&hang).sum()),'changed_vs112_E':int((changed&~hang).sum()),'changed_outside_scope':int((changed&~scope).sum()),'scope_rows':int(scope.sum()),'route_counts':pd.Series([x['route'] for x in trace]).value_counts().to_dict(),'labels_read':False,'actual_v20_tested':False}
assert report['changed_outside_scope']==0
for component in ['en','ko','hs']:
    a=read(D/f'full/{component}_pred.csv');b=read(D/f'subset/{component}_pred.csv')
    assert (a.loc[b.index].target_vessel_id==b.target_vessel_id).all()
    report[f'{component}_subset_rows']=len(b);report[f'{component}_subset_mismatches']=0
(D/'ROUTING_REPLAY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
if (E/'full_pred.csv').exists():
    er=read(E/'full_pred.csv');assert er.index.tolist()==t[hang].index.tolist()
    assert set(er.target_vessel_id)<=valid
    scores={}
    for name,ids in [('all_korean',t[hang].index),('voting_scope_110gap_lt4',t[scope].index),('protected_110gap_ge4',t[hang & (gap>=4)].index)]:
        s={'rows':len(ids)}
        for key,a,b in [('erde_112',er,en),('erde_hsw',er,hs),('112_hsw',en,hs)]:
            n=int((a.loc[ids].target_vessel_id==b.loc[ids].target_vessel_id).sum());s[key]={'same':n,'fraction':n/len(ids) if len(ids) else None}
        scores[name]=s
    (E/'AGREEMENT.json').write_text(json.dumps({'cohort':'corrupted_train_local_check_seed7; no accuracy or Public extrapolation','agreement':scores},ensure_ascii=False,indent=2))
    print(json.dumps(scores,ensure_ascii=False,indent=2))
original=json.loads((D/'source_hashes.json').read_text());unchanged={p:hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in original.items()}
(D/'FINAL_SOURCE_CHECK.json').write_text(json.dumps(unchanged,indent=2));assert all(unchanged.values())
(D/'ENVIRONMENT.json').write_text(json.dumps({'platform':platform.platform(),'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,'runtime_note':'Mac local CPU; overlapping independent QA jobs. Not actual v20 or L4 timing.'},indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
