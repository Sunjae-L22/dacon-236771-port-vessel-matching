"""Report the prespecified translated-string ablation against rankless88."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv');parts=[];reports={}
    for fold in range(5):
        path=ROOT/f'astra/translated90_fold{fold}/predictions.csv'
        if not path.exists():continue
        new=pd.read_csv(path).rename(columns={'correct':'translated'})
        old=pd.read_csv(ROOT/f'astra/rankless87_fold{fold}/predictions.csv').rename(columns={'correct':'rankless88'})
        old['target']=old.target.map(lambda i:cat.vessel_id.iloc[int(i)])
        d=new.merge(old[['fold','sample_id','variant','target','rankless88']],on=['fold','sample_id','variant','target'],validate='one_to_one')
        assert len(d)==len(new)==len(old)
        parts.append(d);s=d[d.variant!='native_all_rows'];native=d[d.variant=='native_all_rows'];orig=d[d.variant=='orig']
        delta=float(s.translated.mean()-s.rankless88.mean())
        reports[str(fold)]={'baseline':float(s.rankless88.mean()),'translated':float(s.translated.mean()),'delta':delta,'native_baseline':float(native.rankless88.mean()),'native_translated':float(native.translated.mean()),'by_variant':d.groupby('variant')[['rankless88','translated']].mean().to_dict('index')}
        reports[str(fold)]['pilot_gate']=bool(delta>=.005 and native.translated.mean()>=native.rankless88.mean() and orig.translated.mean()>=orig.rankless88.mean())
    if not parts:return
    d=pd.concat(parts,ignore_index=True);stress=d[d.variant!='native_all_rows'];v=stress.groupby(['fold','target'])[['rankless88','translated']].mean();a=(v.translated-v.rankless88).to_numpy();rng=np.random.default_rng(90)
    ci=np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()
    result={'folds':reports,'vessels':len(v),'delta':float(a.mean()),'bootstrap95':ci,'limits':'Exploratory method comparison on reused outer folds; not a Public LB forecast or comparison to full-data79 ensemble.'}
    (ROOT/'astra/translated90_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
