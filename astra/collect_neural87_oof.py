"""Author: Astra. Prespecified weight .1; aggregate cross-fitted neural predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def compare(d,base,new):
    m=d[d.method.isin([base,new])].pivot(index=['fold','target','sample_id','variant'],columns='method',values='correct').astype(float)
    vessels=m.groupby(['fold','target']).mean();delta=(vessels[new]-vessels[base]).to_numpy();rng=np.random.default_rng(87)
    bootstrap=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    variants=m.groupby('variant').mean();folds=m.groupby('fold').mean()
    report=dict(vessels=len(vessels),baseline=float(vessels[base].mean()),neural=float(vessels[new].mean()),delta=float(delta.mean()),bootstrap_95=np.quantile(bootstrap,[.025,.975]).tolist(),variants=variants.to_dict('index'),folds=folds.to_dict('index'))
    original='orig' if 'orig' in variants.index else 'native_all_rows'
    report['eligible']=bool(report['delta']>=.005 and report['bootstrap_95'][0]>0 and variants.loc[original,new]>=variants.loc[original,base] and (folds[new]-folds[base]).min()>=-.02)
    return report

def main():
    frames=[];seen=set()
    for f in range(5):
        out=ROOT/f'astra/neural87_fold{f}';manifest=json.loads((out/'eval_manifest.json').read_text());d=pd.read_csv(out/'predictions.csv');targets=set(d.target)
        assert not seen&targets;seen|=targets;assert len(d)==manifest['queries']*4;frames.append(d)
    assert len(seen)==382
    d=pd.concat(frames,ignore_index=True)
    stress=d[d.variant.ne('native_all_rows')];native=d[d.variant.eq('native_all_rows')]
    assert len(native)==1191*4
    report=dict(baseline_fusion=compare(stress,'baseline','baseline_nn'),enhanced_fusion=compare(stress,'enhanced','enhanced_nn'),native_ranker=compare(native,'baseline','enhanced'),native_baseline_fusion=compare(native,'baseline','baseline_nn'),native_enhanced_fusion=compare(native,'enhanced','enhanced_nn'),protocol='5-fold cross-fitted encoders/rankers; fixed 900 steps and weight .1 selected in prior exploratory three-way pilot; not nested CV',gate='delta>=0.5pp, vessel-bootstrap lower95>0, no original loss, no fold loss>2pp; fixed before OOF results')
    d.to_csv(ROOT/'astra/neural87_oof_predictions.csv',index=False)
    (ROOT/'astra/neural87_oof_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
