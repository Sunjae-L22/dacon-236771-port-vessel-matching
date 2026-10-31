"""Author: Astra. Aggregate untouched folds 1-4 separately from development fold 0."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def describe(d):
    m=d.pivot(index=['fold','target','variant'],columns='method',values='correct').astype(float)
    per_vessel=m.groupby(['fold','target']).mean()
    delta=per_vessel.enhanced-per_vessel.baseline
    rng=np.random.default_rng(87)
    bootstrap=delta.to_numpy()[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    variants=m.groupby('variant').mean()
    folds=m.groupby('fold').mean()
    report=dict(vessels=len(per_vessel),baseline=float(per_vessel.baseline.mean()),enhanced=float(per_vessel.enhanced.mean()),delta=float(delta.mean()),bootstrap_95=np.quantile(bootstrap,[.025,.975]).tolist(),variants=variants.to_dict('index'),folds=folds.to_dict('index'),corrections=int(((m.baseline==0)&(m.enhanced==1)).sum()),regressions=int(((m.baseline==1)&(m.enhanced==0)).sum()))
    nonnoise=variants.loc[[v for v in variants.index if not v.endswith('noise')]]
    report['eligible_for_full_training']=bool(report['delta']>=.01 and report['bootstrap_95'][0]>0 and variants.loc['orig','enhanced']>=variants.loc['orig','baseline'] and (nonnoise.enhanced-nonnoise.baseline).mean()>=0 and (folds.enhanced-folds.baseline).min()>=-.02)
    return report

def main():
    rows=[];seen=set()
    for fold in range(5):
        out=ROOT/f'astra/ranker84_fold{fold}'
        manifest=json.loads((out/'manifest.json').read_text())
        d=pd.read_csv(out/'predictions.csv');d['fold']=fold
        targets=set(d.target)
        assert not targets&seen;seen|=targets
        assert len(targets)==manifest['validation_vessels']
        assert len(d)==len(targets)*9*2
        assert d.groupby(['target','method']).size().eq(9).all()
        rows.append(d)
    assert len(seen)==382
    d=pd.concat(rows,ignore_index=True)
    report=dict(primary_confirmation_folds_1_to_4=describe(d[d.fold.ne(0)]),all_five_folds=describe(d),protocol='Target-vessel disjoint; distinct Korean vessels; fixed augmentation seed; training-only statistics and LLM filtering/rescoring; KO_DICT=0; same single-seed augmentation for paired methods',gate='delta >=1pp, vessel-bootstrap lower95 >0, no original loss, non-noise mean >=0, no fold loss >2pp; fixed before confirmation results')
    d.to_csv(ROOT/'astra/ranker84_oof_predictions.csv',index=False)
    (ROOT/'astra/ranker84_oof_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
