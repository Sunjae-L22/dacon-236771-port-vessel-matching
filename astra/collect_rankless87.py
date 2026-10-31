"""Author: Astra. Rankless ablation: folds 1-2 development, 3-4 confirmation."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def compare(d,base='baseline'):
    m=d[d.method.isin([base,'rankless'])].pivot(index=['fold','target','sample_id','variant'],columns='method',values='correct').astype(float)
    vessels=m.groupby(['fold','target']).mean();delta=(vessels.rankless-vessels[base]).to_numpy();rng=np.random.default_rng(88)
    ci=np.quantile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[.025,.975]).tolist()
    variants=m.groupby('variant').mean();folds=m.groupby('fold').mean();original='orig' if 'orig' in variants.index else 'native_all_rows'
    nonnoise=variants.loc[[v for v in variants.index if not v.endswith('noise')]]
    result=dict(vessels=len(vessels),baseline=float(vessels[base].mean()),rankless=float(vessels.rankless.mean()),delta=float(delta.mean()),bootstrap_95=ci,variants=variants.to_dict('index'),folds=folds.to_dict('index'))
    result['eligible']=bool(result['delta']>=.01 and ci[0]>0 and variants.loc[original,'rankless']>=variants.loc[original,base] and (nonnoise.rankless-nonnoise[base]).mean()>=0 and (folds.rankless-folds[base]).min()>=-.02)
    return result

def main():
    cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv');rows=[]
    for f in range(5):
        d=pd.read_csv(ROOT/f'astra/rankless87_fold{f}/predictions.csv');d['target']=d.target.map(lambda i:cat.vessel_id.iloc[int(i)])
        b=pd.read_csv(ROOT/f'astra/neural87_fold{f}/predictions.csv');b=b[b.method.isin(['baseline','enhanced'])]
        assert len(d)*2==len(b)
        check=d.merge(b[b.method.eq('baseline')],on=['fold','sample_id','variant'],suffixes=('_new','_base'));assert len(check)==len(d) and check.target_new.eq(check.target_base).all()
        rows.extend([d,b])
    d=pd.concat(rows,ignore_index=True);stress=d[d.variant.ne('native_all_rows')];native=d[d.variant.eq('native_all_rows')]
    assert stress.target.nunique()==382 and len(native)==1191*3
    primary=compare(stress[stress.fold.isin([3,4])]);native_report=compare(native)
    report=dict(primary_confirmation_folds_3_4=primary,all_five_folds=compare(stress),native_all_rows=native_report,versus_ranked_enhanced=compare(stress,'enhanced'),eligible_for_full_training=bool(primary['eligible'] and native_report['delta']>=0),protocol='One rank feature removed; same enhanced candidate set, training queries and other 70 features. Folds1-2 inspected during development; folds3-4 reserved confirmation.',gate='Confirmation gain>=1pp, lower vessel-bootstrap95>0, no orig loss, no non-noise mean loss, no fold loss>2pp, all-native-vessel mean nonnegative; fixed before confirmation')
    d.to_csv(ROOT/'astra/rankless87_oof_predictions.csv',index=False);(ROOT/'astra/rankless87_oof_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
