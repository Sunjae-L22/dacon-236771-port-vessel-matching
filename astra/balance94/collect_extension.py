"""Aggregate completed folds, including early futility stop; preserve prior pilot report."""
from pathlib import Path
import json,hashlib,sys,collections
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'astra'),str(ROOT/'shared/experiments/pinned/build84_astra')]
from coldship91 import training_rows
import cv

def main():
    plan=json.loads((OUT/'extension_plan.json').read_text());assert hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest()==plan['original_two_fold_report_sha256']
    stop=json.loads((OUT/'early_stop.json').read_text()) if (OUT/'early_stop.json').exists() else None
    observed=stop['completed_folds'] if stop else list(range(5))
    tr,cat=cv.load();folds=cv.folds(tr);pieces=[];fold_reports={};source_mix={}
    for fold in observed:
        directory=OUT/f'fold{fold}';fold_reports[str(fold)]=json.loads((directory/'report.json').read_text());p=pd.read_csv(directory/'paired_predictions.csv');pieces.append(p)
        q,_=training_rows(tr[folds!=fold],fold);rows=q.to_dict('records');counts=collections.Counter(row['target_vessel_id'] for row in rows);w=np.array([counts[row['target_vessel_id']]**-.5 for row in rows]);w/=w.mean();before=collections.Counter();after=collections.Counter()
        for row,weight in zip(rows,w):source=str(row['sample_id']).split('_')[0];before[source]+=1;after[source]+=float(weight)
        source_mix[str(fold)]={'before':{k:v/sum(before.values()) for k,v in before.items()},'after':{k:v/sum(after.values()) for k,v in after.items()}}
    d=pd.concat(pieces,ignore_index=True);assert not d.duplicated(['target','variant','sample_id']).any();d.to_csv(OUT/'extension_predictions.csv',index=False)
    stress=d[d.variant!='native_all_rows'];native=d[d.variant=='native_all_rows'];original=d[d.variant=='orig'];noise=d[d.variant=='trunc_noise'];vessel=(stress.assign(delta=stress.correct.astype(float)-stress.baseline_correct.astype(float)).groupby('target').delta.mean().to_numpy())
    rng=np.random.default_rng(9405);ci=np.quantile(rng.choice(vessel,size=(10000,len(vessel)),replace=True).mean(1),[.025,.975]).tolist();delta=float(vessel.mean());native_delta=float(native.correct.mean()-native.baseline_correct.mean());original_delta=float(original.correct.mean()-original.baseline_correct.mean());noise_delta=float(noise.correct.mean()-noise.baseline_correct.mean())
    gates=dict(mean_gain_at_least_005=delta>=.005,bootstrap95_lower_positive=ci[0]>0,native_no_loss=native_delta>=-1e-12,each_fold_loss_at_most_01=min(r['stress_delta'] for r in fold_reports.values())>=-.01,trunc_noise_no_loss=noise_delta>=-1e-12,original_no_loss=original_delta>=-1e-12)
    report=dict(status='EARLY_STOP_FUTILITY' if stop else 'exploratory_extension_after_inconclusive_confirmation',original_two_fold_gate='FAILED; original report.json unchanged',completed_folds=observed,full_5fold_oof=len(observed)==5,vessels=len(vessel),stress_queries=len(stress),stress_baseline=float(stress.baseline_correct.mean()),stress_balance=float(stress.correct.mean()),vessel_mean_delta=delta,bootstrap95=ci,native_baseline=float(native.baseline_correct.mean()),native_balance=float(native.correct.mean()),native_delta=native_delta,original_delta=original_delta,trunc_noise_delta=noise_delta,corrected=int(((~stress.baseline_correct)&stress.correct).sum()),regressed=int((stress.baseline_correct&(~stress.correct)).sum()),fold_reports=fold_reports,variants=d.groupby('variant')[['baseline_correct','correct']].mean().to_dict('index'),source_mix=source_mix,gates=gates,gate_scope='Observed-fold diagnostics only; maximum-fold-loss failure is already decisive' if stop else 'Full five-fold aggregate',eligible_for_full_training=bool(len(observed)==5 and all(gates.values())),early_stop=stop,limitations=plan['limitations']+(['Only completed folds0,1,4 are summarized; fold2 training aborted and fold3 not started. This is not full 5-fold OOF.'] if stop else []))
    (OUT/'extension_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
