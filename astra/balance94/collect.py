"""Collect the prespecified two-fold vessel weighting pilot; no further tuning."""
from pathlib import Path
import json,sys,collections
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'astra'),str(ROOT/'shared/experiments/pinned/build84_astra')]
from coldship91 import training_rows
import cv

def main():
    tr,cat=cv.load();folds=cv.folds(tr);reports={};all_predictions=[]
    for fold in [0,4]:
        directory=OUT/f'fold{fold}';r=json.loads((directory/'report.json').read_text())
        queries,_=training_rows(tr[folds!=fold],fold);rows=queries.to_dict('records');counts=collections.Counter(row['target_vessel_id'] for row in rows)
        w=np.array([counts[row['target_vessel_id']]**-.5 for row in rows]);w/=w.mean();before=collections.Counter();after=collections.Counter()
        for row,weight in zip(rows,w):source=str(row['sample_id']).split('_')[0];before[source]+=1;after[source]+=float(weight)
        diagnostic=dict(query_counts=dict(before),normalized_weight_mass=dict(after),before_fraction={k:v/sum(before.values()) for k,v in before.items()},after_fraction={k:v/sum(after.values()) for k,v in after.items()},interpretation='Vessel weighting also changes the native versus LLM training mass; the effects are not isolated.')
        (directory/'weight_diagnostics.json').write_text(json.dumps(diagnostic,indent=2));r['source_mix']=diagnostic;reports[f'fold{fold}']=r
        p=pd.read_csv(directory/'paired_predictions.csv');all_predictions.append(p)
    combined=pd.concat(all_predictions);combined.to_csv(OUT/'paired_predictions.csv',index=False)
    confirm=reports['fold4'];reports['eligible_for_broader_validation']=bool(confirm['stress_delta']>=.005 and confirm['bootstrap95'][0]>0 and confirm['native_delta']>=0 and confirm['original_delta']>=0)
    reports['limits']=['Only development fold0 and confirmation fold4, not full 5-fold OOF.', 'Source mixture changes with the vessel weights.', 'Each model is one 600-tree seed84 ranker; no direct heldout comparison against full-data Public79 ensemble.', 'Public LB is unknown; this pilot has no submission ZIP.']
    (OUT/'report.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
