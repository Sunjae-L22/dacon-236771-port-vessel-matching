"""Join predictions by query_id and keep LLM source/token cohorts distinct."""
from pathlib import Path
import argparse,json
import pandas as pd,numpy as np
OUT=Path(__file__).resolve().parent

def metric(frame):
    result={'rows':len(frame),'vessels':frame.target_vessel_id.nunique()}
    columns=[c for c in ['correct','recall','top20_recall','top100_recall'] if c in frame]
    for column in columns:
        values=frame[column]
        if values.dtype=='object' or pd.api.types.is_string_dtype(values.dtype):values=values.map({'True':True,'False':False,'true':True,'false':False,True:True,False:False})
        assert values.notna().all(),column
        row=values.astype(float);vessel=row.groupby(frame.target_vessel_id).mean();result[column+'_row']=float(row.mean());result[column+'_vessel_macro']=float(vessel.mean())
        if column=='correct':
            rng=np.random.default_rng(970220);a=vessel.to_numpy();result['correct_vessel_bootstrap95']=np.quantile(rng.choice(a,size=(5000,len(a)),replace=True).mean(1),[.025,.975]).tolist()
    return result

def main(args):
    meta=pd.read_csv(OUT/'llm_holdout_fold2.csv');pred=pd.read_csv(args.predictions);assert pred.query_id.is_unique
    selected=pred[pred.query_id.isin(set(meta.query_id))].copy();assert len(selected)==len(meta),f'Need every LLM holdout query, found {len(selected)}/{len(meta)}'
    if 'target_vessel_id' in selected:
        truth=dict(zip(meta.query_id,meta.target_vessel_id));assert all(truth[q]==t for q,t in zip(selected.query_id,selected.target_vessel_id))
        selected=selected.drop(columns=['target_vessel_id'])
    d=meta[['query_id','target_vessel_id','generation_source','fit_token_coverage']].merge(selected,on='query_id',validate='one_to_one',suffixes=('','_prediction'))
    result={'overall':metric(d),'by_generation_source':{str(k):metric(g) for k,g in d.groupby('generation_source')},'by_fit_token_coverage':{str(k):metric(g) for k,g in d.groupby('fit_token_coverage')},'cohort':'LLM-generated aliases only; not combined with native Korean or partial-name stress','source_predictions':str(Path(args.predictions).resolve())}
    if args.output:Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output');main(p.parse_args())
