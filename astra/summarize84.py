"""Author: Astra. Reproducible paired vessel bootstrap and artifact QA receipts."""
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def paired(d, target, method, correct, methods):
    # All variants and repeated query rows for a vessel stay in one resampled unit.
    m = d.groupby([target, method])[correct].mean().unstack()
    delta = (m[methods[1]] - m[methods[0]]).to_numpy()
    rng = np.random.default_rng(84)
    boots = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
    return dict(vessels=len(m), baseline=float(m[methods[0]].mean()),
                enhanced=float(m[methods[1]].mean()), delta=float(delta.mean()),
                bootstrap_95_ci=np.quantile(boots, [.025, .975]).tolist())

def main():
    frozen = pd.read_csv(ROOT/'astra/frozen84_ranked/predictions.csv')
    frozen = frozen[frozen.exclude_train & frozen.role.eq('eval')]
    result = {'84_frozen_eval_ship_macro': paired(frozen, 'target', 'method', 'correct', ['baseline', 'expanded'])}
    result['84_frozen_eval_row_mean'] = frozen.groupby('method').correct.mean().to_dict()
    n = pd.read_csv(ROOT/'astra/ngram84/scored_candidates.csv')
    n = n[n.role.eq('eval')].copy()
    picks = []
    for method, weight in [('baseline', 0), ('neural', .1)]:
        n['score'] = n.base_z + weight*n.nn_z
        p = n.sort_values(['query_sample_id','score','rank','candidate_vessel_id'], ascending=[True,False,True,True]).groupby('query_sample_id').head(1).copy()
        p['method'] = method
        picks.append(p)
    result['85_neural_eval_ship_macro'] = paired(pd.concat(picks), 'target_vessel_id', 'method', 'is_target', ['baseline','neural'])
    result['limitations'] = ['No Public LB measured for 84, 85 or 86', 'Accuracy validation is three-way vessel holdout, not full five-fold OOF', 'Five-fold results measure candidate recall only', 'Retrieval development reused holdout feedback; its bootstrap interval is exploratory', 'ByT5 L4 pilot did not improve eval and is excluded from all ZIPs']
    ranker = pd.read_csv(ROOT/'astra/ranker84_fold0/predictions.csv')
    result['retrained_ranker_fold0'] = paired(ranker, 'target', 'method', 'correct', ['baseline','enhanced'])
    result['retrained_ranker_fold0']['full_oof'] = False
    for number in (84,85,86):
        path=ROOT/f'astra/build{number}_manifest.json'
        receipt=json.loads(path.read_text())
        z=ROOT/f'submissions/{number}_submit.zip'
        assert hashlib.sha256(z.read_bytes()).hexdigest()==receipt['sha256']
        log_path=ROOT/f'astra/check{number}.log'
        log=log_path.read_text() if log_path.exists() else ''
        match=re.search(r'rows 3000  time ([\d.]+)s',log)
        if match and 'row independence: subset 600 rows, mismatches 0' in log:
            receipt['local_check']={'status':'PASS','full_rows':3000,'subset_rows':600,'mismatches':0,'full_seconds':float(match[1]),'log':f'astra/check{number}.log'}
            path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        elif number == 85:
            receipt['local_check']={'status':'FAIL','reason':'Local PyTorch/LightGBM coexistence segmentation fault (return code -11), reproduced in minimal test','replacement':86}
            path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        result[str(number)+'_qa']=receipt['local_check']
    (ROOT/'astra/results84_85.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
