"""Frozen all-pairs Borda policy, reusing no_score_all19 classifiers without fit."""
from pathlib import Path
import os,sys,time,datetime,json,pickle,hashlib
sys.dont_write_bytecode=True
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
import numpy as np
import lightgbm as lgb

HERE=Path(__file__).resolve().parent
PAIR=HERE.parent/'pairwise'
START=time.monotonic()
CUTOFF=datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp()

def checktime():
    if time.time()>=CUTOFF or time.monotonic()-START>40*60:
        raise RuntimeError('07:00 cutoff or 40-minute watchdog')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def choose(F, ids, incumbent, hard, model):
    """No labels, query scores, metadata truth, or target-in-top20 flags accepted."""
    if not hard or len(ids)<2:
        return {'prediction':incumbent,'borda_winner':incumbent,'gate_probability':None,
                'borda':None,'symmetry_error':0.,'pair_count':0}
    F=np.asarray(F,np.float32)
    assert F.shape==(len(ids),62) and np.isfinite(F).all()
    assert len(set(ids))==len(ids) and ids[0]==incumbent
    n=len(ids);ii,jj=np.where(~np.eye(n,dtype=bool))
    A=F[ii];B=F[jj]
    X=np.concatenate([A,B,A-B],axis=1)
    assert X.shape==(n*(n-1),186)
    raw=np.zeros((n,n),np.float64)
    raw[ii,jj]=model.predict(X,num_threads=2)
    assert np.isfinite(raw).all()
    sym=.5*(raw+1-raw.T)
    np.fill_diagonal(sym,0.)
    borda=np.sum(sym,axis=1)/(n-1)
    winner=int(np.argmax(borda))  # stable original candidate order for equal values
    gate=float(sym[winner,0]) if winner!=0 else None
    pred=ids[winner] if winner!=0 and gate>=.95 else incumbent
    error=float(np.max(np.abs((sym+sym.T)[ii,jj]-1)))
    assert error<1e-12
    return {'prediction':pred,'borda_winner':ids[winner],'borda_winner_rank':winner+1,
            'gate_probability':gate,'borda':borda.tolist(),'symmetry_error':error,
            'pair_count':n*(n-1)}

def metrics(rows):
    out={}
    for name in ('clean','noise','all'):
        rr=[r for r in rows if name=='all' or r['variant'].startswith('clean')==(name=='clean')]
        rescue=[r for r in rr if r['baseline']!=r['target'] and r['prediction']==r['target']]
        harm=[r for r in rr if r['baseline']==r['target'] and r['prediction']!=r['target']]
        net_by_ship={}
        for r in rr:
            net_by_ship[r['target']]=net_by_ship.get(r['target'],0)+int(r['prediction']==r['target'])-int(r['baseline']==r['target'])
        out[name]={'rows':len(rr),'vessels':len(net_by_ship),'changed':sum(r['prediction']!=r['baseline'] for r in rr),
                   'rescue':len(rescue),'harm':len(harm),'net':len(rescue)-len(harm),
                   'rescue_vessels':len({r['target'] for r in rescue}),'harm_vessels':len({r['target'] for r in harm}),
                   'net_positive_vessels':sum(v>0 for v in net_by_ship.values()),'net_negative_vessels':sum(v<0 for v in net_by_ship.values()),
                   'hard_rows':sum(r['hard'] for r in rr),'target_absent_top20':sum(r['hard'] and not r['truth_in_top20'] for r in rr)}
    return out

def evaluate(fitfold, confirmfold):
    checktime()
    modelpath=PAIR/f'ablation/no_score_all19/fit{fitfold}.txt'
    model=lgb.Booster(model_file=str(modelpath))
    meta=json.loads((PAIR/f'ablation/no_score_all19/fit{fitfold}_TRAIN.json').read_text())
    fitships=set(meta['fit_vessel_ids'])
    assert model.num_feature()==186 and 'kc_score_difference' not in model.feature_name()
    rows=pickle.loads((PAIR/f'fold{confirmfold}_features.pkl').read_bytes())
    assert not fitships & {r['target'] for r in rows}
    out=[];pairs=0;maximum_error=0.;validation=[]
    for k,r in enumerate(rows):
        checktime()
        hard=bool(r['korean'] and r['gap110']<4)
        assert hard==r['hard']
        result=choose(r.get('F'),r.get('ids',[]),r['baseline'],hard,model)
        pairs+=result['pair_count'];maximum_error=max(maximum_error,result['symmetry_error'])
        if hard and len(validation)<3:
            # Independent permutation check, original incumbent returned first.
            idx=[0]+list(range(len(r['ids'])-1,0,-1))
            alternate=choose(r['F'][idx],[r['ids'][i] for i in idx],r['baseline'],True,model)
            reordered=np.asarray(alternate['borda'])[np.argsort(idx)]
            assert np.allclose(result['borda'],reordered,atol=1e-12)
            if len(set(result['borda']))==len(result['borda']):
                assert alternate['prediction']==result['prediction']
            validation.append(r['query_id'])
        record={k:v for k,v in r.items() if k not in ('F','scores')}
        record.update(result);out.append(record)
    protected=sum(not r['hard'] and r['prediction']!=r['baseline'] for r in out)
    assert protected==0
    m=metrics(out)
    report={'fit_fold':fitfold,'confirm_fold':confirmfold,'metrics':m,'screen_pass':m['clean']['harm']==0 and m['noise']['net']>0,
            'protected_changes':protected,'symmetry_error_max':maximum_error,'ordered_pairs_scored':pairs,
            'permutation_check_queries':validation,'fit_confirmation_vessel_overlap':0,
            'fixed_gate':.95,'model_sha256':sha(modelpath),'feature_width':186,
            'scope':'Cross-fold pair head on single-model OOF112 features. Not strict nested CV, not C_vote, not Public.'}
    tag=f'fit{fitfold}_confirm{confirmfold}'
    (HERE/f'{tag}_ROWS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
    (HERE/f'{tag}_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report

def main():
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    assert protocol['source_sha256']==sha(Path(__file__))
    reports=[evaluate(2,4),evaluate(4,2)]
    result={'directions':reports,'both_pass':all(r['screen_pass'] for r in reports),'seconds':time.monotonic()-START,
            'new_training':False,'actual400_read':False,'fifteen_vessel_confirmation_read':False,
            'submissions':0,'threshold_search':False,'new_candidate_zip':False}
    (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'both_pass':result['both_pass'],'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
