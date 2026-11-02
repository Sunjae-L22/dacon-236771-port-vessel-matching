"""Frozen Borda/.95 diagnostic on actual C_vote 400-row development execution."""
from pathlib import Path
import os,sys,json,pickle,csv,time,datetime,hashlib,math,unicodedata
sys.dont_write_bytecode=True
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
assert os.environ.get('PYTHONHASHSEED')=='0'
import numpy as np
import lightgbm as lgb
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PAIR=HERE.parent/'pairwise'
DATA=ROOT/'astra/new_direction_20261101/local_cvote/eval'
CUTOFF=datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp()

def checktime():
    if time.time()>=CUTOFF:raise RuntimeError('07:00 cutoff')

def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checktime();start=time.monotonic()
    protocol=json.loads((HERE/'PRODUCTION_PROTOCOL.json').read_text())
    assert sha(Path(__file__))==protocol['source_sha256']
    modelpath=PAIR/'ablation/no_score_all19/fit2plus4.txt'
    lock=json.loads((PAIR/'FOLLOWUP_LOCK.json').read_text())
    assert sha(modelpath)==lock['model_sha256']==protocol['model_sha256']
    model=lgb.Booster(model_file=str(modelpath));assert model.num_feature()==186 and model.num_trees()==250
    prep=json.loads((PAIR/'PRODUCTION_PREPARE.json').read_text())
    assert Path(prep['data_dir']).samefile(DATA)
    assert model.feature_name()==[p+n for p in ('A_','B_','D_') for n in prep['feature_names']]
    assert 'kc_score_difference' not in model.feature_name()
    assert prep['full_candidate_ID_order_mismatches']==prep['top20_and_top1_mismatches']==0
    features={r['sample_id']:r for r in pickle.loads((PAIR/'production112_features.pkl').read_bytes())}
    test=read(DATA/'test.csv')
    en={r['sample_id']:r for r in read(DATA/'en_pred.csv')}
    hs={r['sample_id']:r['target_vessel_id'] for r in read(DATA/'hs_pred.csv')}
    v20={r['sample_id']:r['target_vessel_id'] for r in read(HERE.parent/'v20_hard_pred.csv')}
    assert len({r['sample_id'] for r in test})==len(test)==400

    def one(row):
        checktime();sid=row['sample_id'];e=en[sid]['target_vessel_id'];h=hs[sid]
        gap=float(en[sid]['ko_gap']) if en[sid].get('ko_gap') else float('nan')
        ko=any('가'<=c<='힣' for c in unicodedata.normalize('NFC',row['vessel_name']))
        protected=not ko or gap>=4
        if ko:assert math.isfinite(gap)
        if protected or e==h:incumbent=e
        else:
            assert sid in v20
            incumbent=v20[sid]
        reason='english' if not ko else ('gap_ge4' if gap>=4 else None)
        pred=incumbent;best=None;inc=None;gate=None;borda=None;winner_id=None;sym_error=0.
        if not protected:
            r=features[sid];ids=r['ids'];assert ids[0]==e
            if incumbent not in ids:reason='outside112_top20'
            elif len(ids)<2:reason='singleton'
            else:
                inc=ids.index(incumbent);F=np.asarray(r['F'],np.float32);n=len(ids)
                assert F.shape==(n,62) and np.isfinite(F).all() and len(set(ids))==n
                ii,jj=np.where(~np.eye(n,dtype=bool));A=F[ii];B=F[jj]
                X=np.concatenate([A,B,A-B],axis=1);assert X.shape==(n*(n-1),186)
                raw=np.zeros((n,n));raw[ii,jj]=model.predict(X,num_threads=2)
                sym=.5*(raw+1-raw.T);np.fill_diagonal(sym,0.)
                sym_error=float(np.max(np.abs((sym+sym.T)[ii,jj]-1)))
                assert sym_error<1e-12
                borda=(sym.sum(axis=1)/(n-1)).tolist();best=int(np.argmax(borda));winner_id=ids[best]
                if best!=inc:
                    gate=float(sym[best,inc])
                    if gate>=.95:pred=winner_id
        return {'sample_id':sid,'name':row['vessel_name'],'baseline112':e,'hsw':h,'v20':v20.get(sid),
                'cvote':incumbent,'prediction':pred,'changed':pred!=incumbent,
                'gap110':gap if math.isfinite(gap) else None,'protected_reason':reason,
                'incumbent112_rank':inc+1 if inc is not None else None,'borda_winner112_rank':best+1 if best is not None else None,
                'borda_winner':winner_id,'gate_probability':gate,'borda':borda,'symmetry_error':sym_error}

    out=[one(row) for row in test];byid={r['sample_id']:r for r in out}
    assert all(one(row)==byid[row['sample_id']] for row in reversed(test))
    assert all(not r['changed'] or r['protected_reason'] is None for r in out)
    frozen=HERE/'production_predictions_before_truth.jsonl'
    frozen.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
    predictions_sha=sha(frozen)
    # Labels are first read after the complete prediction file has been saved.
    truth={r['sample_id']:r for r in read(DATA/'truth.csv')}
    assert set(truth)==set(byid)
    for r in out:r.update(target=truth[r['sample_id']]['target_vessel_id'],variant=truth[r['sample_id']]['variant'])
    def measure(rr):
        res=[r for r in rr if r['cvote']!=r['target'] and r['prediction']==r['target']]
        harm=[r for r in rr if r['cvote']==r['target'] and r['prediction']!=r['target']]
        return {'rows':len(rr),'vessels':len({r['target'] for r in rr}),'base_correct':sum(r['cvote']==r['target'] for r in rr),
                'new_correct':sum(r['prediction']==r['target'] for r in rr),'changed':sum(r['changed'] for r in rr),
                'rescue':len(res),'harm':len(harm),'net':len(res)-len(harm),
                'rescue_vessels':len({r['target'] for r in res}),'harm_vessels':len({r['target'] for r in harm})}
    metrics={typ:measure([r for r in out if typ=='all' or r['variant'].startswith('clean')==(typ=='clean')]) for typ in ('clean','noise','all')}
    mechanism={}
    for kind in ('restore112top1','other_selection'):
        rr=[r for r in out if r['changed'] and (r['prediction']==r['baseline112'])==(kind=='restore112top1')]
        mechanism[kind]={**measure(rr),'sample_ids':[r['sample_id'] for r in rr]}
    report={'scope':'Actual frozen C_vote runtime predictions on 400 table-seen synthetic DEVELOPMENT rows; not Public or independent confirmation',
            'model':str(modelpath),'model_sha256':sha(modelpath),'features':186,'threshold':.95,'input_rows':len(out),
            'new_training':False,'new_parameters':False,'threshold_search':False,'fifteen_vessel_confirmation_read':False,
            'eligible_rows':sum(r['protected_reason'] is None for r in out),'outside112_top20_rows':sum(r['protected_reason']=='outside112_top20' for r in out),
            'protected_changes':0,'row_reversal_mismatches':0,'symmetry_error_max':max(r['symmetry_error'] for r in out),
            'predictions_saved_before_truth_sha256':predictions_sha,'metrics':metrics,'mechanism_breakdown':mechanism,'seconds':time.monotonic()-start}
    report['diagnostic_pass']=metrics['clean']['harm']==0 and metrics['noise']['net']>0
    (HERE/'production_CVOTE_ROWS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
    (HERE/'production_CVOTE_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
