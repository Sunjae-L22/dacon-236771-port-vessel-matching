"""CLI: append-only extra cache over incumbent97 IDs; audit only until training authorized."""
from pathlib import Path
import os,sys,json,hashlib,time,argparse,fcntl
os.environ['KO_DICT']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;BASE=ROOT/'astra/incumbent97';SRC=BASE/'src'
sys.path.insert(0,str(SRC));import cv,opus_lib as B
from features import FEATURE_NAMES,extra_features,append_features

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def context(fold):
    out=BASE/f'fold{fold}';plan=json.loads((out/'plan.json').read_text());assert len(plan['features'])==60 and plan['features']==B.FEATURES,'Require matched incumbent60, not a91 surrogate'
    for name,sha in plan['source']['sources'].items():assert file_hash(SRC/name)==sha
    tr,cat=cv.load();fit=tr[cv.folds(tr)!=fold];va=tr[cv.folds(tr)==fold];support=json.loads((out/'dictionary_support.json').read_text())
    B.KA._UNIVERSAL_KO_SUBWORDS=[(r['ko'],r['en']) for r in support['retained']];assert len(B.KA._UNIVERSAL_KO_SUBWORDS)==plan['dictionary_retained']
    ix=B.Index(cat,fit);llm=B.llm_korean_rows(fit,out/'llm_fit.csv');assert len(llm)==plan['llm_rows'];ix.ko=B.KA.KoMatcher(cat,pd.concat([fit,llm],ignore_index=True))
    assert not set(fit.target_vessel_id)&set(va.target_vessel_id)
    provenance={'baseline_plan_sha256':file_hash(out/'plan.json'),'dictionary_support_sha256':file_hash(out/'dictionary_support.json'),'llm_fit_sha256':file_hash(out/'llm_fit.csv'),'feature_code_sha256':file_hash(OUT/'features.py'),'baseline_source_hashes':plan['source']['sources'],'dictionary_retained':plan['dictionary_retained'],'fit_vessels':fit.target_vessel_id.nunique(),'heldout_vessels':va.target_vessel_id.nunique(),'feature_names':FEATURE_NAMES}
    return ix,out,provenance

def sources(out,mode,block,kind):
    if mode in ['audit','train-block']:
        rows=json.loads((out/'blocks'/f'{block}.json').read_text());paths=sorted((out/'features'/block).glob('*.npz'))
        assert paths,f'No completed incumbent block cache yet: {block}'
        return rows,paths
    paths=list(out.glob(f'eval_{kind}_*.npz'));assert len(paths)==1,f'Expected one completed eval cache for {kind}, found {len(paths)}'
    rows=json.loads((out/f'eval_{kind}_queries.json').read_text());return rows,paths

def calculate(ix,rows,path,limit=None):
    with np.load(path) as archive:z={key:archive[key] for key in archive.files}
    X=z['X'];idx=z['idx'];groups=z['groups'];assert X.shape==(len(idx),60) and X.dtype==np.float32;assert groups.sum()==len(X)
    query_index=z['query_index'] if 'query_index' in z else np.arange(len(groups));assert len(query_index)==len(groups)
    original_hash=array_hash(X);index_hash=array_hash(idx);E=np.empty((len(X),3),np.float32);offset=0;checked=0;nonzero_queries=0
    for local,(qindex,size) in enumerate(zip(query_index,groups)):
        if limit is not None and checked>=limit:break
        row=rows[int(qindex)];assert ix.id2i[row['target_vessel_id']]==int(z['target'][local]);pool=idx[offset:offset+size];F=X[offset:offset+size]
        E[offset:offset+size]=extra_features(ix.ko,row.get('vessel_name'),pool,F[:,B.FEATURES.index('ko_tail_sim')]);augmented=np.column_stack((F,E[offset:offset+size]))
        assert np.array_equal(augmented[:,:60].view(np.uint32),F.view(np.uint32))
        nonzero_queries+=bool(np.any(E[offset:offset+size]));checked+=1;offset+=size
    assert array_hash(X)==original_hash and array_hash(idx)==index_hash,'Base features or candidates mutated'
    meta=dict(source_cache_sha256=file_hash(path),original_feature_array_sha256=original_hash,candidate_idx_sha256=index_hash,query_count=int(checked),candidate_rows=int(offset),queries_with_nonzero_extra=int(nonzero_queries),nonzero_counts=(E[:offset]!=0).sum(0).tolist(),extra_min=E[:offset].min(0).tolist() if offset else [],extra_max=E[:offset].max(0).tolist() if offset else [],base_feature_bit_mismatches=0,candidate_mutations=0)
    return E[:offset],z,meta

def guard_tests():
    class Dummy:
        cat_tok_e2k=[[('x','x','x','x'),('ab','ab','ab','b')],[('x','x','x','x'),('ba','ba','ba','b')]]
        def __init__(self,tail):self.tail=tail
        def translate_ko(self,name):return {'tok_projections':[('zzzzz','zzzzz','zz'),self.tail]}
    old=np.array([.2,.1],np.float32);r=extra_features(Dummy(('a','a','')),'가 아',[0,1],old)
    assert np.isclose(r[0,0],.88) and np.isclose(r[0,1],.43) and np.isclose(r[0,2],.68)
    assert np.all(r[1]==0)
    for token in [('', '', ''),('', 'z', '')]:assert np.all(extra_features(Dummy(token),'가 아',[0,1],old)==0)
    assert np.all(extra_features(Dummy(('a','a','')),'ASCII',[0,1],old)==0)
    assert extra_features(Dummy(('a','a','')),'가 아',[],np.zeros(0,np.float32)).shape==(0,3)
    return {'empty_translation_guard':'PASS','empty_roman_token_guard':'PASS','nonhangul_zero':'PASS','positive_partial_prefix_and_gain':'PASS','no_candidate_case':'PASS'}

def main(args):
    start=time.time();guards=guard_tests();ix,baseline,provenance=context(args.fold);rows,paths=sources(baseline,args.mode,args.block,args.kind)
    if args.mode=='audit':
        extra,z,meta=calculate(ix,rows,paths[0],limit=args.max_queries);report={'status':'PASS','mode':'audit_only_no_training','fold':args.fold,'source_block':args.block,'source_chunk':str(paths[0]),'guards':guards,'provenance':provenance,**meta,'seconds':time.time()-start}
        dump(OUT/'audit.json',report);print(json.dumps(report,ensure_ascii=False,indent=2));return
    reports=[]
    for path in paths:
        category=args.block if args.mode=='train-block' else args.kind;dst=OUT/f'fold{args.fold}'/args.mode/category/path.name;dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if dst.exists():
                old=json.loads(dst.with_suffix('.json').read_text());assert old['source_cache_sha256']==file_hash(path) and old['provenance']==provenance;reports.append(old);continue
            E,z,meta=calculate(ix,rows,path);arrays={key:z[key] for key in ['idx','groups','target','query_index','allowed'] if key in z};arrays['extra_features']=E
            temp=dst.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(dst);report={**meta,'provenance':provenance,'output_extra_cache_sha256':file_hash(dst)};dump(dst.with_suffix('.json'),report);reports.append(report);print('EXTRA_CACHE',category,path.name,meta['query_count'],meta['candidate_rows'],flush=True)
    dump(OUT/f'fold{args.fold}'/f'{args.mode}_{args.block if args.mode=="train-block" else args.kind}_report.json',dict(status='EXTRA_FEATURE_CACHE_ONLY',reports=reports,seconds=time.time()-start,training_run=False,submission_created=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=2);p.add_argument('--mode',choices=['audit','train-block','eval'],default='audit');p.add_argument('--block',default='seed0_corrupt');p.add_argument('--kind',choices=['old9','benchmark','llm_holdout'],default='old9');p.add_argument('--max-queries',type=int,default=80);main(p.parse_args())
