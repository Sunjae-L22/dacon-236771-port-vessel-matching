"""Build allowed-data-only package for one fixed cross-encoder experiment."""
import os
os.environ['KO_DICT']='0';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,random,sys,time,unicodedata,zipfile
import numpy as np,pandas as pd
from rapidfuzz import process,fuzz
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;PACKAGE=OUT/'package';DATA=PACKAGE/'data';DATA.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'));sys.path.insert(0,str(PACKAGE))
import cv,opus_lib as B
from model import normalize,romanize,fields

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    started=time.time();tr,cat=cv.load();folds=cv.folds(tr);fit=tr[np.isin(folds,[0,1,3])].copy();tune=tr[folds==4].copy();outer=tr[folds==2].copy()
    fitids=set(fit.target_vessel_id);tuneids=set(tune.target_vessel_id);outerids=set(outer.target_vessel_id)
    assert not fitids&(tuneids|outerids) and not tuneids&outerids
    ix=B.Index(cat,fit);assert B.KA._UNIVERSAL_KO_SUBWORDS==[]
    catalog=[{'id':str(v),'name':str(n),'norm':normalize(n),'phon':str(p)} for v,n,p in zip(cat.vessel_id,ix.names_raw,ix.ko.cat_en2ko)]
    id2i={r['id']:i for i,r in enumerate(catalog)};cn=[r['norm'] for r in catalog];cp=[r['phon'] for r in catalog]
    genpath=ROOT/'shared/experiments/pinned/build70/llm_korean_scored.csv';gen=pd.read_csv(genpath);gen=gen[gen.vessel_id.isin(fitids)].copy()
    gen['score']=[ix.ko.score(q,id2i[v])['ko_combined_score'] for q,v in zip(gen.ko,gen.vessel_id)]
    gen.to_csv(DATA/'llm_rescored_fit013.csv',index=False)
    llm=B.llm_korean_rows(fit,DATA/'llm_rescored_fit013.csv');native=fit[fit.vessel_name.map(B.is_hangul)].copy()
    pairs=[]
    for source,frame in [('native',native),('llm',llm)]:
        for row in frame.to_dict('records'):
            assert row['target_vessel_id'] in fitids
            pairs.append({'q':str(row['vessel_name']),'t':id2i[row['target_vessel_id']],'source':source,'sample_id':str(row['sample_id'])})
    aux=np.random.default_rng(97).choice(len(cat),size=min(4096,len(cat)),replace=False)
    for i in aux:pairs.append({'q':catalog[int(i)]['name'],'t':int(i),'source':'catalog','sample_id':f'CAT_{int(i)}'})
    print('PAIRS',len(native),len(llm),'aux',len(aux),flush=True)
    lexcache={};phoncache={}
    def lex(target):
        if target not in lexcache:
            lexcache[target]=[int(i) for _,_,i in process.extract(cn[target],cn,scorer=fuzz.ratio,limit=64) if cn[i]!=cn[target]][:24]
        return lexcache[target]
    def phon(query,target):
        key=(query,cn[target])
        if key not in phoncache:
            q=normalize(romanize(query)).replace('eu','')
            phoncache[key]=[int(i) for _,_,i in process.extract(q,cp,scorer=fuzz.ratio,limit=64) if cn[i]!=cn[target]][:24]
        return phoncache[key]
    for number,row in enumerate(pairs):
        row['lex']=lex(row['t']);row['phon']=phon(row['q'],row['t']) if row['source']!='catalog' else row['lex']
        assert all(cn[i]!=cn[row['t']] for i in row['lex']+row['phon'])
        if (number+1)%1000==0:print('MINED',number+1,len(pairs),'seconds',round(time.time()-started),flush=True)
    def evaluation(frame,split):
        examples=[]
        for row in frame[frame.vessel_name.map(B.is_hangul)].drop_duplicates('target_vessel_id').to_dict('records'):
            name=str(row['vessel_name']);target=id2i[row['target_vessel_id']]
            for variant in ['orig','edit2','drop_first','drop_last','trunc','reorder','jamo_edit']:
                rng=np.random.default_rng(int(row['sample_id'].split('_')[-1])+97000);words=name.split();query=name
                if variant=='edit2':query=B._typo(name,2,rng)
                if variant=='drop_first' and len(words)>1:query=' '.join(words[1:])
                if variant=='drop_last' and len(words)>1:query=' '.join(words[:-1])
                if variant=='trunc':query=name[:max(3,int(.6*len(name)))]
                if variant=='reorder':query=' '.join(words[1:]+words[:1])
                if variant=='jamo_edit':
                    chars=list(unicodedata.normalize('NFKD',name));index=int(rng.integers(len(chars)));chars.pop(index);query=''.join(chars)
                candidates=list(dict.fromkeys([target]+lex(target)[:16]+phon(query,target)[:16]))
                examples.append({'sample_id':str(row['sample_id']),'q':query,'t':target,'candidates':candidates,'variant':variant,'split':split})
        return examples
    tune_rows=evaluation(tune,'tune4');outer_rows=evaluation(outer,'outer2')
    chars=set(chr(i) for i in range(0x1100,0x1200))|set(chr(i) for i in range(32,127))
    for row in catalog:
        chars.update(unicodedata.normalize('NFKD',row['name']).lower());chars.update(row['phon'])
    for row in pairs:
        chars.update(unicodedata.normalize('NFKD',row['q']).lower());chars.update(romanize(row['q']))
    config={'seed':97,'d_model':128,'layers':2,'heads':4,'feedforward':384,'dropout':.1,'max_length':160,'field_budgets':[64,24,40,28],
            'vocab':{c:i+4 for i,c in enumerate(sorted(chars))},'learning_rate':.0003,'weight_decay':.01,'steps':2000,'warmup_steps':100,'query_batch':24,'candidates_per_query':8,'eval_batch':256,'checkpoint_interval':500,
            'selection':'highest fold4 conditional name discrimination; earliest checkpoint on ties; no outer2 use',
            'loss':'multi-positive listwise logsumexp; identical normalized catalog names never false negatives; ambiguous same-prefix/suffix negatives masked for partial-name augmentation; partial loss weight0.2',
            'supervised_folds':[0,1,3],'tune_fold':4,'outer_holdout_fold':2,'source_mix':{'native':.4,'llm':.4,'catalog':.2}}
    files={'catalog.json':catalog,'pairs.json':pairs,'tune.json':tune_rows,'outer2.json':outer_rows}
    for name,obj in files.items():(DATA/name).write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':')))
    (PACKAGE/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    provenance={'allowed_inputs':{'train.csv':sha(ROOT/'data/open/train.csv'),'vessel_catalog.csv':sha(ROOT/'data/open/vessel_catalog.csv'),'train_only_llm_scored.csv':sha(genpath)},
                'split':'Fixed cv.folds(seed42); fit0/1/3, tune4, final outer2','supervised_fit_vessels':len(fitids),'tune_vessels':len(tuneids),'outer_vessels':len(outerids),
                'native_rows_all_retained':len(native),'llm_rows_all_after_fit_only_rescore_retained':len(llm),'llm_rescore_lexicon':'fit0/1/3 only, KO_DICT0; original full-data score overwritten before threshold .70',
                'catalog_selfsupervised_rows':len(aux),'catalog_selfsupervision_scope':'Allowed catalog English-name corruption only, no Korean aliases or train labels from tune/outer ships',
                'no_training_outer_korean_or_labels':True,'tune_queries':len(tune_rows),'outer_queries':len(outer_rows),'evaluation_truth_included_in_conditional_candidate_pool':True,
                'matched79_comparator':'External candidate JSONL supplied later; no claim from conditional hard-negative metric alone','data_sha256':{n:sha(DATA/n) for n in files},'seconds':time.time()-started}
    (DATA/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2),flush=True)

if __name__=='__main__':main()
