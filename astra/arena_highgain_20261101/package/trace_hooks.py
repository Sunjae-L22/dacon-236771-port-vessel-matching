"""Record unchanged C_vote component outputs and v20 candidates. No target input."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.util
sys.dont_write_bytecode=True

def digest_row(row):
    import math
    normalized={str(k):('' if v is None or (isinstance(v,float) and math.isnan(v)) else str(v)) for k,v in row.items() if k not in {'target_vessel_id','truth'}}
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def load_entry(script):
    script=Path(script).resolve();sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location('entry',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def attach_neural_trace(trace_path,plugin=None,receipts=None):
    import numpy as np
    from src.encoder_views import EncoderViewsEnsemble
    original=EncoderViewsEnsemble.score_records
    def traced(self,rows,signals,blocks):
        old=self.baseline.score_groups;captured=[];view_calls=[];view_originals=[]
        def baseline_scores(bs):
            result=old(bs);captured.extend([np.asarray(x).copy() for x in result]);return result
        self.baseline.score_groups=baseline_scores
        for scorer,weight in self.views:
            method=scorer.score;view_originals.append((scorer,method))
            def score(row,candidates,beams,method=method,name=scorer.name):
                t=time.perf_counter();out=method(row,candidates,beams)
                view_calls.append(dict(sample_id=str(row.get('sample_id','')),view=name,candidate_ids=[str(c['vessel_id']) for c in candidates],scores=np.asarray(out).tolist(),seconds=time.perf_counter()-t))
                return out
            scorer.score=score
        try:final=original(self,rows,signals,blocks)
        finally:
            self.baseline.score_groups=old
            for scorer,method in view_originals:scorer.score=method
        if len(captured)!=len(rows) or len(final)!=len(rows):raise RuntimeError('Incomplete baseline trace')
        with Path(trace_path).open('a',encoding='utf-8') as f:
            for row,sig,base,local in zip(rows,signals,captured,final):
                sid=str(row.get('sample_id',''));ids=[str(self.r.ids[int(i)]) for i in sig['cand']]
                keep=np.argsort(-base,kind='stable')[:self.top_k];z=base[keep];p=z/z.sum()
                record=dict(sample_id=sid,input_fingerprint=digest_row(row),candidate_ids=ids,base_prob=base.tolist(),final_scores=np.asarray(local).tolist(),top20_ids=[ids[i] for i in keep],beams=list(sig.get('llm_beams') or []),neural_skipped=(len(keep)<2 or float(p[0]-p[1])>self.weight/(1-self.weight)+1e-6),views=[v for v in view_calls if v['sample_id']==sid])
                if plugin is not None:record['experiment']=plugin.evaluate(row,sig,base,np.asarray(local),self,receipts)
                f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        return final
    EncoderViewsEnsemble.score_records=traced

def run_child(script,test,catalog,output):
    import inspect
    stage=Path(os.environ['ASTRA_TRACE_DIR']);stage.mkdir(parents=True,exist_ok=True)
    pkg=Path(script).resolve().parent.name;t=time.time()
    if pkg=='ko':
        sys.path.insert(0,str(Path(script).resolve().parent/'model'))
        attach_neural_trace(stage/'v20_candidates.jsonl')
    m=load_entry(script);ps=inspect.signature(m.run).parameters
    restore=lambda:None
    if pkg=='hs':restore=attach_hsw_trace(stage/'hs_candidates.jsonl')
    elif pkg=='en':restore=attach_112_trace(script,stage/'en_candidates.jsonl')
    key='output_path' if 'output_path' in ps else 'out'
    try:m.run(test_path=test,catalog_path=catalog,**{key:output})
    finally:restore()
    (stage/f'{pkg}_timing.json').write_text(json.dumps(dict(start=t,end=time.time(),elapsed=time.time()-t,pid=os.getpid())))

def attach_hsw_trace(trace_path):
    import numpy as np
    import pipeline
    old=pipeline.score_candidates
    def score(bundle,df,cat,*args,**kwargs):
        cands,scores=old(bundle,df,cat,*args,**kwargs)
        gaps=pipeline.tonnage_gaps(df,cat,cands) if pipeline.TON_GATE is not None else None
        allowed=pipeline.allowed(bundle,cands,cat,gaps=gaps)
        with Path(trace_path).open('a',encoding='utf-8') as f:
            for row,ids,raw,mask in zip(df.to_dict('records'),cands,scores,allowed):
                order=sorted([i for i in range(len(ids)) if mask[i]],key=lambda i:(-float(raw[i]),i))
                record=dict(sample_id=str(row['sample_id']),input_fingerprint=digest_row(row),candidate_ids=[str(cat.ids[int(i)]) for i in ids],scores=np.asarray(raw).tolist(),allowed=np.asarray(mask).tolist(),top20_ids=[str(cat.ids[int(ids[i])]) for i in order[:20]])
                f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        return cands,scores
    pipeline.score_candidates=score
    return lambda:setattr(pipeline,'score_candidates',old)

def attach_112_trace(script,trace_path):
    import ast,unicodedata
    script=str(Path(script).resolve());tree=ast.parse(Path(script).read_text());sites={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            call=node.value
            if isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='preds' and call.func.attr=='append':
                sites[node.lineno]='callsign' if 'ix.ids[cs]' in ast.unparse(call) else 'ranked'
    if len(sites)!=2:raise RuntimeError('112 append sites changed; audit trace before use')
    old=sys.gettrace()
    def observe(frame,event,arg):
        if frame.f_code.co_filename!=script:return None
        if event=='line' and frame.f_lineno in sites:
            v=frame.f_locals;row=v['row'];name=unicodedata.normalize('NFC',str(row.get('vessel_name','')))
            if not any('가'<=c<='힣' for c in name):return observe
            record=dict(sample_id=str(row['sample_id']),input_fingerprint=digest_row(row),kind=sites[frame.f_lineno])
            if record['kind']=='callsign':record.update(candidate_ids=[str(v['ix'].ids[v['cs']])],scores=[99.],top20_ids=[str(v['ix'].ids[v['cs']])],gap110=99.)
            else:
                ids=v['idx'];score=v['score'];allowed=v['allowed'];ix=v['ix']
                record.update(candidate_ids=[str(ix.ids[int(i)]) for i in ids],scores=[float(x) for x in score],top20_ids=[str(ix.ids[int(ids[j])]) for j in allowed[:20]],gap110=float(v['row_gap']),score_family='kc112' if v['use_kc'] else '110')
            with Path(trace_path).open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        return observe
    sys.settrace(observe)
    return lambda:sys.settrace(old)
