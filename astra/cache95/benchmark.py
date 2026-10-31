"""Reproduce coldship inner indexes; exact feature parity and bounded-cache benchmark."""
from pathlib import Path
import os,sys,json,time,hashlib,gc,cProfile,pstats,subprocess
os.environ['KO_DICT']='0';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import cv,opus_lib as B,astra_lib as A
from memo import InstanceMemo,FrozenQInfo

def rss_bytes():return int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024

def main():
    started=time.time();source=ROOT/'astra/coldship91/fold0';rows=json.loads((source/'queries.json').read_text());inner=json.loads((source/'inner_folds.json').read_text());tr,cat=cv.load();fit=tr[cv.folds(tr)!=0]
    llm=B.llm_korean_rows(fit,ROOT/'astra/ranker84_fold0/llm_fit.csv');selected=sorted(np.random.default_rng(95).choice(len(rows),100,replace=False).tolist());sample=[rows[i] for i in selected]
    plan=dict(query_indices=selected,query_sample='100 uniformly sampled actual training rows with seed95, then original order restored',training_fold=0,inner_parts=3,source_queries_sha256=hashlib.sha256((source/'queries.json').read_bytes()).hexdigest(),source_inner_sha256=hashlib.sha256((source/'inner_folds.json').read_bytes()).hexdigest(),score_cache_limit_per_instance=50000,translation_limit_per_instance=2048,minimum_speedup=1.3)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2));indexes=[];memos=[]
    for part in range(3):
        targets={k for k,v in inner.items() if v!=part};sub=fit[fit.target_vessel_id.isin(targets)];gen=llm[llm.target_vessel_id.isin(targets)]
        ix=A.Index(cat,sub);ix.ko=B.KA.KoMatcher(cat,pd.concat([sub,gen],ignore_index=True));indexes.append(ix);memos.append(InstanceMemo(ix.ko))
        print('INDEX',part,len(sub),len(gen),'seconds',round(time.time()-started),flush=True)
    def run_batch(batch,reference=None):
        outputs=[];mismatches=[];t=time.perf_counter()
        for k,row in enumerate(batch):
            part=inner[str(row['target_vessel_id'])];ix=indexes[part];cand=A.query_candidates(ix,row);idx,F=A.query_features(ix,row,cand)
            assert F.dtype==np.float32 and F.shape[1]==71
            if reference is not None:
                c0,i0,f0=reference[k]
                if cand!=c0 or idx!=i0 or not np.array_equal(F.view(np.uint32),f0.view(np.uint32)):mismatches.append(k)
            else:outputs.append((cand,idx,F))
        return time.perf_counter()-t,outputs,mismatches
    # Warm existing module-level pure normalization caches identically for both paths.
    run_batch(sample[:5]);gc.collect()
    profile=cProfile.Profile();profile.enable();run_batch(sample[:30]);profile.disable();profile.dump_stats(str(OUT/'baseline.prof'))
    stats=pstats.Stats(profile);hot=[]
    for (filename,line,name),(primitive,calls,total,cumulative,callers) in sorted(stats.stats.items(),key=lambda x:-x[1][3])[:30]:hot.append(dict(function=name,file=filename,line=line,calls=calls,self_seconds=total,cumulative_seconds=cumulative))
    (OUT/'profile.json').write_text(json.dumps(hot,indent=2))
    baseline,reference,_=run_batch(sample);rss_before=rss_bytes();print('BASELINE',baseline,'RSS',rss_before,flush=True)
    for memo in memos:memo.attach();memo.clear()
    cold,_,mismatches=run_batch(sample,reference);rss_after=rss_bytes();cold_stats=[memo.stats() for memo in memos];print('CACHED_COLD',cold,'MISMATCHES',mismatches,'RSS',rss_after,flush=True)
    warm,_,warm_mismatches=run_batch(sample,reference);warm_stats=[memo.stats() for memo in memos]
    # A copied result must not poison future cache reads; cached qinfo cannot mutate.
    row=sample[0];m=memos[inner[str(row['target_vessel_id'])]];info=m.matcher.translate_ko(row['vessel_name']);idx=reference[0][1][0];a=m.matcher._score_from_qinfo(info,idx);original=a[m.keys[0]];a[m.keys[0]]=12345.;assert m.matcher._score_from_qinfo(info,idx)[m.keys[0]]==original
    try:info['trans']='POISON';raise AssertionError('mutable qinfo')
    except TypeError:pass
    try:info._fingerprint=('POISON',);raise AssertionError('mutable fingerprint')
    except TypeError:pass
    for memo in memos:memo.detach(clear=True)
    # Reverse measurement order after caches have been warmed; caching starts empty.
    for memo in memos:memo.attach();memo.clear()
    cold_repeat,_,repeat_mismatches=run_batch(sample,reference)
    for memo in memos:memo.detach(clear=True)
    baseline_repeat,_,baseline_mismatches=run_batch(sample,reference)
    speedup=(baseline+baseline_repeat)/(cold+cold_repeat);calls=sum(s['scores']['hits']+s['scores']['misses'] for s in cold_stats);hits=sum(s['scores']['hits'] for s in cold_stats)
    report=dict(query_count=len(sample),unique_query_names=len({r['vessel_name'] for r in sample}),inner_counts={str(i):sum(inner[str(r['target_vessel_id'])]==i for r in sample) for i in range(3)},features=71,candidate_rows=sum(len(i) for c,i,f in reference),baseline_seconds=[baseline,baseline_repeat],cold_cached_seconds=[cold,cold_repeat],paired_speedup=speedup,warm_cached_seconds=warm,warm_speedup=baseline/warm,score_cache_hit_rate=hits/max(calls,1),cold_cache_stats=cold_stats,warm_cache_stats=warm_stats,rss_before_cache_bytes=rss_before,rss_after_cache_bytes=rss_after,rss_change_bytes=rss_after-rss_before,exact_float32_bit_mismatch_rows=mismatches+warm_mismatches+repeat_mismatches+baseline_mismatches,mutable_result_poisoning_test='PASS',immutable_qinfo_test='PASS',eligible=bool(speedup>=1.3 and not (mismatches+warm_mismatches+repeat_mismatches+baseline_mismatches)),seconds_total=time.time()-started,limits=['This is a 100-row training feature benchmark, not an end-to-end full-fold run.','Warm repeated-batch speed is diagnostic and is not the primary speedup.','Cache is safe only while each instance dictionary and catalog arrays remain fixed.','No running pipeline or submission file was modified.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
