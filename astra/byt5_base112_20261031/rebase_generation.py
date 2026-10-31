"""Rebase existing G4 generated names onto112. Does not generate, train, or submit."""
from pathlib import Path
import sys,json,hashlib,zipfile,time
import numpy as np,lightgbm as lgb
from paired_probe import P,ROOT,OUT,KoCat,kc_features,canon,read

def expand(fold,generated,dest,selected_ids=None):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);start=time.time()
    cat,ix,fx,_=P.context(fold);names=cat.vessel_name.astype(str).tolist();excluded=set(ix.cs_map.values())
    kc=KoCat(cat,OUT/'snapshot/catalog_ko.csv');model=lgb.Booster(model_file=str(OUT/f'snapshot/fold{fold}_kc.txt'))
    ps=read(OUT/f'fold{fold}/packets112.jsonl');queries={q['query_id']:q for q in json.loads((ROOT/f'astra/noise113/fold{fold}/eval/queries.json').read_text())}
    if selected_ids is not None:ps=[p for p in ps if p['query_id'] in selected_ids]
    cache={};rows=[];requests=[];checks=0;new_count=0
    for p in ps:
        row=dict(p,retrieval_pred=p['baseline'],beam_ids=[],new_candidates=0,cv_reference='112_recipe_proxy_one_kc_model')
        if p['eligible']:
            q=queries[p['query_id']];key=tuple(str(q.get(k)) for k in ['vessel_name','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year'])
            if key not in cache:
                cand=fx.add(P.B.query_candidates(ix,q),q)
                for rk,(i,_) in enumerate(kc.search(q['vessel_name'],50)):
                    e=cand.setdefault(int(i),[999,999,999]);e[0]=min(e[0],rk)
                def ranked(c):
                    idx,F=P.B.query_features(ix,q,c);scores=model.predict(np.hstack([F,kc_features(kc,q['vessel_name'],idx)]),num_threads=2)
                    allowed=[j for j,i in enumerate(idx) if i not in excluded] or list(range(len(idx)));order=sorted(allowed,key=lambda j:(-float(scores[j]),idx[j]))
                    return [dict(index=int(idx[j]),id=str(ix.ids[idx[j]]),name=names[idx[j]],score=float(scores[j])) for j in order]
                before=ranked(cand);beam=[];added=set()
                assert p['name'] in generated,'G4 generation missing query; do not silently regenerate or assume empty'
                for item in generated[p['name']]:
                    for vid in item['ids']:
                        i=ix.id2i[vid];assert i not in excluded and canon(names[i])==canon(item['name'])
                        beam.append(vid)
                        if i not in cand:cand[i]=[999,999,999];added.add(i)
                after=ranked(cand) if added else before
                cache[key]=(before,after,sorted(set(beam)),len(added))
            before,after,beam,added=cache[key]
            assert before[0]['id']==p['baseline']
            assert [c['id'] for c in before[:100]]==[c['id'] for c in p['candidates']]
            assert max(abs(a['score']-b['score']) for a,b in zip(before,p['candidates']))<1e-6
            amap={c['id']:c for c in after};pool=[amap[c['id']] for c in p['candidates']];seen={c['id'] for c in pool}
            for c in after[:100]+[amap[v] for v in beam]:
                if c['id'] not in seen:pool.append(c);seen.add(c['id'])
            row.update(retrieval_pred=after[0]['id'],beam_ids=beam,new_candidates=added,candidates=pool,original112_candidates=p['candidates'],original112_full_ids=[c['id'] for c in before],expanded_full_ids=[c['id'] for c in after])
            # Scoring request deliberately omits validation targets, priors, and outcomes.
            requests.append(dict(query_id=p['query_id'],name=p['name'],candidates=[dict(id=c['id'],name=c['name']) for c in pool]))
            checks+=1;new_count+=added
        rows.append(row)
    (dest/'expanded112.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));(dest/'score_requests112.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests))
    summary=dict(fold=fold,rows=len(rows),eligible_rows=checks,added_candidate_total=new_count,reference='112_recipe_proxy',kc_models=1,production112_kc_models=6,baseline_replay_mismatches=0,seconds=time.time()-start,new_training_steps=0,public_submissions=0)
    (dest/'rebase_report.json').write_text(json.dumps(summary,indent=2));return rows,summary

def import_and_rebase(path):
    path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest();dest=OUT/'generation_returns'/digest[:12];dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        status=json.loads(z.read('FINAL_STATUS.json'))
        assert status['status']=='REUSE_EVALUATION_COMPLETE_NO_SUBMISSION',status['status']
        (dest/'source_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
        summaries=[]
        for fold in [2,4]:
            gname=f'fold{fold}/generation_report.json';meta=json.loads(z.read(gname));assert meta['beams']==16 and meta['length_penalty']==0.0
            generated=json.loads(z.read(f'fold{fold}/generated.json'))
            _,summary=expand(fold,generated,dest/f'fold{fold}');summaries.append(summary)
    result=dict(source_zip_sha256=digest,active_baseline='112',stage='112_CANDIDATES_READY_NEURAL_RESCORING_REQUIRED',folds=summaries,L4_baseline_seconds_user_reported=710,L4_candidate_end_to_end_seconds=None,not_actual_production112_cv=True)
    (dest/'REBASE_STATUS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':import_and_rebase(sys.argv[1])
