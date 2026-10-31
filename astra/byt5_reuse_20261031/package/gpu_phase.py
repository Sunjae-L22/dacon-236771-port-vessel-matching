"""Only loads previous weights locally. Never trains, downloads, or writes weights."""
import argparse,json,time,csv,hashlib
from pathlib import Path
from neural import setup,load_model,read_jsonl,save_json,score_names,eligible
from trie_search import CatalogTrie,search
HERE=Path(__file__).resolve().parent

def generate(fold,model_path,out):
    device=setup();model=load_model(model_path,device)
    records=list(csv.DictReader((HERE/'data/vessel_catalog.csv').open(encoding='utf-8-sig')))
    excluded=set(json.loads((HERE/f'data/excluded_fold{fold}.json').read_text()));trie=CatalogTrie(records,excluded)
    packets=read_jsonl(HERE/f'data/eval_fold{fold}.jsonl');names=list(dict.fromkeys(p['name'] for p in packets if p['gap']<4))
    out=Path(out);out.mkdir(parents=True,exist_ok=True);results={};start=time.time()
    for i,name in enumerate(names):
        results[name]=search(model,name,trie,device)
        if i==0 or (i+1)%25==0 or i==len(names)-1:
            elapsed=time.time()-start;print('TRIE',fold,i+1,len(names),'seconds',round(elapsed,1),'remaining_estimate',round(elapsed/(i+1)*(len(names)-i-1)),flush=True)
            save_json(out/'generated.json',results)
    # Explicit other-query-order check with the actual trained model.
    sample=names[:2];repeat={n:search(model,n,trie,device) for n in reversed(sample)}
    assert all(results[n]==repeat[n] for n in sample),'generation order dependence'
    save_json(out/'generation_report.json',dict(queries=len(names),catalog_names=len(trie.by_name),trie_nodes=trie.nodes,seconds=time.time()-start,device=device,beams=16,length_penalty=0.0,order_checks=len(sample),order_mismatches=0))

def score(fold,model_path,out):
    device=setup();model=load_model(model_path,device);out=Path(out)
    packets=read_jsonl(out/'expanded_packets.jsonl');old={p['query_id']:p for p in read_jsonl(HERE/f'data/old_scores_fold{fold}.jsonl')};cache={};start=time.time();max_diff=0.
    with (out/'scored_packets.jsonl').open('w') as f:
        for i,p in enumerate(packets):
            if p['gap']>=4:continue
            n=p['original_top100_n'];names=[c['name'] for c in p['candidates']]
            if not eligible(p['name'],names):
                f.write(json.dumps(dict(query_id=p['query_id'],scores=None,reason='overlength_fallback'))+'\n');f.flush();continue
            # Preserve the original100 microbatch shapes; score new names separately.
            def scores(ns):
                key=(p['name'],tuple(ns))
                if key not in cache:cache[key]=score_names(model,p['name'],ns,device)
                return cache[key]
            values=scores(names[:n])+(scores(names[n:]) if len(names)>n else [])
            previous=old[p['query_id']]['scores']
            if previous is not None:
                diff=max(abs(a-b) for a,b in zip(values[:n],previous));max_diff=max(max_diff,diff)
                if diff>.02:raise RuntimeError(f'previous checkpoint score drift {diff}')
            f.write(json.dumps(dict(query_id=p['query_id'],scores=values),ensure_ascii=False)+'\n');f.flush()
            if (i+1)%100==0:print('SCORE',fold,i+1,len(packets),round(time.time()-start),flush=True)
    save_json(out/'scoring_report.json',dict(seconds=time.time()-start,previous_score_max_abs_diff=max_diff,device=device,weights_reused=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['generate','score']);p.add_argument('--fold',type=int,required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args();(generate if a.phase=='generate' else score)(a.fold,a.model,a.out)
