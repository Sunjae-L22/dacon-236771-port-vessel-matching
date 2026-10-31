import argparse,json,time
from pathlib import Path
from neural import *

def metric(rows):
    return dict(n=len(rows),vessels=len({r['vessel'] for r in rows}),base_correct=sum(r['baseline']==r['target'] for r in rows),new_correct=sum(r['pred']==r['target'] for r in rows),rescue=sum(r['baseline']!=r['target'] and r['pred']==r['target'] for r in rows),harm=sum(r['baseline']==r['target'] and r['pred']!=r['target'] for r in rows),changed=sum(r['baseline']!=r['pred'] for r in rows),outside_scope=sum(r['baseline']!=r['pred'] and not r['eligible'] for r in rows),top100_oracle_rescue=sum(r['baseline']!=r['target'] and r['target_in_top100'] for r in rows))

def evaluate(data,model_path,out):
    device=setup();model=load_model(model_path,device);packets=read_jsonl(data);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];cache={};started=time.time()
    with (out/'paired_rows.jsonl').open('w') as f:
        for i,p in enumerate(packets):
            names=[c['name'] for c in p['candidates']];ok=p['gap']<4 and eligible(p['name'],names);scores=None
            if ok:
                key=(p['name'],tuple(names))
                if key not in cache:cache[key]=score_names(model,p['name'],names,device)
                scores=cache[key]
            pred=decision(p,scores);r={k:p[k] for k in ['query_id','name','variant','vessel','target','baseline','gap']}
            r.update(pred=pred,eligible=ok,target_in_top100=any(c['id']==p['target'] for c in p['candidates']),scores=scores,candidate_ids=[c['id'] for c in p['candidates']]);rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
            if (i+1)%100==0:print('EVAL',i+1,len(packets),round(time.time()-started),flush=True)
    clean=metric([r for r in rows if r['variant'].startswith('clean')]);noise=metric([r for r in rows if not r['variant'].startswith('clean')]);allm=metric(rows)
    reduction=(noise['rescue']-noise['harm'])/max(1,noise['n']-noise['base_correct'])
    report=dict(all=allm,clean=clean,noise=noise,noise_error_reduction=reduction,minimum_gate=clean['harm']==0 and allm['outside_scope']==0 and reduction>=.10,large_gain_diagnostic=reduction>=.54,seconds=time.time()-started,device=device,warning='Development folds and synthetic noise do not establish Public gains. Full QA build is unreviewed, never auto-submit.',by_variant={v:metric([r for r in rows if r['variant']==v]) for v in sorted({r['variant'] for r in rows})})
    save_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args();evaluate(a.data,a.model,a.out)
