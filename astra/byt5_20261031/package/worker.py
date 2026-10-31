"""Subprocess-only neural inference. Input schema has no ground truth."""
import sys,json,time
from pathlib import Path
from neural import setup,load_model,score_names,eligible,decision

def run(model_path,inputs,outputs):
    device=setup();model=load_model(model_path,device);packets=json.loads(Path(inputs).read_text());results=[];started=time.time()
    for p in packets:
        assert 'target' not in p and 'vessel' not in p
        names=[c['name'] for c in p['candidates']];ok=p['gap']<4 and eligible(p['name'],names)
        scores=score_names(model,p['name'],names,device) if ok else None
        results.append(dict(row_number=p['row_number'],baseline=p['baseline'],pred=decision(p,scores),eligible=ok,gap=p['gap']))
    Path(outputs).write_text(json.dumps(results,ensure_ascii=False));print('NEURAL',len(results),'seconds',round(time.time()-started,2),flush=True)

if __name__=='__main__':run(*sys.argv[1:])
