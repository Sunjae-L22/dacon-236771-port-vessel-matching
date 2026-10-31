"""CPU plumbing tests with random tiny weights. Never performance evidence."""
import tempfile,json,sys,subprocess,zipfile,hashlib
from pathlib import Path
import numpy as np
from neural import *
from train import train
from build import build
from evaluate import evaluate

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);device=setup();assert device=='cpu','Smoke intentionally runs CPU'
    rows=[dict(name=k,target=e,vessel=v,source=s) for v,k,e in [('A','써니 아카시아','Sunny Acacia'),('B','스타 오션','Star Ocean')] for s in ['native','llm']]
    data=out/'toy_train.jsonl';data.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    train(data,out/'tiny','unused',steps=3,batch_size=4,tiny=True)
    model=load_model(out/'tiny/model','cpu');names=['Sunny Acacia','Star Ocean','Sea 123']
    scores=score_names(model,'써니 아카시아',names,'cpu');single=[score_names(model,'써니 아카시아',[n],'cpu')[0] for n in names]
    assert np.allclose(scores,single,atol=2e-6,rtol=0), (scores,single)
    # Independent direct forward, matching positive CE / EOS / padding mask.
    x=batch_tokens([source('써니 아카시아')],'cpu');y=batch_tokens([canonical(names[0])],'cpu',True)
    with torch.inference_mode():direct=-float(model(input_ids=x,attention_mask=x.ne(0),labels=y,decoder_attention_mask=y.ne(-100),use_cache=False).loss)
    assert abs(direct-scores[0])<2e-6
    assert not eligible('가'*100,names) and not eligible('가',['A'*96])
    p=dict(row_number=0,query_id='toy0',name='써니 아카시아',gap=.1,baseline='A',candidates=[dict(index=i,id=chr(65+i),name=n,score=1-i*.1) for i,n in enumerate(names)])
    assert decision(dict(p,gap=4),[0,100,0])=='A'
    assert decision(p,None)=='A'
    # Inference worker sees no target; adding/reordering other queries must not change results.
    q=dict(p,row_number=1,name='스타 오션')
    outs=[]
    for label,ps in [('forward',[p,q]),('reverse',[q,p]),('singleton',[p])]:
        inp=out/f'{label}.json';res=out/f'{label}_out.json';inp.write_text(json.dumps(ps))
        subprocess.run([sys.executable,str(Path(__file__).with_name('worker.py')),str(out/'tiny/model'),str(inp),str(res)],check=True)
        outs.append({r['row_number']:r['pred'] for r in json.loads(res.read_text())})
    assert outs[0]==outs[1] and outs[0][0]==outs[2][0]
    ev=out/'toy_eval.jsonl';ev.write_text('\n'.join(json.dumps(dict(r,target='A',vessel='A',variant='clean_keep' if i==0 else 'delete_missing')) for i,r in enumerate([p,q])))
    report=evaluate(ev,out/'tiny/model',out/'eval');assert report['all']['outside_scope']==0
    base=Path(__file__).with_name('baseline110.zip');result=build(base,out/'tiny/model',out/'tiny_plumbing_NOT_FOR_SUBMISSION.zip')
    checks=dict(status='PASS_TINY_CPU_ONLY',steps=3,conditional_scores_match_direct_CE=True,batched_singleton_logp_max_diff=float(np.max(np.abs(np.array(scores)-single))),worker_row_order_and_subset=True,overlength_fallback=True,easy_immutable=True,original_payloads_unchanged=len(result['original_payloads_unchanged']),torch=torch.__version__,transformers=__import__('transformers').__version__,performance_tested=False)
    save_json(out/'SMOKE_REPORT.json',checks);print(json.dumps(checks,indent=2))

if __name__=='__main__':main(sys.argv[1])
