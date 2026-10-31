import argparse,json,time,math,hashlib
from pathlib import Path
from neural import *

def train(data,out,pretrained,steps=1500,batch_size=32,tiny=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('completed training exists; choose a new output directory')
    device=setup();rows=read_jsonl(data);sampler=Sampler(rows)
    if not tiny and device!='cuda':raise RuntimeError('Real training requires CUDA; CPU only allowed for --tiny smoke')
    if tiny:model=T5ForConditionalGeneration(T5Config(vocab_size=384,d_model=32,d_ff=64,num_layers=1,num_decoder_layers=1,num_heads=2,d_kv=16,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1,dropout_rate=0.0))
    else:model=load_model(pretrained,device,local=True)
    model.to(device);model.config.use_cache=False;model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    started=time.time();losses=[]
    for step in range(1,steps+1):
        # Fixed final step; evaluation is never consulted by the optimizer/schedule.
        lr=1e-4*min(step/100.,1.)*.5*(1+math.cos(math.pi*max(0,step-100)/max(1,steps-100))) if not tiny else 1e-3
        for g in optimizer.param_groups:g['lr']=lr
        samples=sampler.batch(batch_size);x=batch_tokens([r[0] for r in samples],device);y=batch_tokens([r[1] for r in samples],device,True)
        optimizer.zero_grad(set_to_none=True)
        with amp(device):loss=model(input_ids=x,attention_mask=x.ne(0),labels=y,decoder_attention_mask=y.ne(-100),use_cache=False).loss
        if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(float(loss.detach()))
        if step==1 or step%50==0 or step==steps:
            msg=dict(step=step,steps=steps,loss=sum(losses[-50:])/len(losses[-50:]),seconds=time.time()-started,lr=lr)
            print(json.dumps(msg),flush=True)
            with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(msg)+'\n')
    model.eval();model.save_pretrained(out/'model',safe_serialization=True)
    report=dict(status='COMPLETE',steps=steps,batch_size=batch_size,seconds=time.time()-started,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,torch=torch.__version__,seed=SEED,train_file_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),train_pairs=len(rows),dropped_overlength=sampler.dropped,last_loss=losses[-1],smoke_only=tiny)
    save_json(out/'COMPLETE.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--pretrained',required=True);p.add_argument('--steps',type=int,default=1500);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--tiny',action='store_true');a=p.parse_args();train(a.data,a.out,a.pretrained,a.steps,a.batch_size,a.tiny)
