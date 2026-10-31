"""Training reads native/LLM train pairs and tune fold4 only; never outer2.json."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time
import unicodedata
import torch
from model import CrossEncoder,Tokenizer,normalize,score_pairs

class VesselCycle:
    def __init__(self, rows, rng):
        self.rows=rows;self.rng=rng;self.by=defaultdict(list)
        for i,row in enumerate(rows):self.by[row['t']].append(i)
        self.keys=list(self.by);self.queue=[];self.positions=defaultdict(int);self.seen=set();self.seen_targets=set()
        for group in self.by.values():rng.shuffle(group)
    def take(self):
        if not self.queue:self.queue=self.keys.copy();self.rng.shuffle(self.queue)
        target=self.queue.pop();group=self.by[target];position=self.positions[target];self.positions[target]+=1
        index=group[position%len(group)];self.seen.add(index);self.seen_targets.add(target)
        return self.rows[index]

def corrupt(query,rng):
    draw=rng.random();words=query.split()
    if draw<.3:return query,'orig',1.
    if .5<=draw<.6:
        chars=list(unicodedata.normalize('NFKD',query))
        def jamo_range(c):
            n=ord(c)
            for low,high in [(0x1100,0x1112),(0x1161,0x1175),(0x11A8,0x11C2)]:
                if low<=n<=high:return low,high
            return None
        positions=[i for i,c in enumerate(chars) if jamo_range(c)]
        if positions:
            for _ in range(rng.choice([1,2])):
                positions=[i for i,c in enumerate(chars) if jamo_range(c)]
                if not positions:break
                index=rng.choice(positions);low,high=jamo_range(chars[index]);op=rng.randrange(3)
                if op==0:chars.pop(index)
                elif op==1:chars[index]=chr(rng.randint(low,high))
                else:chars.insert(index,chr(rng.randint(low,high)))
            return ''.join(chars),'jamo_edit',.7
    if draw<.6:
        chars=list(query)
        for _ in range(rng.choice([1,2])):
            if len(chars)<3:break
            i=rng.randrange(len(chars));op=rng.randrange(3)
            if op==0:chars.pop(i)
            elif op==1:chars.insert(i,chars[i])
            elif i+1<len(chars):chars[i],chars[i+1]=chars[i+1],chars[i]
        return ''.join(chars),'edit',1.
    if draw<.7 and len(words)>1:return ' '.join(words[1:]),'drop_first',.2
    if draw<.8 and len(words)>1:return ' '.join(words[:-1]),'drop_last',.2
    if draw<.9 and len(words)>1:return ' '.join(words[1:]+words[:1]),'reorder',.7
    return query[:max(3,int(len(query)*rng.uniform(.6,.9)))],'truncate',.2

def group(row,catalog,rng,group_size):
    target=row['t'];positive=catalog[target];q,kind,weight=corrupt(row['q'],rng)
    proposed=[target]
    for key,n in [('lex',3),('phon',3)]:
        options=row.get(key,[])
        if options:proposed.extend(rng.sample(options,min(n,len(options))))
    unique=list(dict.fromkeys(proposed))
    while len(unique)<group_size:
        i=rng.randrange(len(catalog))
        if i not in unique:unique.append(i)
    unique=unique[:group_size];rng.shuffle(unique)
    labels=[catalog[i]['norm']==positive['norm'] for i in unique]
    valid=[True]*len(unique);tokens=positive['name'].lower().split()
    # Partial names cannot identify a particular same-prefix/suffix ship. Mask
    # those negatives and weakly weight all truncated/deleted-word examples.
    for j,i in enumerate(unique):
        if labels[j] or not tokens:continue
        other=catalog[i]['name'].lower().split()
        if not other:continue
        ambiguous=(kind in ('truncate','drop_last') and tokens[0]==other[0]) or (kind=='drop_first' and tokens[-1]==other[-1])
        if ambiguous:valid[j]=False
    assert any(labels)
    return [(q,catalog[i]) for i in unique],labels,valid,weight

def tune(model,tokenizer,rows,catalog,device,batch_size):
    correct=0;total=0;losses=[];by=defaultdict(lambda:[0,0])
    pairs=[(row['q'],catalog[i]) for row in rows for i in row['candidates']]
    all_scores=score_pairs(model,tokenizer,pairs,device,batch_size);offset=0
    for row in rows:
        values=all_scores[offset:offset+len(row['candidates'])];offset+=len(row['candidates'])
        winner=max(range(len(values)),key=lambda j:(values[j],-row['candidates'][j]))
        ok=catalog[row['candidates'][winner]]['norm']==catalog[row['t']]['norm']
        correct+=int(ok);total+=1;by[row['variant']][0]+=int(ok);by[row['variant']][1]+=1
    return {'conditional_name_accuracy':correct/max(total,1),'rows':total,'by_variant':{k:a/b for k,(a,b) in by.items()},'caveat':'True candidate is included in the hard-negative evaluation set; this is conditional name discrimination, not end-to-end vessel accuracy.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--config',type=Path,default=Path('config.json'));p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    config=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=True)
    if a.device=='cuda':assert torch.cuda.is_available(),'CUDA unavailable; refusing an accidental long CPU run'
    if a.device=='cpu' and not a.smoke:raise RuntimeError('Long CPU training disabled; --smoke only')
    torch.set_num_threads(2);torch.manual_seed(config['seed']);rng=random.Random(config['seed'])
    if hasattr(torch.backends,'mha'):torch.backends.mha.set_fastpath_enabled(False)
    catalog=json.loads((a.data/'catalog.json').read_text());data=json.loads((a.data/'pairs.json').read_text());tuning=json.loads((a.data/'tune.json').read_text())
    cycles={source:VesselCycle([r for r in data if r['source']==source],rng) for source in ['native','llm','catalog']}
    assert all(c.rows for c in cycles.values())
    model=CrossEncoder(config).to(a.device);tokenizer=Tokenizer(config)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    steps=a.steps or config['steps'];interval=config['checkpoint_interval'];batch=config['query_batch'];groups=config['candidates_per_query']
    if a.smoke:steps=2;interval=1;batch=3;groups=4;tuning=tuning[:4]
    recipe=dict(config);recipe.pop('vocab');recipe.update(parameters=sum(v.numel() for v in model.parameters()),device=a.device,torch_version=torch.__version__,data_provenance=json.loads((a.data/'provenance.json').read_text()),config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest())
    (a.out/'run_config.json').write_text(json.dumps(recipe,indent=2));history=[];best=-1.;start=time.time()
    try:
        for step in range(1,steps+1):
            model.train();pairs=[];labels=[];valid=[];weights=[]
            schedule=['native','llm','native','llm','catalog']
            for i in range(batch):
                row=cycles[schedule[(i+step)%len(schedule)]].take();pr,la,va,we=group(row,catalog,rng,groups)
                pairs.extend(pr);labels.append(la);valid.append(va);weights.append(we)
            ids,segments=tokenizer.batch(pairs,a.device);pos=torch.tensor(labels,device=a.device);allowed=torch.tensor(valid,device=a.device);weight=torch.tensor(weights,device=a.device)
            opt.zero_grad(set_to_none=True)
            lr=config['learning_rate']*min(step/config['warmup_steps'],1.)*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
            for group_opt in opt.param_groups:group_opt['lr']=lr
            context=torch.autocast('cuda',dtype=torch.bfloat16) if a.device=='cuda' else nullcontext()
            with context:
                scores=model(ids,segments).reshape(batch,groups).float()
                loss_row=scores.masked_fill(~allowed,float('-inf')).logsumexp(1)-scores.masked_fill(~pos,float('-inf')).logsumexp(1)
                loss=(weight*loss_row).sum()/weight.sum()
            assert torch.isfinite(loss),loss
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%50==0 or a.smoke:print(json.dumps({'step':step,'loss':float(loss.detach()),'seconds':round(time.time()-start,1)}),flush=True)
            if step%interval==0 or step==steps:
                metrics=tune(model,tokenizer,tuning,catalog,a.device,config['eval_batch']);metrics.update(step=step,seconds=time.time()-start)
                state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':config,'step':step,'tune':metrics}
                torch.save(state,a.out/f'step_{step:04d}.pt')
                if metrics['conditional_name_accuracy']>best:
                    best=metrics['conditional_name_accuracy'];torch.save(state,a.out/'best.pt')
                history.append(metrics);(a.out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(metrics),flush=True)
        if a.smoke:
            testpairs=[(cycles['native'].rows[0]['q'],catalog[cycles['native'].rows[0]['t']]),(cycles['llm'].rows[0]['q'],catalog[cycles['llm'].rows[0]['t']])]
            together=score_pairs(model,tokenizer,testpairs,'cpu',2);separate=score_pairs(model,tokenizer,testpairs,'cpu',1)
            error=max(abs(a-b) for a,b in zip(together,separate));assert error<1e-5,error
        else:error=None
        report={'status':'COMPLETE','steps':steps,'seconds':time.time()-start,'best_tune':best,'outer2_read':False,'batch_reference_max_error':error,'source_coverage':{s:{'rows_available':len(c.rows),'rows_seen':len(c.seen),'vessels_available':len(c.keys),'vessels_seen':len(c.seen_targets)} for s,c in cycles.items()}}
        (a.out/'complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    except BaseException as error:
        (a.out/'failure.json').write_text(json.dumps({'error':repr(error),'seconds':time.time()-start},indent=2));raise

if __name__=='__main__':main()
