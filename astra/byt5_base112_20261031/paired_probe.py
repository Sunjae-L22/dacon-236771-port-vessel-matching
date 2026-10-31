"""112-recipe held-out proxy. No full-fit112 weights are used for validation."""
from pathlib import Path
import sys,json,time,unicodedata
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astra/byt5_reuse_20261031/package'))
import baseline_phase as P
sys.path.insert(0,str(OUT/'snapshot'))
from kocat import KoCat,kc_features
import lightgbm as lgb

def canon(s):return ' '.join(unicodedata.normalize('NFC',str(s)).upper().split())
def blen(s):return len(s.encode('utf-8'))+1
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def metric(rows,pred):
    rescue=[r for r in rows if r['p112']!=r['target'] and r[pred]==r['target']]
    harm=[r for r in rows if r['p112']==r['target'] and r[pred]!=r['target']]
    return dict(n=len(rows),vessels=len({r['target'] for r in rows}),base112_correct=sum(r['p112']==r['target'] for r in rows),rescue=len(rescue),harm=len(harm),net=len(rescue)-len(harm),rescue_vessels=len({r['target'] for r in rescue}),harm_vessels=len({r['target'] for r in harm}),changed=sum(r[pred]!=r['p112'] for r in rows),already112_repaired_then_harmed=sum(r['p110']!=r['target'] and r['p112']==r['target'] and r[pred]!=r['target'] for r in rows),outside_dual_hard=sum(r[pred]!=r['p112'] and not r['dual_hard'] for r in rows))

def main(fold):
    start=time.time();dest=OUT/f'fold{fold}';dest.mkdir(exist_ok=True)
    cat,ix,fx,_=P.context(fold);excluded=set(ix.cs_map.values());names=cat.vessel_name.astype(str).tolist()
    kc=KoCat(cat,OUT/'snapshot/catalog_ko.csv');km=lgb.Booster(model_file=str(OUT/f'snapshot/fold{fold}_kc.txt'));assert km.num_feature()==62 and km.num_trees()==600
    assert km.feature_name()==P.B.FEATURES+['kc_sim','kc_margin']
    old=ROOT/'astra/byt5_20261031';qs=json.loads((ROOT/f'astra/noise113/fold{fold}/eval/queries.json').read_text());ps=read(old/f'package/data/eval_fold{fold}.jsonl');ns=read(old/f'gpu_return_20261031/fold{fold}/eval/paired_rows.jsonl')
    previous={r['query_id']:r['pred'] for r in read(ROOT/f'astra/byt5_reuse_20261031/Rseq_fold{fold}_rows.jsonl')}
    assert len(qs)==len(ps)==len(ns)==len(previous), 'Paired evaluation inputs must have identical coverage'
    train=read(old/f'package/data/train_fold{fold}.jsonl');ratio=float(np.median([blen(canon(r['target']))/blen(unicodedata.normalize('NFC',r['name'])) for r in train]))
    shots={p[0] for p in json.loads((OUT/'snapshot/fewshot_pairs.json').read_text())};cache={};rows=[];packets=[]
    for j,(q,p,n) in enumerate(zip(qs,ps,ns)):
        assert q['query_id']==p['query_id']==n['query_id'];assert p['baseline']==n['baseline']
        p112=p['baseline'];gap112=p['gap'];c112=p['candidates'];full_ids=[c['id'] for c in c112]
        if p['gap']<4:
            key=tuple(str(q.get(k)) for k in ['vessel_name','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year'])
            if key not in cache:
                cand=fx.add(P.B.query_candidates(ix,q),q)
                for rank,(i,_) in enumerate(kc.search(q['vessel_name'],50)):
                    e=cand.setdefault(int(i),[999,999,999]);e[0]=min(e[0],rank)
                idx,F=P.B.query_features(ix,q,cand);s=km.predict(np.hstack([F,kc_features(kc,q['vessel_name'],idx)]),num_threads=2)
                allowed=[k for k,i in enumerate(idx) if i not in excluded] or list(range(len(idx)));order=sorted(allowed,key=lambda k:(-float(s[k]),idx[k]))
                ranked=[dict(index=int(idx[k]),id=str(ix.ids[idx[k]]),name=names[idx[k]],score=float(s[k])) for k in order]
                cache[key]=ranked
            ranked=cache[key];p112=ranked[0]['id'];gap112=ranked[0]['score']-ranked[1]['score'] if len(ranked)>1 else 99.;c112=ranked[:100];full_ids=[c['id'] for c in ranked]
        dual=p['gap']<4 and gap112<4;covered=False;pred=p112;unguarded=p112
        if n['scores'] is not None:
            scoremap={c['id']:v for c,v in zip(p['candidates'],n['scores'])};covered=p112 in scoremap
            if covered:
                available=[c for c in c112 if c['id'] in scoremap];den=ratio*blen(unicodedata.normalize('NFC',p['name']))
                best=min(available,key=lambda c:(-(c['score']+2*scoremap[c['id']]*blen(canon(c['name']))/den),c['index']))
                unguarded=best['id'];pred=unguarded if dual else p112
        target=p['target'];primary=names[ix.id2i[target]] not in shots;prev=previous[p['query_id']]
        rows.append(dict(query_id=p['query_id'],name=p['name'],variant=p['variant'],target=target,p110=p['baseline'],p112=p112,p112_byt5=pred,p112_byt5_oldgate=unguarded,previous110_byt5=prev,old_proposal_overlay_on112=prev if prev!=p['baseline'] else p112,gap110=p['gap'],gap112=gap112,dual_hard=dual,nn_base112_covered=covered,primary=primary,target_in112_full=target in full_ids,target_in112_top100=any(c['id']==target for c in c112)))
        packets.append(dict(query_id=p['query_id'],name=p['name'],variant=p['variant'],vessel=target,target=target,baseline=p112,gap=gap112,gate110_gap=p['gap'],eligible=dual,primary=primary,candidates=c112))
        if (j+1)%200==0:print('BASE112',fold,j+1,len(ps),round(time.time()-start),flush=True)
    primary=[r for r in rows if r['primary']];report=dict(fold=fold,reference='112_recipe_proxy',production112_kc_models=6,validation_kc_models=1,validation110_gate_models=6 if fold==2 else 1,not_deployed112_predictions=True,seconds=time.time()-start,excluded_fewshot_rows=len(rows)-len(primary),rows=len(primary),methods={})
    for key in ['p112_byt5','p112_byt5_oldgate','old_proposal_overlay_on112']:
        report['methods'][key]={cohort:metric([r for r in primary if cohort=='all' or r['variant'].startswith('clean')==(cohort=='clean')],key) for cohort in ['all','clean','noise']}
    report['overlap']=dict(base112_repairs110=sum(r['p110']!=r['target'] and r['p112']==r['target'] for r in primary),base112_harms110=sum(r['p110']==r['target'] and r['p112']!=r['target'] for r in primary),old_byt5_repairs110_already_fixed112=sum(r['p110']!=r['target'] and r['previous110_byt5']==r['target'] and r['p112']==r['target'] for r in primary),old_byt5_repairs110_not_fixed112=sum(r['p110']!=r['target'] and r['previous110_byt5']==r['target'] and r['p112']!=r['target'] for r in primary),eligible_dual_hard=sum(r['dual_hard'] for r in primary),dual_hard_base112_nn_covered=sum(r['dual_hard'] and r['nn_base112_covered'] for r in primary),missing112_full_hard=sum(r['dual_hard'] and not r['target_in112_full'] for r in primary))
    report['remaining112_errors']={cohort:dict(total=sum(r['p112']!=r['target'] for r in primary if r['variant'].startswith('clean')==(cohort=='clean')),dual_hard=sum(r['p112']!=r['target'] and r['dual_hard'] for r in primary if r['variant'].startswith('clean')==(cohort=='clean'))) for cohort in ['clean','noise']}
    (dest/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));(dest/'packets112.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in packets));(dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main(int(sys.argv[1]))
