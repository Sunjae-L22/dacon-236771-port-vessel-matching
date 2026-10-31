import json,sys,unicodedata
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def canon(s):return ' '.join(unicodedata.normalize('NFC',s).upper().split())
def length(s):return len(s.encode('utf-8'))+1

def metric(rows,key):
    resc=[r for r in rows if r['baseline']!=r['target'] and r[key]==r['target']];harm=[r for r in rows if r['baseline']==r['target'] and r[key]!=r['target']]
    err=sum(r['baseline']!=r['target'] for r in rows)
    return dict(n=len(rows),vessels=len({r['vessel'] for r in rows}),base_errors=err,rescue=len(resc),harm=len(harm),net=len(resc)-len(harm),error_reduction=(len(resc)-len(harm))/max(1,err),rescue_vessels=len({r['vessel'] for r in resc}),harm_vessels=len({r['vessel'] for r in harm}),changed=sum(r[key]!=r['baseline'] for r in rows),outside_scope=sum(r[key]!=r['baseline'] and r['gap']>=4 for r in rows))

def evaluate(fold,out):
    out=Path(out);ps=read(out/'expanded_packets.jsonl');scores={r['query_id']:r['scores'] for r in read(out/'scored_packets.jsonl')};original={r['query_id']:r for r in read(HERE/f'data/eval_fold{fold}.jsonl')}
    train=read(HERE/f'data/train_fold{fold}.jsonl');ratio=float(np.median([length(canon(r['target']))/length(unicodedata.normalize('NFC',r['name'])) for r in train]));rows=[]
    for p in ps:
        old=p['baseline'];q=original[p['query_id']];r={k:p[k] for k in ['query_id','name','variant','vessel','target','baseline','gap']}
        r.update(retrieval_only=p['retrieval_pred'],top100_seq=old,expanded_seq=old,expanded_seq_global_guard=old,new_candidates=p['new_candidates'],target_in_original_full=p['target'] in p['original_full_ids'] if p['gap']<4 else None,target_in_expanded_full=p['target'] in p['expanded_full_ids'] if p['gap']<4 else None,target_in_beam=p['target'] in p['beam_ids'],old_in_beam=old in p['beam_ids'])
        if p['gap']<4 and scores[p['query_id']] is not None:
            cs=p['candidates'];values=scores[p['query_id']];assert len(cs)==len(values)
            denominator=ratio*length(unicodedata.normalize('NFC',p['name']));nn=[v*length(canon(c['name']))/denominator for v,c in zip(values,cs)]
            j=min(range(len(cs)),key=lambda i:(-(cs[i]['score']+2*nn[i]),cs[i]['index']));pred=cs[j]['id'];r['expanded_seq']=pred
            # Registered ambiguity safeguard: keep110 when its name is also in global beam.
            r['expanded_seq_global_guard']=pred if pred in p['beam_ids'] and old not in p['beam_ids'] else old
            n=p['original_top100_n'];assert [c['id'] for c in cs[:n]]==[c['id'] for c in q['candidates']]
            j=min(range(n),key=lambda i:(-(q['candidates'][i]['score']+2*nn[i]),q['candidates'][i]['index']));r['top100_seq']=cs[j]['id']
        rows.append(r)
    methods={}
    for method in ['top100_seq','retrieval_only','expanded_seq','expanded_seq_global_guard']:
        m={group:metric([r for r in rows if group=='all' or r['variant'].startswith('clean')==(group=='clean')],method) for group in ['all','clean','noise']}
        m['clean_gate']=m['clean']['harm']==0 and m['all']['outside_scope']==0;m['large_gain_scale_diagnostic']=m['noise']['error_reduction']>=.54;methods[method]=m
    hard=[r for r in rows if r['gap']<4];missing=[r for r in hard if not r['target_in_original_full']]
    report=dict(fold=fold,fit_length_ratio=ratio,methods=methods,original_missing_hard=len(missing),newly_recalled=sum(r['target_in_expanded_full'] for r in missing),newly_recalled_vessels=len({r['vessel'] for r in missing if r['target_in_expanded_full']}),hard=len(hard),generated_candidates=sum(r['new_candidates'] for r in hard),status='DEVELOPMENT_EVAL_ONLY_NO_SUBMISSION',warning='Reused development folds; changing the score formula was motivated by these folds. No independent confirmation, full training, candidate build, or Public forecast.')
    (out/'paired_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2));return report

if __name__=='__main__':evaluate(int(sys.argv[1]),sys.argv[2])
