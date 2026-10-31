"""CPU/GPU reference inference on matched-baseline candidate JSONL.

Each row: q, candidates (catalog integer indices), baseline_scores, t,
sample_id, variant. Weight selection is allowed only with --split tune4.
Apply a previously selected --weight on outer2; this script never searches
weights or checkpoints on outer2.
"""
import argparse,json
from pathlib import Path
import numpy as np
from model import load_checkpoint,score_pairs

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--split',choices=['tune4','outer2'],required=True);p.add_argument('--weight',type=float);p.add_argument('--top-k',type=int,default=100);a=p.parse_args()
    assert a.split=='tune4' or a.weight is not None,'Freeze fusion weight on tune4 before outer2 evaluation'
    model,tokenizer,saved=load_checkpoint(a.checkpoint,a.device);catalog=json.loads((a.data/'catalog.json').read_text());rows=[json.loads(line) for line in a.candidates.read_text().splitlines() if line.strip()]
    weights=[a.weight] if a.weight is not None else [0.,.1,.25,.5,1.]
    results=[];prepared=[];pairs=[];neural_scores=[]
    for row in rows:
        full_ids=row['candidates'];full_base=np.array(row['baseline_scores']);allowed=row.get('allowed',[True]*len(full_ids))
        assert len(full_ids)==len(full_base)==len(allowed) and len(full_ids)>0
        valid=[j for j,ok in enumerate(allowed) if ok] or list(range(len(full_ids)))
        ordered=sorted(valid,key=lambda j:(-float(full_base[j]),full_ids[j]));base_win=full_ids[ordered[0]]
        if 'baseline_pred' in row:assert row['baseline_pred']==base_win
        chosen=ordered[:a.top_k];ids=[full_ids[j] for j in chosen];base=full_base[chosen]
        assert base_win in ids
        prepared.append((row,ids,base,base_win));pairs.extend((row['q'],catalog[i]) for i in ids)
    all_nn=score_pairs(model,tokenizer,pairs,a.device,256);offset=0
    for row,ids,base,base_win in prepared:
        nn=np.array(all_nn[offset:offset+len(ids)]);offset+=len(ids)
        cached={key:row[key] for key in ['q','sample_id','variant','cohort','family','source_language','source'] if key in row}
        cached.update(query_id=row.get('query_id',row['sample_id']+'_'+row.get('variant','orig')),candidates=ids,nn_raw_scores=nn.tolist(),baseline_scores=base.tolist(),baseline_pred=base_win,target=row['t'])
        neural_scores.append(cached)
        bs=(base-base.mean())/max(float(base.std()),1e-6);ns=(nn-nn.mean())/max(float(nn.std()),1e-6)
        # Normalization is within one query's candidate set; never across queries.
        for w in weights:
            score=bs+w*ns;win=min(range(len(ids)),key=lambda j:(-float(score[j]),ids[j]));prediction=base_win if w==0 else ids[win]
            results.append({'sample_id':row['sample_id'],'query_id':row.get('query_id',row['sample_id']+'_'+row.get('variant','orig')),'variant':row.get('variant','orig'),'cohort':row.get('cohort','unspecified'),'weight':w,'pred':prediction,'baseline_pred':base_win,'target':row['t'],'correct':prediction==row['t']})
    report={'split':a.split,'checkpoint_step':saved['step'],'top_k':a.top_k,'fusion':'Per-query candidate-set z scores for both baseline and neural; baseline positive scale invariant; shortlist always includes full allowed baseline winner','metrics':{str(w):sum(r['correct'] for r in results if r['weight']==w)/sum(r['weight']==w for r in results) for w in weights},'predictions':results,'neural_score_cache_schema':1,'neural_scores':neural_scores}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['predictions','neural_scores']}))

if __name__=='__main__':main()
