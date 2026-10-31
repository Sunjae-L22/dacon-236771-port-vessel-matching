from pathlib import Path
import json,unicodedata,hashlib
import numpy as np
root=Path('/Users/sunjae/Desktop/항만');base=root/'astra/byt5_20261031';out=root/'astra/byt5_reuse_20261031';reports={}
canon=lambda s:' '.join(unicodedata.normalize('NFC',s).upper().split())
blen=lambda s:len(s.encode('utf-8'))+1
for fold in [2,4]:
 train=[json.loads(s) for s in (base/f'package/data/train_fold{fold}.jsonl').read_text().splitlines()]
 ratio=float(np.median([blen(canon(r['target']))/blen(unicodedata.normalize('NFC',r['name'])) for r in train]));ps=[json.loads(s) for s in (base/f'package/data/eval_fold{fold}.jsonl').read_text().splitlines()];rs=[json.loads(s) for s in (base/f'gpu_return_20261031/fold{fold}/eval/paired_rows.jsonl').read_text().splitlines()];result=[]
 for p,r in zip(ps,rs):
  assert p['query_id']==r['query_id'];pred=p['baseline'];vals=None
  if r['scores'] is not None:
   den=ratio*blen(unicodedata.normalize('NFC',p['name']));vals=[s*blen(canon(c['name']))/den for s,c in zip(r['scores'],p['candidates'])]
   k=min(range(len(vals)),key=lambda i:(-(p['candidates'][i]['score']+2*vals[i]),p['candidates'][i]['index']));pred=p['candidates'][k]['id']
  result.append(dict(query_id=p['query_id'],name=p['name'],vessel=p['vessel'],variant=p['variant'],baseline=p['baseline'],old_neural=r['pred'],pred=pred,target=p['target'],gap=p['gap']))
 def metric(z):
  resc=[r for r in z if r['baseline']!=r['target'] and r['pred']==r['target']];harm=[r for r in z if r['baseline']==r['target'] and r['pred']!=r['target']]
  return dict(n=len(z),rescue=len(resc),harm=len(harm),net=len(resc)-len(harm),rescue_vessels=len({r['vessel'] for r in resc}),harm_vessels=len({r['vessel'] for r in harm}),changed=sum(r['baseline']!=r['pred'] for r in z),outside=sum(r['baseline']!=r['pred'] and r['gap']>=4 for r in z))
 reports[f'fold{fold}']=dict(fit_median_target_source_byte_ratio=ratio,clean=metric([r for r in result if r['variant'].startswith('clean')]),noise=metric([r for r in result if not r['variant'].startswith('clean')]),by_variant={v:metric([r for r in result if r['variant']==v]) for v in sorted({r['variant'] for r in result})})
 (out/f'Rseq_fold{fold}_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result));print(fold,reports[f'fold{fold}'],flush=True)
(out/'Rseq_report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
