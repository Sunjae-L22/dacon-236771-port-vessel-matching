"""Post-hoc evaluation only. Never import into the inference path."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
from collections import Counter
import neural_rescue as nr


def main():
    ap=argparse.ArgumentParser()
    for name in ('packets','results','truth','output'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--status',type=Path)
    a=ap.parse_args()
    status_path=a.status or a.results.with_suffix('.status.json')
    status=json.loads(status_path.read_text())
    if status.get('state')!='COMPLETE': raise ValueError('Only COMPLETE results may be diagnosed')
    for key,path in [('packets_sha256',a.packets),('results_sha256',a.results)]:
        if status.get(key)!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Sealed artifact hash mismatch: '+key)
    packet_rows=[json.loads(line) for line in a.packets.read_text().splitlines() if line.strip()]
    result_rows=[json.loads(line) for line in a.results.read_text().splitlines() if line.strip()]
    for name,rows in [('packet',packet_rows),('result',result_rows)]:
        if len({r['sample_id'] for r in rows})!=len(rows): raise ValueError('Duplicate '+name+' sample IDs')
    if status.get('rows')!=len(packet_rows): raise ValueError('Status row count mismatch')
    p={r['sample_id']:r for r in packet_rows}
    pred={r['sample_id']:r for r in result_rows}
    with a.truth.open(encoding='utf-8-sig') as f: labels=list(csv.DictReader(f))
    if len({r['sample_id'] for r in labels})!=len(labels): raise ValueError('Duplicate labels')
    if set(p)!=set(pred) or set(p)!={r['sample_id'] for r in labels}: raise ValueError('Exact cohort join required')
    # Reuse the inference contract before any label-dependent calculation.
    # Zero-mass/corrupt vectors cannot be scored or diagnosed as normal outputs.
    for sid,packet in p.items(): nr.verify_result(packet,pred[sid])
    counts={}; detail=[]
    for r in labels:
        sid,y=r['sample_id'],r['target_vessel_id']; x,z=p[sid],pred[sid]
        if y in x['excluded']: raise ValueError('Invalid cohort: truth is a production-excluded vessel')
        order=sorted(range(len(x['ids'])),key=lambda i:-x['base'][i])
        top=order[:20]; mass=sum(x['base'][i] for i in top)
        if mass <= 0: raise ValueError('Invalid baseline top20 probability mass')
        skip=len(top)<2 or (x['base'][top[0]]-x['base'][top[1]])/mass>1/3+1e-6
        rank=next((i+1 for i,j in enumerate(order) if x['ids'][j]==y),None)
        eligible=nr.hangul(x['row']) and x['gap']<4 and x['en']!=x['hs']
        before=z['baseline']==y; after=z['candidate']==y
        d=dict(sample_id=sid,truth=y,scenario=r.get('scenario','unknown'),
               baseline_correct=before,candidate_correct=after,
               candidate_absent=rank is None,base_rank=rank,
               rank_below20=rank is not None and rank>20,
               baseline_neural_skip=skip,
               post_neural_misranking=rank is not None and rank<=20 and not skip and x['v20']!=y,
               router_override=x['v20']==y and not before,
               scope_eligible=eligible,truth_in_rescue_pool=y in z['pool'],
               encoder_rejected_truth=y in z['pool'] and y not in z['shortlist'])
        detail.append(d)
        c=counts.setdefault(d['scenario'],Counter())
        c.update(rows=1,baseline_correct=int(before),candidate_correct=int(after),
                 rescue=int(not before and after),harm=int(before and not after),
                 changed=int(z['changed']),scope_error_ceiling=int(not before and eligible),
                 pool_error_ceiling=int(not before and eligible and y in z['pool']))
        for key in ('candidate_absent','rank_below20','baseline_neural_skip','post_neural_misranking','router_override'):
            c['error_'+key]+=int(not before and d[key])
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.write_text(json.dumps({'label_scope':'SYNTHETIC,TABLE-SEEN/STYLE-BIASED',
                                   'status':'HOLD','counts':counts,'rows':detail},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
