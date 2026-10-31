"""Replay frozen views against label-free captured packets; never emits a submission.
Every failure after output reservation has an ABORTED status; no partial is required
before scoring starts. Parent process must enforce cold-start wall-time watchdog.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
from neural_challenge import challenge


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--packets', type=Path, required=True)
    ap.add_argument('--catalog', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baseline-seconds', type=float, required=True)
    args=ap.parse_args()
    started=time.monotonic()
    partial=args.output.with_suffix(args.output.suffix+'.partial')
    manifest=args.output.with_suffix(args.output.suffix+'.status.json')
    # Preserve a previous run rather than replacing its status on collision.
    if any(p.exists() for p in (args.output,partial,manifest)):
        raise FileExistsError('refuse overwrite of previous run')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    status={'state':'RUNNING','stage':'prepare','rows':0,'status':'HOLD'}
    manifest.write_text(json.dumps(status))
    count=changed=0
    try:
        if not math.isfinite(args.baseline_seconds) or args.baseline_seconds < 0:
            raise ValueError('invalid measured baseline duration')
        budget=min(900.0,1800.0-args.baseline_seconds-120.0)
        status['budget_seconds']=budget
        if budget<=0:raise RuntimeError('baseline exhausted time budget')
        def check_budget(stage):
            status['stage']=stage
            if time.monotonic()-started>budget:
                raise TimeoutError('challenge budget exceeded: '+stage)
        check_budget('verify_sources')
        spec=json.loads((args.model_dir/'encoder_views.json').read_text())
        if spec['top_k']!=20 or spec['weight']!=.25:
            raise RuntimeError('unexpected baseline mixing specification')
        if [(v['kind'],v['recipe'],v['weight']) for v in spec['views']] != [('qwen','evidence',1.0),('encoder','phonetic',1.0)]:
            raise RuntimeError('unexpected frozen view specification')
        for name,digest in spec['source_hashes'].items():
            path=args.model_dir/'src'/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise RuntimeError('source hash mismatch: '+name)
        excluded=set(map(str,json.loads((args.model_dir/'train_vessels.json').read_text())))
        status['packets_sha256']=hashlib.sha256(args.packets.read_bytes()).hexdigest()
        status['catalog_sha256']=hashlib.sha256(args.catalog.read_bytes()).hexdigest()
        status['spec_sha256']=hashlib.sha256((args.model_dir/'encoder_views.json').read_bytes()).hexdigest()
        check_budget('read_packets')
        packets=[json.loads(s) for s in args.packets.read_text().splitlines() if s.strip()]
        if not packets:raise ValueError('empty inference cohort')
        if len({p['sample_id'] for p in packets})!=len(packets):raise ValueError('duplicate sample IDs')
        status['input_rows']=len(packets)
        for p in packets:
            if {'truth','target','target_vessel_id'} & (set(p['row'])|set(p)):
                raise ValueError('label-bearing inference row')
        check_budget('load_dependencies')
        sys.path.insert(0,str(args.model_dir.resolve()))
        import pandas as pd
        import torch
        from src.encoder_views import ViewScorer
        torch.set_num_threads(2)
        if not torch.cuda.is_available():raise RuntimeError('actual GPU run required')
        catalog_frame=pd.read_csv(args.catalog,encoding='utf-8-sig')
        catalog={str(r['vessel_id']):r for r in catalog_frame.to_dict('records')}
        if len(catalog)!=len(catalog_frame):raise ValueError('duplicate catalog IDs')
        # Validate all real component IDs even when no candidate scoring occurs.
        for p in packets:
            from neural_challenge import route
            route(p['row'].get('vessel_name'),p['gap110'],p['votes'])
            if any(v not in catalog for v in p['votes'].values()):
                raise ValueError('component output outside official catalog')
        views={}
        for v in spec['views']:
            check_budget('load_model:'+v['kind'])
            scorer=ViewScorer(args.model_dir/v['dir'],v['kind'],v['recipe'])
            check_budget('loaded_model:'+v['kind'])
            def score(row,docs,beams,s=scorer):
                check_budget('score:'+s.name)
                result=s.score(row,docs,beams)
                check_budget('scored:'+s.name)
                return result
            views[scorer.name]=score
        check_budget('score_rows')
        with partial.open('x') as f:
            for p in packets:
                check_budget('score_rows')
                result=challenge(p,catalog,excluded,views)
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                count+=1;changed+=result['prediction']!=result['baseline']
        check_budget('synchronize')
        torch.cuda.synchronize()
        check_budget('finalize')
        status.update(rows=count,changed=changed,gpu=torch.cuda.get_device_name(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_version=torch.__version__,cuda_version=torch.version.cuda)
        partial.rename(args.output)
        status.update(state='COMPLETE',stage='complete')
    except Exception as e:
        status.update(state='ABORTED',rows=count,error=repr(e))
        raise
    finally:
        status.update(seconds=time.monotonic()-started,partial_present=partial.exists(),
                      completed_output_present=args.output.exists())
        manifest.write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
