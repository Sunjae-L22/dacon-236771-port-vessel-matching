"""Offline experimental scorer for captured real C_vote packets. Writes no submission.

Example: python run_gpu.py --model-root /content/C_vote/model/ko/model \
  --packets trace.joined.jsonl --catalog data/vessel_catalog.csv --output rescue.jsonl
Baseline ZIP SHA256 and exact baseline replay must be checked by the parent runner.
"""
import os
import sys
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import time
import neural_rescue as nr


def check_deadline(deadline, stage):
    if time.monotonic() >= deadline:
        raise TimeoutError(f'Experiment budget expired: {stage}')


class BudgetScorer:
    def __init__(self, scorer, deadline):
        self.scorer, self.deadline = scorer, deadline
        self.pairs, self.seconds = 0, 0.0
    def score(self, *args):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Experiment budget expired before batch')
        started = time.monotonic()
        result = self.scorer.score(*args)
        self.seconds += time.monotonic()-started
        self.pairs += len(args[1])
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Experiment budget expired during batch')
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-root', type=Path, required=True)
    ap.add_argument('--packets', type=Path, required=True)
    ap.add_argument('--catalog', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--budget-seconds', type=int, default=600)
    args = ap.parse_args()
    if not 1 <= args.budget_seconds <= 900:
        raise ValueError('Budget must be 1..900 seconds; parent must reserve total <=1800s')
    if args.output.exists() or args.output.with_suffix('.status.json').exists():
        raise FileExistsError('Refusing to overwrite previous results')
    started = time.monotonic()
    deadline = started + args.budget_seconds
    status = {'state':'RUNNING', 'mechanism':'frozen_dual_view_rescue_v1',
              'status':'HOLD', 'budget_seconds':args.budget_seconds}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial = args.output.with_suffix('.partial.jsonl')
    try:
        spec = json.loads((args.model_root/'encoder_views.json').read_text())
        if spec['top_k'] != 20 or spec['weight'] != .25:
            raise ValueError('Wrong baseline encoder-view configuration')
        expected_views = [('qwen','evidence',1.0),('encoder','phonetic',1.0)]
        if [(v['kind'],v['recipe'],v['weight']) for v in spec['views']] != expected_views:
            raise ValueError('Unexpected frozen scorer configuration')
        for filename, sha in spec['source_hashes'].items():
            if hashlib.sha256((args.model_root/'src'/filename).read_bytes()).hexdigest() != sha:
                raise ValueError(f'Source hash mismatch: {filename}')
        status['packets_sha256'] = hashlib.sha256(args.packets.read_bytes()).hexdigest()
        status['catalog_sha256'] = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
        packets = [json.loads(line) for line in args.packets.read_text().splitlines() if line.strip()]
        if len(set(p['sample_id'] for p in packets)) != len(packets):
            raise ValueError('Duplicate sample IDs')
        train_ids=set(json.loads((args.model_root/'train_vessels.json').read_text()))
        for p in packets:
            nr.validate(p)
            if set(p['excluded']) != set(p['ids']) & train_ids:
                raise ValueError('Trace exclusion disagrees with original production train list')
        packets.sort(key=lambda p:p['sample_id'])
        eligible = [p for p in packets if nr.hangul(p['row']) and p['gap']<4 and p['en']!=p['hs']]
        status['rows'], status['eligible_rows'] = len(packets),len(eligible)
        # This is resource planning, never a change to rowwise inference.
        import pandas as pd
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required for actual scorer run')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.manual_seed(0)
        status['hardware'] = {'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__,
                              'cuda':torch.version.cuda}
        catalog_frame = pd.read_csv(args.catalog,encoding='utf-8-sig')
        if catalog_frame['vessel_id'].astype(str).duplicated().any(): raise ValueError('Duplicate catalog IDs')
        catalog = {str(r['vessel_id']):r for r in catalog_frame.to_dict('records')}
        for p in packets:
            if any(p[field] not in catalog for field in ('v20','en','hs')):
                raise ValueError('Voter ID is not in official catalog')
            if any(i not in catalog for i in p['ids']):
                raise ValueError('Trace candidate is not in official catalog')
        sys.path.insert(0,str(args.model_root))
        from src.encoder_views import ViewScorer
        enc = BudgetScorer(ViewScorer(args.model_root/'encoder','encoder','phonetic'),started+args.budget_seconds)
        qw = BudgetScorer(ViewScorer(args.model_root/'qwen','qwen','evidence'),started+args.budget_seconds)
        check_deadline(deadline, 'model load')
        results=[]
        scored=0
        scoring_start=time.monotonic()
        with partial.open('w',encoding='utf-8') as out:
            for p in packets:
                check_deadline(deadline, 'before row')
                result=nr.run_one(p,catalog,enc,qw)
                results.append(result)
                out.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                if result['pool']: scored+=1
                if scored==8 and len(eligible)>8:
                    projected=(time.monotonic()-scoring_start)/scored*len(eligible)
                    status['projected_scoring_seconds_after8']=projected
                    if projected > started+args.budget_seconds-scoring_start:
                        raise TimeoutError('Projected runtime exceeds frozen budget; partial results are diagnostic only')
        # Prediction-level order/subset replay on up to8 deterministic rows.
        by_id={r['sample_id']:r for r in results}
        for p in reversed(eligible[:8]):
            check_deadline(deadline, 'before subset replay')
            replay=nr.run_one(p,catalog,enc,qw)
            prior=by_id[p['sample_id']]
            for key in ('candidate','v20_after','pool','shortlist','reason'):
                if replay[key]!=prior[key]: raise RuntimeError(f'Subset replay mismatch: {key}')
        check_deadline(deadline, 'after rows and replay')
        status['results_sha256'] = hashlib.sha256(partial.read_bytes()).hexdigest()
        check_deadline(deadline, 'before COMPLETE commit')
        partial.rename(args.output)
        try:
            check_deadline(deadline, 'after result rename')
        except TimeoutError:
            args.output.rename(partial)
            raise
        status.update(state='COMPLETE', changed=sum(r['changed'] for r in results),
                      encoder_pairs=enc.pairs,qwen_pairs=qw.pairs,
                      encoder_seconds=enc.seconds,qwen_seconds=qw.seconds,
                      gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                      subset_replay_rows=min(8,len(eligible)))
    except Exception as exc:
        status.update(state='ABORT',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['seconds']=time.monotonic()-started
        args.output.with_suffix('.status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
