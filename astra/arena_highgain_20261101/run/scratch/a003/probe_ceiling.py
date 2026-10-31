"""Read-only stale112 falsification; labels used only in this separate evaluator."""
import sys
sys.dont_write_bytecode = True
import collections
import fcntl
import json
from pathlib import Path

ROOT = Path('/Users/sunjae/Desktop/항만')
counts = collections.Counter()
with (ROOT / 'astra/arena_highgain_20261101/CPU.lock').open('r') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for fold in (2, 4):
        path = ROOT / f'astra/byt5_base112_20261031/fold{fold}/packets112.jsonl'
        full_rows = {}
        for line in (path.parent / 'rows.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r['query_id'] in full_rows:
                raise ValueError('duplicate query_id in full-pool provenance')
            full_rows[r['query_id']] = r
        seen = set()
        with path.open() as source:
            for line in source:
                p = json.loads(line)
                if p['query_id'] in seen:
                    raise ValueError('duplicate packet query_id')
                seen.add(p['query_id'])
                r = full_rows[p['query_id']]
                if r['target'] != p['target'] or r['p112'] != p['baseline']:
                    raise ValueError('full-pool provenance alignment mismatch')
                counts['rows'] += 1
                if p['gate110_gap'] >= 4:
                    counts['protected'] += 1
                    continue
                counts['eligible112proxy'] += 1
                ids = [c['id'] for c in p['candidates']]
                # Never feed target into candidate selection or inference.
                truth = p['target']
                if p['baseline'] == truth:
                    counts['eligible_correct'] += 1
                    continue
                counts['eligible_wrong'] += 1
                rank = ids.index(truth) + 1 if truth in ids else None
                if rank is None:
                    bucket = ('beyond_saved_top100_full_present' if r['target_in112_full']
                              else 'full_pool_absent')
                else:
                    if not r['target_in112_full']:
                        raise ValueError('inconsistent candidate coverage provenance')
                    bucket = ('top20' if rank <= 20 else
                          'rank21_64' if rank <= 64 else 'rank65_128' if rank <= 128 else 'below128')
                counts['wrong_' + bucket] += 1
        if seen != set(full_rows):
            raise ValueError('full-pool provenance cohort mismatch')
report = {'kind': 'STALE112_PROXY_ONLY_NOT_CVOTE', 'counts': dict(counts),
          'limitations': ['KC1 vs production KC6', 'no v20/HSW outputs',
                          'ranks are 112 scores, not v20 base',
                          'packets save top100 only; rows.jsonl supplies full coverage, not exact rank beyond100',
                          'labels used only for evaluation; no selection or tuning']}
Path('proxy_ceiling.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False))
