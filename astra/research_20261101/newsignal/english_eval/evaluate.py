"""Frozen English hash-fold-0 intrinsic evaluation; no Korean inputs or tuning."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import unicodedata
from collections import defaultdict
from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE.parents[1]
SOURCE = BASE / 'english_pairs.jsonl'
PARAMS = BASE / 'translit/NORMALIZED_FIT.json'
MODEL = BASE / 'translit/normalized_channel.py'
CATALOG = ROOT / 'data/open/vessel_catalog.csv'

def norm(s):
    return ''.join(c.upper() for c in unicodedata.normalize('NFC', str(s)) if c.isalnum())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def aggregate(rows):
    result = {'n': len(rows), 'unique_vessels': len({r['vessel_id'] for r in rows})}
    for name in ('baseline', 'channel', 'strict'):
        result[name + '_correct'] = sum(r[name] == r['vessel_id'] for r in rows)
    for name in ('channel', 'strict'):
        result[name + '_changed'] = sum(r[name] != r['baseline'] for r in rows)
        result[name + '_rescue'] = sum(r['baseline'] != r['vessel_id'] and r[name] == r['vessel_id'] for r in rows)
        result[name + '_harm'] = sum(r['baseline'] == r['vessel_id'] and r[name] != r['vessel_id'] for r in rows)
        result[name + '_net'] = result[name + '_rescue'] - result[name + '_harm']
    result['true_candidate_prefilter'] = sum(r['true_prefilter'] for r in rows)
    result['true_positive_mass'] = sum(r['true_mass'] > 0 for r in rows)
    result['no_positive_channel_candidates'] = sum(r['positive_candidate_count'] == 0 for r in rows)
    return result

def main():
    start = time.perf_counter()
    params = json.loads(PARAMS.read_text())
    spec = importlib.util.spec_from_file_location('frozen_normalized_channel', MODEL)
    channel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(channel)
    train_ids = set(params['train_vessel_ids'])
    original = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    rows = [r for r in original if int(hashlib.sha256(('channel-v1:' + r['vessel_id']).encode()).hexdigest()[:8], 16) % 5 == 0]
    assert rows and all(r['fold'] == 0 for r in rows)
    assert not (train_ids & {r['vessel_id'] for r in rows})
    assert params['source_sha256'] == digest(SOURCE)
    allcat = sorted(csv.DictReader(CATALOG.open(encoding='utf-8-sig')), key=lambda r: r['vessel_id'])
    catalog = [r for r in allcat if r['vessel_id'] not in train_ids]
    ids = [r['vessel_id'] for r in catalog]
    names = [norm(r['vessel_name']) for r in catalog]
    canonical = dict(zip(ids, names))
    assert len(ids) == len(set(ids))
    assert all(canonical[r['vessel_id']] == r['clean'] for r in rows)
    timing = defaultdict(float)
    results = []
    for number, row in enumerate(rows):
        q = row['query']
        t = time.perf_counter()
        # extractOne returns first equal maximum; ids are sorted ascending.
        b = process.extractOne(q, names, scorer=fuzz.ratio)
        baseline = ids[b[2]]
        timing['baseline_seconds'] += time.perf_counter() - t
        t = time.perf_counter()
        candidates = process.extract(q, names, scorer=DamerauLevenshtein.distance, score_cutoff=2, limit=None)
        timing['prefilter_seconds'] += time.perf_counter() - t
        t = time.perf_counter()
        masses = [(channel.score(clean, q, params), ids[idx]) for clean, _, idx in candidates]
        masses = sorted((p, vid) for p, vid in masses if p > 0)
        mass_by_id = {vid: p for p, vid in masses}
        total = math.fsum(p for p, _ in masses)
        if total:
            winner_mass, winner = min(masses, key=lambda pair: (-pair[0], pair[1]))
            share = winner_mass / total
        else:
            winner, winner_mass, share = baseline, 0.0, 0.0
        strict = winner if share >= 0.99 else baseline
        coeff = channel.path_coefficients(row['clean'], q)
        sequences = [key for key in coeff if key]
        if row['clean'] == q:
            op = 'exact'
        elif sequences:
            min_steps = min(len(key.split('/')) for key in sequences)
            op = '|'.join(sorted(key for key in sequences if len(key.split('/')) == min_steps))
        else:
            op = 'outside_0_1_2_support'
        timing['channel_seconds'] += time.perf_counter() - t
        result = {
            'vessel_id': row['vessel_id'], 'clean': row['clean'], 'query': q,
            'native_group': 'clean' if q == row['clean'] else 'actual_noisy',
            'operation': op, 'source_label': row['label'],
            'baseline': baseline, 'baseline_score': b[1], 'channel': winner, 'strict': strict,
            'channel_winner_mass': winner_mass, 'channel_total_mass': total,
            'winner_mass_share': share, 'prefilter_count': len(candidates),
            'positive_candidate_count': len(masses),
            'true_prefilter': any(ids[idx] == row['vessel_id'] for _, _, idx in candidates),
            'true_mass': mass_by_id.get(row['vessel_id'], 0.0),
        }
        results.append(result)
        if (number + 1) % 200 == 0:
            print(json.dumps({'completed': number + 1, 'total': len(rows), 'seconds': round(time.perf_counter() - start, 2)}), flush=True)
    group = defaultdict(list)
    for row in results:
        group[row['operation']].append(row)
    summary = {
        'kind': 'intrinsic English name-only evaluation, NOT C_vote; no Public inference',
        'catalog_original': len(allcat), 'excluded_fit_vessels': len(train_ids), 'catalog_evaluated': len(catalog),
        'heldout_fit_vessel_overlap': 0, 'hash_fold': 0,
        'baseline': 'full eligible catalog fuzz.ratio (normalized Indel similarity), ascending vessel_id ties',
        'channel': 'unrestricted DamerauLevenshtein <=2 prefilter, all normalized 0/1/2 path mass, equal vessel prior',
        'strict': 'frozen >=0.99 winner share of total positive candidate mass; else baseline',
        'all': aggregate(results),
        'native_clean': aggregate([r for r in results if r['native_group'] == 'clean']),
        'actual_noisy': aggregate([r for r in results if r['native_group'] == 'actual_noisy']),
        'by_operation': {op: aggregate(rr) for op, rr in sorted(group.items())},
        'timing': {**timing, 'total_seconds': time.perf_counter() - start},
        'sha256': {str(p.relative_to(ROOT)): digest(p) for p in (SOURCE, PARAMS, MODEL, CATALOG, Path(__file__))},
        'parameters': {key: params[key] for key in ('operation_probabilities', 'step_probabilities')},
        'warnings': [
            'These are actual supplied normalized English forms, not generated corruption; clean means normalized equality, not raw spelling equality.',
            'Existing C_vote predictions and Korean outcomes are not read; no claim of C_vote gain.',
            'No parameter or gate tuning; heldout truths used only for scoring and post-hoc operation labels.',
            'Posterior mass share is model-relative, not empirically calibrated confidence.',
            'fit excludes unsupported training forms, while this test retains all heldout forms.',
        ],
    }
    (OUT / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
