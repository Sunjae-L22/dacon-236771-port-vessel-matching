"""Author: Astra. Add missing known train positives to cold-ship ranking groups.

Only training candidates change. Outer-fold inference uses the untouched shared
evaluation candidate cache; all training statistics omit each query's vessel.
"""
from pathlib import Path
import os
os.environ['KO_DICT'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import ast
import fcntl
import gc
import hashlib
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'astra/positive96'
SOURCE = ROOT / 'astra/coldship91/fold0'
sys.path.insert(0, str(ROOT / 'shared/experiments/pinned/build84_astra'))
import cv, opus_lib as B, astra_lib as A


def assert_rank_reconstruction_safe():
    source = (ROOT / 'shared/experiments/pinned/build84_astra/opus_lib.py').read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'query_features')
    rank_fields = []
    for n in ast.walk(function):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Name) and n.value.value.id == 'cand':
            assert isinstance(n.slice, ast.Constant)
            rank_fields.append(n.slice.value)
    assert set(rank_fields) == {0, 1}, rank_fields
    return rank_fields


def main():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
    keep = [i for i, name in enumerate(A.FEATURES) if name != 'rank_name']
    names = [A.FEATURES[i] for i in keep]
    rank_name, rank_gt = A.FEATURES.index('rank_name'), A.FEATURES.index('rank_gt')
    source_plan = json.loads((SOURCE / 'plan.json').read_text())
    assert source_plan['features'] == names and source_plan['fold'] == 0
    rows = json.loads((SOURCE / 'queries.json').read_text())
    inner = json.loads((SOURCE / 'inner_folds.json').read_text())
    read_rank_fields = assert_rank_reconstruction_safe()
    plan = {'fold': 0, 'source_signature': source_plan['signature'], 'queries': len(rows), 'features': names,
            'change': 'Append target with ranks [999,999,999] only to training groups missing a positive; recompute all group features',
            'inference_candidates_changed': False, 'rank_fields_used_in_features': read_rank_fields,
            'prespecified_gate': {'stress_gain_vs_88_min': 0.005, 'stress_gain_vs_coldship_min': 0.003, 'native_loss_allowed': 0, 'orig_loss_allowed': 0},
            'source_plan_sha256': hashlib.sha256((SOURCE / 'plan.json').read_bytes()).hexdigest()}
    plan_path = OUT / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    train, cat = cv.load(); folds = cv.folds(train)
    fit = train[folds != 0].copy(); held = set(train[folds == 0].target_vessel_id)
    assert set(inner) == set(fit.target_vessel_id) and not set(inner) & held
    llm = B.llm_korean_rows(fit, ROOT / 'astra/ranker84_fold0/llm_fit.csv')
    source_paths = sorted((SOURCE / 'features').glob('*.npz'))
    cache = OUT / 'features'; cache.mkdir(exist_ok=True)
    indexes = []
    learned_targets = []
    if any(not (cache / p.name).exists() for p in source_paths):
        for part in range(3):
            targets = {v for v, n in inner.items() if n != part}
            sub = fit[fit.target_vessel_id.isin(targets)]
            gen = llm[llm.target_vessel_id.isin(targets)]
            assert not targets & held
            assert set(sub.target_vessel_id) <= targets and set(gen.target_vessel_id) <= targets
            learned_targets.append(targets)
            ix = A.Index(cat, sub)
            ix.ko = B.KA.KoMatcher(cat, pd.concat([sub, gen], ignore_index=True))
            indexes.append(ix)
            print('INDEX', part, len(sub), len(gen), flush=True)
        for path in source_paths:
            destination = cache / path.name
            if destination.exists():
                continue
            xs, ys, ids, groups, repaired = [], [], [], [], []
            with np.load(path) as z:
                original_X, original_y, original_ids = z['X'], z['y'], z['idx']
                offset = 0
                for local, size in enumerate(z['groups']):
                    index = int(z['query_index'][local]); row = rows[index]
                    part = int(z['inner_fold'][local]); target = int(z['target'][local])
                    assert part == inner[row['target_vessel_id']]
                    ix = indexes[part]
                    assert ix.ids[target] == row['target_vessel_id']
                    assert row['target_vessel_id'] not in learned_targets[part]
                    oldX = original_X[offset:offset + size]
                    oldy = original_y[offset:offset + size]
                    oldidx = original_ids[offset:offset + size]
                    assert int(oldy.sum()) in (0, 1)
                    missing = not bool(oldy.any())
                    assert missing == (not bool(z['present'][local]))
                    if missing:
                        cand = {int(i): [int(x[rank_name]), int(x[rank_gt]), 999] for i, x in zip(oldidx, oldX)}
                        idx_check, feature_check = A.query_features(ix, row, cand)
                        np.testing.assert_array_equal(idx_check, oldidx)
                        np.testing.assert_array_equal(feature_check, oldX)
                        assert target not in cand
                        cand[target] = [999, 999, 999]
                        idx, features = A.query_features(ix, row, cand)
                        label = np.array([i == target for i in idx], dtype=np.int8)
                        assert label.sum() == 1 and len(idx) == size + 1
                        xs.append(features[:, keep]); ys.append(label); ids.extend(idx); groups.append(len(idx))
                    else:
                        # Retain every float and label exactly for already-positive groups.
                        xs.append(oldX[:, keep]); ys.append(oldy); ids.extend(oldidx.tolist()); groups.append(int(size))
                    repaired.append(missing); offset += size
                assert offset == len(original_y)
                metadata = {key: z[key] for key in ('query_index', 'target', 'inner_fold')}
            temp = destination.with_suffix('.tmp.npz')
            np.savez_compressed(temp, X=np.concatenate(xs), y=np.concatenate(ys), idx=np.array(ids, dtype=np.int32), groups=np.array(groups, dtype=np.int32), repaired=np.array(repaired), **metadata)
            temp.replace(destination)
            print('REPAIRED_CHUNK', path.name, sum(repaired), 'elapsed', round(time.time() - started), flush=True)
            del xs, ys, original_X, original_y; gc.collect()
        del indexes; gc.collect()
    paths = [cache / p.name for p in source_paths]
    total, all_groups, all_y, repaired = 0, [], [], []
    for path in paths:
        with np.load(path) as z:
            groups = z['groups']; y = z['y']; offset = 0
            for size in groups:
                assert y[offset:offset + size].sum() == 1
                offset += size
            assert offset == len(y)
            all_groups.extend(groups.tolist()); all_y.append(y); repaired.extend(z['repaired'].tolist()); total += len(y)
    assert len(all_groups) == len(rows) == 8952
    assert sum(repaired) == 341
    training_receipt = {'queries': len(rows), 'groups_repaired': int(sum(repaired)), 'groups_reused_unchanged': len(rows) - int(sum(repaired)), 'fraction_repaired': float(np.mean(repaired)), 'groups_with_positive_after': len(rows), 'all_repaired_original_features_exactly_reproduced_before_injection': True, 'candidate_rows': total}
    (OUT / 'training_receipt.json').write_text(json.dumps(training_receipt, indent=2) + '\n')
    model_path = OUT / 'ranker.txt'
    if not model_path.exists():
        X = np.lib.format.open_memmap(cache / 'matrix.npy', mode='w+', dtype=np.float32, shape=(total, len(keep)))
        offset = 0
        for path in paths:
            with np.load(path) as z:
                n = len(z['y']); X[offset:offset + n] = z['X']; offset += n
        X.flush(); y = np.concatenate(all_y)
        assert sum(all_groups) == len(X) == len(y)
        params = dict(objective='lambdarank', metric='ndcg', eval_at=[1], learning_rate=.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=84, deterministic=True, force_col_wise=True)
        print('TRAIN', total, 'candidates', round(time.time() - started), flush=True)
        model = lgb.train(params, lgb.Dataset(X, y, group=all_groups, feature_name=names), 600)
        model.save_model(str(model_path)); del X, y; gc.collect()
    else:
        model = lgb.Booster(model_file=str(model_path))
    print('TRAINED', round(time.time() - started), flush=True)
    results = []
    with np.load(ROOT / 'astra/neural87_fold0/eval_features_enhanced.npz') as z:
        scores = model.predict(z['X'][:, keep], num_threads=1)
        idx, allowed = z['idx'], z['allowed']; offset = 0
        for query, size in enumerate(z['groups']):
            pool = np.arange(offset, offset + size); valid = pool[allowed[pool]]
            if not len(valid):
                valid = pool
            winner = min(valid, key=lambda j: (-float(scores[j]), int(idx[j])))
            results.append(dict(fold=0, target=int(z['target'][query]), variant=str(z['variant'][query]), sample_id=str(z['sample_id'][query]), method='positive96', pred=int(idx[winner]), correct=bool(idx[winner] == z['target'][query])))
            offset += size
    predictions = pd.DataFrame(results); predictions.to_csv(OUT / 'predictions.csv', index=False)
    paired = predictions
    for name, path in [('rankless', ROOT / 'astra/rankless87_fold0/predictions.csv'), ('coldship', SOURCE / 'predictions.csv')]:
        old = pd.read_csv(path).rename(columns={'pred': f'{name}_pred', 'correct': f'{name}_correct'})
        paired = paired.merge(old[['target', 'sample_id', 'variant', f'{name}_pred', f'{name}_correct']], on=['target', 'sample_id', 'variant'], validate='one_to_one')
    paired.to_csv(OUT / 'paired_predictions.csv', index=False)
    report = {'fold': 0, 'seconds': time.time() - started, 'training': training_receipt, 'inference_candidates_changed': False}
    for label, frame in [('stress', paired[paired.variant.ne('native_all_rows')]), ('native', paired[paired.variant.eq('native_all_rows')]), ('orig', paired[paired.variant.eq('orig')])]:
        report[label] = {'positive96': float(frame.correct.mean()), 'rankless88': float(frame.rankless_correct.mean()), 'coldship': float(frame.coldship_correct.mean()), 'delta_vs_88': float(frame.correct.mean() - frame.rankless_correct.mean()), 'delta_vs_coldship': float(frame.correct.mean() - frame.coldship_correct.mean())}
    report['prespecified_pilot_gate_passed'] = bool(report['stress']['delta_vs_88'] >= .005 and report['stress']['delta_vs_coldship'] >= .003 and min(report['native']['delta_vs_88'], report['native']['delta_vs_coldship'], report['orig']['delta_vs_88'], report['orig']['delta_vs_coldship']) >= 0)
    report['variants'] = paired.groupby('variant')[['rankless_correct', 'coldship_correct', 'correct']].mean().to_dict('index')
    report['validation_caveat'] = 'Exploratory fold0 pilot using a previously used outer evaluation fold; no fold4 confirmation run.'
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
