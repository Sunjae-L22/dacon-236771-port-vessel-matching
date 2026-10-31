"""Matched79 additive-prefix experiment; default is read-only preflight.

Use --mode train only after resource authorization. No test data, production
model evaluation, training-row cap, candidate changes, or old-feature changes.
"""
from pathlib import Path
import argparse
import fcntl
import gc
import hashlib
import json
import os
import sys
import time

os.environ['KO_DICT'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import lightgbm as lgb

import cache_features as C
from features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'astra'))
import incumbent97 as BASE

SEEDS = [0, 5000, 1000, 2000, 6000, 7000]
KINDS = ['old9', 'benchmark', 'llm_holdout']


def load_plan(fold):
    out = BASE.BASE/f'fold{fold}'
    plan = json.loads((out/'plan.json').read_text())
    assert plan['features'] == C.B.FEATURES and len(plan['features']) == 60
    assert plan['lgb_params'] == BASE.PARAMS
    assert 'seed' not in BASE.PARAMS, 'Explicit seed=0 changes the baseline subordinate seeds'
    assert plan['trees'] == 600 and plan['training_seeds'] == SEEDS
    for name, expected in plan['source']['sources'].items():
        assert C.file_hash(BASE.SRC/name) == expected, f'Baseline source changed: {name}'
    return out, plan


def ordered_chunks(out, plan, seed):
    paths, total_queries = [], 0
    reference_rows = []
    order = plan['seeds'][str(seed)]['order']
    assert order == ['real', 'nogt', f'seed{seed}_noisy', f'seed{seed}_corrupt', f'seed{seed}_rename', 'llm']
    for name in order:
        rowpath = out/'blocks'/f'{name}.json'
        rows = json.loads(rowpath.read_text())
        reference_rows.extend(rows)
        for lo in range(0, len(rows), 500):
            path = out/'features'/name/f'{lo:06d}.npz'
            assert path.exists(), f'Wait for the matched baseline cache: {path}'
            paths.append((name, path, rowpath, lo, min(500, len(rows)-lo)))
        total_queries += len(rows)
    assert total_queries == plan['seeds'][str(seed)]['queries']
    signature = hashlib.sha256(json.dumps(reference_rows, sort_keys=True, default=str).encode()).hexdigest()
    assert signature == plan['seeds'][str(seed)]['queries_sha256'], 'Baseline query records/order changed'
    assert (out/f'seed{seed}/ranker.txt').exists(), 'Require a completed matched counterpart'
    return paths


def preflight(fold, seeds):
    out, plan = load_plan(fold)
    reports = []
    for seed in seeds:
        paths = ordered_chunks(out, plan, seed)
        count = 0
        groups = 0
        for _, path, _, lo, size in paths:
            with np.load(path) as z:
                g = z['groups']; y = z['y']; ix = z['idx']; target = z['target']
                assert len(g) == size and g.sum() == len(y) == len(ix)
                assert np.array_equal(z['query_index'], np.arange(lo, lo+size))
                assert np.array_equal(y, ix == np.repeat(target, g))
                groups += len(g); count += len(y)
        assert groups == plan['seeds'][str(seed)]['queries']
        reports.append(dict(augmentation_seed=seed, chunks=len(paths), groups=groups,
                            candidate_rows=count, matrix63_bytes=count*63*4))
    return dict(status='PREFLIGHT_ONLY_NO_TRAINING', fold=fold, seeds=reports,
                baseline_plan_sha256=C.file_hash(out/'plan.json'),
                feature_names=plan['features']+FEATURE_NAMES,
                params=dict(BASE.PARAMS), trees=600, training_row_cap=None,
                additive_features_only=True, existing_feature_count=60,
                candidate_generation='Reused complete immutable incumbent97 caches',
                explicit_seed_parameter=False, expected_default_subseeds={'feature_fraction_seed':2, 'bagging_seed':3},
                resource_limit='One training process, num_threads=2; parent authorizes scheduling',
                training_run=False, submission_created=False)


def ensure_extra(ix, baseline, fold, mode, category, rows, paths, provenance):
    outputs = []
    for path in paths:
        dst = HERE/f'fold{fold}'/mode/category/path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if dst.exists():
                report = json.loads(dst.with_suffix('.json').read_text())
                assert report['provenance'] == provenance
                assert report['source_cache_sha256'] == C.file_hash(path)
                assert report['output_extra_cache_sha256'] == C.file_hash(dst)
            else:
                E, z, report = C.calculate(ix, rows, path)
                assert len(E) == len(z['idx']) and E.dtype == np.float32 and np.isfinite(E).all()
                arrays = {key:z[key] for key in ['idx', 'groups', 'target', 'query_index', 'allowed'] if key in z}
                arrays['extra_features'] = E
                tmp = dst.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, **arrays); tmp.replace(dst)
                report.update(provenance=provenance, output_extra_cache_sha256=C.file_hash(dst))
                C.dump(dst.with_suffix('.json'), report)
                print('PREFIX_EXTRA', category, path.name, len(E), flush=True)
            outputs.append(dst)
    return outputs


def training_extras(ix, out, plan, fold, seed, provenance):
    ordered = ordered_chunks(out, plan, seed)
    for block in plan['seeds'][str(seed)]['order']:
        rows = json.loads((out/'blocks'/f'{block}.json').read_text())
        paths = [p for name,p,_,_,_ in ordered if name == block]
        ensure_extra(ix, out, fold, 'train-block', block, rows, paths, provenance)
    return ordered


def verify_arrays(base, extra):
    for key in ['idx', 'groups', 'target', 'query_index', 'allowed']:
        if key in base:
            assert key in extra and np.array_equal(base[key], extra[key]), f'Extra cache changed {key}'
    E = extra['extra_features']
    assert E.dtype == np.float32 and E.shape == (len(base['idx']), 3) and np.isfinite(E).all()
    return E


def construct_matrix(fold, seed, ordered, seedout, expected_queries):
    groups, ys, count = [], [], 0
    for _, path, _, _, _ in ordered:
        with np.load(path) as z:
            groups.extend(z['groups'].tolist()); ys.append(z['y']); count += len(z['y'])
    assert len(groups) == expected_queries
    matrix_path = seedout/'matrix63.npy'
    X = np.lib.format.open_memmap(matrix_path, mode='w+', dtype=np.float32, shape=(count, 63))
    offset = 0
    original_hasher = hashlib.sha256()
    appended_hasher = hashlib.sha256()
    sources = []
    for block, path, _, lo, size in ordered:
        extra_path = HERE/f'fold{fold}/train-block'/block/path.name
        with np.load(path) as z, np.load(extra_path) as e:
            E = verify_arrays(z, e); F = z['X']; n = len(F)
            assert F.dtype == np.float32 and F.shape == (n, 60)
            assert np.array_equal(z['query_index'], np.arange(lo, lo+size))
            assert np.array_equal(z['y'], z['idx'] == np.repeat(z['target'], z['groups']))
            X[offset:offset+n, :60] = F
            X[offset:offset+n, 60:] = E
            assert np.array_equal(X[offset:offset+n, :60].view(np.uint32), F.view(np.uint32))
            original_hasher.update(np.ascontiguousarray(F).view(np.uint8))
            appended_hasher.update(np.ascontiguousarray(X[offset:offset+n, :60]).view(np.uint8))
            offset += n
            sources.append(dict(block=block, chunk=path.name, source_sha256=C.file_hash(path), extra_sha256=C.file_hash(extra_path)))
    X.flush(); y = np.concatenate(ys)
    assert offset == len(X) == len(y) == sum(groups)
    assert original_hasher.hexdigest() == appended_hasher.hexdigest()
    C.dump(seedout/'matrix_manifest.json', dict(candidate_rows=count, query_groups=len(groups), feature_count=63,
        original60_sha256=original_hasher.hexdigest(), appended_first60_sha256=appended_hasher.hexdigest(),
        original_feature_bit_mismatches=0, unchanged_candidate_ids_groups_labels=True, sources=sources))
    return X, y, groups, matrix_path


def check_subseeds(model):
    params = model.params
    assert params.get('seed', 0) == 0
    text = model.model_to_string()
    assert '[feature_fraction_seed: 2]' in text and '[bagging_seed: 3]' in text


def evaluate(fold, seed, ix, out, provenance, model, heldout_ids):
    seedout = HERE/f'fold{fold}/seed{seed}'
    for kind in KINDS:
        rows, paths = C.sources(out, 'eval', None, kind)
        assert set(r['target_vessel_id'] for r in rows) <= heldout_ids
        assert len(paths) == 1
        extra = ensure_extra(ix, out, fold, 'eval', kind, rows, paths, provenance)[0]
        with np.load(paths[0]) as z, np.load(extra) as e:
            E = verify_arrays(z, e)
            F = z['X']
            scores = np.empty(len(F), dtype=np.float64)
            for lo in range(0, len(F), 100000):
                scores[lo:lo+100000] = model.predict(np.column_stack((F[lo:lo+100000], E[lo:lo+100000])), num_threads=1)
            dst = seedout/f'{kind}_scores.npy'; tmp = dst.with_suffix('.tmp.npy')
            np.save(tmp, scores); tmp.replace(dst)
            d = BASE.predict_rows(scores, z, rows, ix.ids, f'prefix98_aug{seed}')
            d.to_csv(seedout/f'{kind}_predictions.csv', index=False)
        print('PREFIX_EVAL', seed, kind, len(d), flush=True)


def comparison(frame):
    rows = []
    for keys, d in frame.groupby(['source_language', 'cohort', 'family', 'variant'], dropna=False):
        by_ship = d.groupby('target_vessel_id')[['correct_baseline', 'correct_prefix']].mean()
        rows.append(dict(zip(['source_language', 'cohort', 'family', 'variant'], keys)) | dict(
            rows=len(d), vessels=d.target_vessel_id.nunique(), baseline_correct=int(d.correct_baseline.sum()),
            prefix_correct=int(d.correct_prefix.sum()), corrected_rows=int((~d.correct_baseline & d.correct_prefix).sum()),
            regressed_rows=int((d.correct_baseline & ~d.correct_prefix).sum()),
            row_delta=float(d.correct_prefix.mean()-d.correct_baseline.mean()),
            vessel_macro_delta=float((by_ship.correct_prefix-by_ship.correct_baseline).mean())))
    return rows


def aggregate(fold):
    out, plan = load_plan(fold)
    own = HERE/f'fold{fold}'
    ids = pd.read_csv(ROOT/'data/open/vessel_catalog.csv', usecols=['vessel_id']).vessel_id.to_numpy()
    reports = {}
    for kind in KINDS:
        paths = [p for p in out.glob(f'eval_{kind}_*.npz') if '.tmp.' not in p.name]
        if not paths:
            continue
        assert len(paths) == 1
        rows = json.loads((out/f'eval_{kind}_queries.json').read_text())
        reports[kind] = {}
        for count in [1, 2, 6]:
            seeds = SEEDS[:count]
            if not all((own/f'seed{s}/{kind}_scores.npy').exists() for s in seeds):
                continue
            assert all((out/f'seed{s}/{kind}_scores.npy').exists() for s in seeds)
            score = sum(np.load(own/f'seed{s}/{kind}_scores.npy') for s in seeds)/count
            base_score = sum(np.load(out/f'seed{s}/{kind}_scores.npy') for s in seeds)/count
            with np.load(paths[0]) as z:
                prefix = BASE.predict_rows(score, z, rows, ids, f'prefix98_{count}models')
                base = BASE.predict_rows(base_score, z, rows, ids, f'matched79_{count}models')
            reference = out/f'{kind}_{count}models_predictions.csv'
            assert reference.exists()
            ref = pd.read_csv(reference)
            assert ref.query_id.tolist() == base.query_id.tolist() and ref.pred.tolist() == base.pred.tolist()
            merged = base.merge(prefix[['query_id', 'target_vessel_id', 'pred', 'correct']],
                on=['query_id', 'target_vessel_id'], suffixes=('_baseline', '_prefix'), validate='one_to_one')
            merged.to_csv(own/f'{kind}_{count}models_paired_predictions.csv', index=False)
            reports[kind][str(count)] = dict(augmentation_seeds=seeds, per_condition=comparison(merged),
                                           comparator_reproduction_mismatches=0)
    C.dump(own/'ensemble_report.json', dict(status='EXPLORATORY_MATCHED_COMPARISON', fold=fold,
        no_combined_cohort_score=True, reports=reports,
        limitations=['Repeated outerfold2 is exploratory.', 'Native Korean only76 vessels.',
                     'Compatibility-jamo production routing uses English65; Korean-only result is diagnostic.',
                     'LLM aliases are synthetic; their targets are assumed.',
                     'No production79 or test predictions used in CV.', 'No Public LB improvement claim.']))
    print('PREFIX_AGGREGATED', str(own/'ensemble_report.json'), flush=True)


def main(args):
    own = HERE/f'fold{args.fold}'
    own.mkdir(parents=True, exist_ok=True)
    if args.mode == 'aggregate':
        aggregate(args.fold); return
    seeds = [args.seed] if args.seed is not None else [0, 5000]
    report = preflight(args.fold, seeds)
    C.dump(own/'preflight.json', report)
    if args.mode == 'preflight':
        print(json.dumps(report, ensure_ascii=False, indent=2)); return
    assert args.seed is not None, 'Run exactly one augmentation seed per process'
    seed = args.seed
    seedout = own/f'seed{seed}'
    seedout.mkdir(parents=True, exist_ok=True)
    with (seedout/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.time()
        ix, out, provenance = C.context(args.fold)
        _, plan = load_plan(args.fold)
        tr, _ = C.cv.load()
        fit_ids = set(tr.loc[C.cv.folds(tr) != args.fold, 'target_vessel_id'])
        heldout_ids = set(tr.loc[C.cv.folds(tr) == args.fold, 'target_vessel_id'])
        assert not fit_ids & heldout_ids
        for block in plan['seeds'][str(seed)]['order']:
            rows = json.loads((out/'blocks'/f'{block}.json').read_text())
            assert set(r['target_vessel_id'] for r in rows) <= fit_ids
        modelpath = seedout/'ranker.txt'
        if args.mode == 'features' and modelpath.exists():
            print('PREFIX_MODEL_ALREADY_PRESENT_NO_TRAINING', seed, flush=True); return
        if args.mode in ['features', 'train'] and not modelpath.exists():
            ordered = training_extras(ix, out, plan, args.fold, seed, provenance)
            if args.mode == 'features':
                print('PREFIX_FEATURES_READY_NO_TRAINING', seed, flush=True); return
            X, y, groups, matrix_path = construct_matrix(args.fold, seed, ordered, seedout, plan['seeds'][str(seed)]['queries'])
            print('PREFIX_TRAIN', seed, 'queries', len(groups), 'candidates', len(y), flush=True)
            model = lgb.train(dict(BASE.PARAMS), lgb.Dataset(X, y, group=groups, feature_name=plan['features']+FEATURE_NAMES), 600)
            check_subseeds(model)
            temp = modelpath.with_suffix('.tmp'); model.save_model(str(temp)); temp.replace(modelpath)
            C.dump(seedout/'model_provenance.json', dict(provenance=provenance, params=dict(BASE.PARAMS),
                trees=600, augmentation_seed=seed, model_sha256=C.file_hash(modelpath),
                source_matrix_manifest_sha256=C.file_hash(seedout/'matrix_manifest.json')))
            del X, y, groups
            gc.collect()
            if not args.keep_matrix:
                matrix_path.unlink()
        else:
            assert modelpath.exists(), 'Evaluation requires an already trained prefix98 model'
            saved = json.loads((seedout/'model_provenance.json').read_text())
            assert saved['provenance'] == provenance and saved['params'] == BASE.PARAMS
            assert saved['model_sha256'] == C.file_hash(modelpath)
            model = lgb.Booster(model_file=str(modelpath))
            assert model.feature_name() == plan['features']+FEATURE_NAMES
            check_subseeds(model)
        evaluate(args.fold, seed, ix, out, provenance, model, heldout_ids)
        C.dump(seedout/'manifest.json', dict(fold=args.fold, augmentation_seed=seed, lgb_params=dict(BASE.PARAMS),
            explicit_seed_parameter=False, trees=600, model_sha256=C.file_hash(modelpath),
            counterpart_sha256=C.file_hash(out/f'seed{seed}/ranker.txt'), provenance=provenance,
            seconds=time.time()-start, training_row_cap=None, submission_created=False))
    aggregate(args.fold)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, default=2)
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--mode', choices=['preflight', 'features', 'train', 'evaluate', 'aggregate'], default='preflight')
    p.add_argument('--keep-matrix', action='store_true')
    main(p.parse_args())
