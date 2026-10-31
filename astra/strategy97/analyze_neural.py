"""Paired, vessel-clustered diagnostics for neural97/97b; never pool cohorts.

Only reads saved outer-fold predictions and training-derived evaluation metadata.
No model fit, weight selection, test data, or leaderboard-to-CV extrapolation.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(values):
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapped = values.map({True: True, False: False, 1: True, 0: False,
                         'True': True, 'False': False, 'true': True, 'false': False})
    assert mapped.notna().all(), 'Invalid boolean values'
    return mapped.astype(bool)


def paired_metric(frame, key, repeats=10000):
    """Resample target vessels, preserving every alias/variant within a vessel."""
    assert len(frame) and repeats > 0
    g = frame.groupby('target_vessel_id', sort=True)
    a = g[['base_correct', 'correct']].mean().to_numpy(dtype=float)
    sums = g[['base_correct', 'correct']].sum().to_numpy(dtype=float)
    sizes = g.size().to_numpy(dtype=float)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    macro_delta, row_delta = [], []
    for start in range(0, repeats, 256):
        ix = rng.integers(len(a), size=(min(256, repeats-start), len(a)))
        macro_delta.append((a[ix, 1] - a[ix, 0]).mean(axis=1))
        row_delta.append((sums[ix, 1] - sums[ix, 0]).sum(axis=1) / sizes[ix].sum(axis=1))
    gain = (~frame.base_correct) & frame.correct
    loss = frame.base_correct & (~frame.correct)
    ship_delta = a[:, 1]-a[:, 0]
    return dict(rows=len(frame), vessels=len(a), baseline_correct=int(frame.base_correct.sum()),
                neural_correct=int(frame.correct.sum()), corrected_rows=int(gain.sum()),
                regressed_rows=int(loss.sum()), net_correct_rows=int(gain.sum()-loss.sum()),
                changed_predictions=int((frame.pred != frame.baseline_pred).sum()),
                vessels_with_corrected_rows=int(frame.loc[gain, 'target_vessel_id'].nunique()),
                vessels_with_regressed_rows=int(frame.loc[loss, 'target_vessel_id'].nunique()),
                net_improved_vessels=int((ship_delta > 0).sum()),
                net_worsened_vessels=int((ship_delta < 0).sum()),
                baseline_row_accuracy=float(frame.base_correct.mean()),
                neural_row_accuracy=float(frame.correct.mean()),
                row_delta=float(frame.correct.mean()-frame.base_correct.mean()),
                row_delta_cluster_bootstrap95=np.quantile(np.concatenate(row_delta), [.025, .975]).tolist(),
                baseline_vessel_macro=float(a[:, 0].mean()), neural_vessel_macro=float(a[:, 1].mean()),
                vessel_macro_delta=float(ship_delta.mean()),
                vessel_macro_delta_bootstrap95=np.quantile(np.concatenate(macro_delta), [.025, .975]).tolist(),
                bootstrap_repeats=repeats)


def metadata(fold):
    baseline = ROOT/f'astra/incumbent97/fold{fold}'
    paths = [baseline/'old9_queries.json', HERE/f'benchmark_fold{fold}.csv', HERE/f'llm_holdout_fold{fold}.csv']
    frames = []
    for path in paths:
        if not path.exists():
            continue
        d = pd.DataFrame(json.loads(path.read_text())) if path.suffix == '.json' else pd.read_csv(path)
        d['metadata_file'] = str(path)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    assert d.query_id.is_unique, 'Evaluation metadata query_id must be unique'
    return d, paths


def main(args):
    run = Path(args.run_dir).resolve()
    path = Path(args.predictions).resolve() if args.predictions else run/'paired_predictions.csv'
    out = Path(args.output_dir).resolve() if args.output_dir else HERE/'neural_analysis'/run.name
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(path)
    assert raw.query_id.is_unique, 'Analyze one frozen fusion setting per invocation'
    assert raw.weight.nunique() == 1, 'Multiple fusion weights require separate analyses'
    catalog_path = ROOT/'data/open/vessel_catalog.csv'
    ids = pd.read_csv(catalog_path, usecols=['vessel_id']).vessel_id.astype(str).to_numpy()
    for column in ['target', 'pred', 'baseline_pred']:
        assert pd.api.types.is_integer_dtype(raw[column]), f'{column} must use catalog indices'
        assert raw[column].between(0, len(ids)-1).all(), 'Catalog index outside catalog'
    for column in ['correct', 'base_correct']:
        raw[column] = boolean(raw[column])
    assert np.array_equal(raw.correct, raw.pred == raw.target)
    assert np.array_equal(raw.base_correct, raw.baseline_pred == raw.target)
    assert np.array_equal(raw.gain, raw.correct.astype(int)-raw.base_correct.astype(int))
    raw['target_vessel_id'] = ids[raw.target]
    raw['pred_vessel_id'] = ids[raw.pred]
    raw['baseline_vessel_id'] = ids[raw.baseline_pred]
    meta, meta_paths = metadata(args.fold)
    assert set(raw.query_id) <= set(meta.query_id), 'Unknown evaluation query IDs'
    joined = raw.merge(meta, on='query_id', suffixes=('', '_metadata'), validate='one_to_one')
    assert joined.target_vessel_id.equals(joined.target_vessel_id_metadata)
    assert joined.variant.equals(joined.variant_metadata)
    assert joined.sample_id.equals(joined.sample_id_metadata)
    for col in ['target_vessel_id_metadata', 'variant_metadata', 'sample_id_metadata']:
        joined.drop(columns=col, inplace=True)
    joined['analysis_cohort'] = joined.cohort
    duplicate_native = (joined.cohort == 'old9') & (joined.variant == 'native_all_rows')
    joined.loc[duplicate_native, 'analysis_cohort'] = 'native_duplicate_old9'
    joined.loc[(joined.cohort == 'old9') & ~duplicate_native, 'analysis_cohort'] = 'old9_stress'

    # Verify the saved comparator against independently produced baseline CSVs.
    comparator = []
    baseline = ROOT/f'astra/incumbent97/fold{args.fold}'
    for kind in ['old9', 'benchmark', 'llm_holdout']:
        p = baseline/f'{kind}_{args.baseline_models}models_predictions.csv'
        if not p.exists():
            continue
        b = pd.read_csv(p)
        z = joined.merge(b[['query_id', 'pred']], on='query_id', suffixes=('', '_reference'), validate='one_to_one')
        assert (z.baseline_vessel_id == z.pred_reference).all(), f'Comparator mismatch: {kind}'
        comparator.append(dict(kind=kind, matched_queries=len(z), path=str(p), sha256=sha(p), mismatch=0))
    assert sum(c['matched_queries'] for c in comparator) == len(joined), 'Every baseline prediction must be verified'

    native = joined[joined.analysis_cohort == 'native']
    dup = joined[joined.analysis_cohort == 'native_duplicate_old9']
    duplicate_audit = dict(native_rows=len(native), old9_native_rows=len(dup), paired_rows=0)
    if len(native) and len(dup):
        z = native.merge(dup, on=['sample_id', 'target_vessel_id'], suffixes=('_native', '_old9'), validate='one_to_one')
        for column in ['pred', 'baseline_pred', 'correct', 'base_correct']:
            assert (z[column+'_native'] == z[column+'_old9']).all(), f'Identical native row prediction mismatch: {column}'
        assert len(z) == len(dup)
        duplicate_audit.update(paired_rows=len(z), prediction_mismatches=0, policy='Excluded duplicate old9 native rows from main cohort metrics')

    records = []
    def add(scope, label, d, duplicate=False):
        if len(d):
            records.append(dict(scope=scope, label=str(label), duplicate_or_overlapping_view=duplicate,
                                **paired_metric(d, scope+'::'+str(label), args.bootstrap)))

    for key, d in joined.groupby('analysis_cohort', sort=True):
        add('cohort', key, d, key == 'native_duplicate_old9')
        for variant, part in d.groupby('variant', sort=True):
            add('variant', key+'::'+variant, part, key == 'native_duplicate_old9')
            if 'any_effective_change' in part and part.any_effective_change.notna().all():
                changed = boolean(part.any_effective_change)
                if changed.any() and not changed.all():
                    add('variant_effectively_changed_only', key+'::'+variant, part[changed], True)
    stress = joined[joined.analysis_cohort == 'ko_stress']
    add('overlapping_view', 'ko_stress_without_compat_jamo', stress[stress.variant != 'ko_compat_jamo'], True)
    if 'priority' in stress:
        add('overlapping_view', 'ko_stress_core_only', stress[stress.priority == 'core'], True)
    llm = joined[joined.analysis_cohort == 'llm_outer2']
    for column in ['generation_source', 'fit_token_coverage']:
        if len(llm):
            assert llm[column].notna().all()
            for key, d in llm.groupby(column, sort=True):
                add('llm_'+column, key, d, True)
    llm_meta = meta[meta.cohort == 'llm_outer2']
    llm_coverage = dict(status='PRESENT' if len(llm) else 'NOT_IN_THIS_RUN',
                        available_rows=len(llm), available_vessels=llm.target_vessel_id.nunique(),
                        expected_rows=len(llm_meta), expected_vessels=llm_meta.target_vessel_id.nunique(),
                        complete=bool(len(llm) and set(llm.query_id) == set(llm_meta.query_id)),
                        label_status='LLM synthetic aliases with assumed targets, not manually validated natural names')
    sets = {k:set(d.target_vessel_id) for k,d in joined.groupby('analysis_cohort')}
    overlaps = [dict(first=a, second=b, shared_vessels=len(sets[a]&sets[b]))
                for i,a in enumerate(sorted(sets)) for b in sorted(sets)[i+1:]]
    config = {}
    for name in ['run_config.json', 'complete.json', 'matched79_outer2.json']:
        p = run/name
        if p.exists():
            x = json.loads(p.read_text())
            config[name] = {k:v for k,v in x.items() if k not in ['predictions', 'metrics']}
    report = dict(status='EXPLORATORY_PAIRED_OUTER_FOLD_ANALYSIS', outer_fold=args.fold,
                  source=str(path), source_sha256=sha(path), fusion_weight=float(raw.weight.iloc[0]),
                  baseline_models=args.baseline_models, comparator_checks=comparator,
                  catalog_sha256=sha(catalog_path),
                  metadata_sha256={str(p):sha(p) for p in meta_paths if p.exists()},
                  native_duplicate_audit=duplicate_audit, vessel_overlap=overlaps,
                  llm_coverage=llm_coverage, metrics=records, run_provenance=config,
                  limitations=[
                      'No grand total: native, LLM aliases, old9 and new stress are different estimands and have overlapping vessels.',
                      'Bootstrap unit is vessel, paired across methods; aliases and perturbations are retained together inside vessel.',
                      'Each interval is an unadjusted percentile 95% interval; many exploratory views are not independent confirmations.',
                      f'Baseline has {args.baseline_models} matched-recipe fold models; it is not the production six-model all-train79 ensemble.',
                      'Outer fold2 has already been reused for multiple experiments. This is not untouched confirmation.',
                      'Only 76 native Korean vessels underlie native and stress cohorts; repeated corruptions do not increase that sample size.',
                      'Synthetic corruption generators are not estimates of unknown public-test corruption probabilities.',
                      'Compatibility-jamo names route to English65 in production; Korean72/NN scores here are diagnostic only.',
                      'No fusion setting selected here and no Public LB or 0.98 improvement claim follows from these results.'
                  ])
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    flat = pd.DataFrame(records)
    for c in ['row_delta_cluster_bootstrap95', 'vessel_macro_delta_bootstrap95']:
        flat[c+'_low'] = flat[c].map(lambda x:x[0])
        flat[c+'_high'] = flat[c].map(lambda x:x[1])
        flat.drop(columns=c, inplace=True)
    flat.to_csv(out/'per_condition.csv', index=False)
    joined.to_csv(out/'enriched_predictions.csv', index=False)
    joined[joined.pred != joined.baseline_pred].to_csv(out/'changed_predictions.csv', index=False)
    joined[joined.correct != joined.base_correct].to_csv(out/'corrected_regressed.csv', index=False)
    print(json.dumps(dict(output=str(out), cohorts=[r for r in records if r['scope']=='cohort'],
                          llm_coverage=llm_coverage, native_duplicate_audit=duplicate_audit), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--predictions')
    p.add_argument('--output-dir')
    p.add_argument('--fold', type=int, default=2)
    p.add_argument('--baseline-models', type=int, default=2)
    p.add_argument('--bootstrap', type=int, default=10000)
    main(p.parse_args())
