"""Author: Astra. Canonical NFC/NFD equivalence diagnostic, no model training."""
import os
os.environ['KO_DICT'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import sys
import json
import time
import unicodedata
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'shared/experiments/pinned/build84_astra'))
sys.path.insert(0, str(ROOT / 'astra'))
import cv, opus_lib as B, astra_lib as A, ranker84 as R


def canonical_row(row):
    row = dict(row)
    row['vessel_name'] = unicodedata.normalize('NFC', str(row.get('vessel_name') or ''))
    return row


def main():
    started = time.time()
    fold = 0
    train, cat = cv.load(); folds = cv.folds(train)
    fit = train[folds != fold].copy()
    va = train[folds == fold].copy()
    va = va[va.vessel_name.map(B.is_hangul)].drop_duplicates('target_vessel_id')
    assert not set(fit.target_vessel_id) & set(va.target_vessel_id)
    assert not set(fit.call_sign) & set(va.call_sign)
    assert B.KA._UNIVERSAL_KO_SUBWORDS == []
    llm = B.llm_korean_rows(fit, ROOT / f'astra/ranker84_fold{fold}/llm_fit.csv')
    assert set(llm.target_vessel_id) <= set(fit.target_vessel_id)
    ix = A.Index(cat, fit)
    ix.ko = B.KA.KoMatcher(cat, pd.concat([fit, llm], ignore_index=True))
    model = lgb.Booster(model_file=str(ROOT / f'astra/rankless87_fold{fold}/ranker.txt'))
    keep = [A.FEATURES.index(n) for n in model.feature_name()]
    assert len(keep) == 70
    train_ids = set(fit.target_vessel_id)
    rows = []
    details = []
    probes = [r for r in R.queries(va, fit) if r['variant'] in ('orig', 'trunc_noise')]
    for number, row in enumerate(probes):
        nfc = canonical_row(row)
        nfd = dict(row, vessel_name=unicodedata.normalize('NFD', nfc['vessel_name']))
        fixed_nfc = canonical_row(nfc)
        fixed_nfd = canonical_row(nfd)
        assert fixed_nfc == fixed_nfd
        outputs = {}
        for encoding, query in [('nfc', nfc), ('nfd', nfd), ('fixed_nfc', fixed_nfc), ('fixed_nfd', fixed_nfd)]:
            cand = A.query_candidates(ix, query)
            idx, X = A.query_features(ix, query, cand)
            X = X[:, keep]
            scores = model.predict(X, num_threads=1)
            eligible = [j for j in range(len(idx)) if ix.ids[idx[j]] not in train_ids]
            eligible = eligible or list(range(len(idx)))
            winner = min(eligible, key=lambda j: (-float(scores[j]), idx[j]))
            pred = ix.ids[idx[winner]]
            outputs[encoding] = (idx, X, pred)
            rows.append(dict(sample_id=row['sample_id'], target=row['target_vessel_id'], variant=row['variant'], encoding=encoding, pred=pred, correct=pred == row['target_vessel_id'], recall=ix.id2i[row['target_vessel_id']] in idx, candidates=len(idx)))
        a, af, ap = outputs['nfc']; b, bf, bp = outputs['nfd']
        common = sorted(set(a) & set(b)); amap = {v: j for j, v in enumerate(a)}; bmap = {v: j for j, v in enumerate(b)}
        eq = np.isclose(af[[amap[i] for i in common]], bf[[bmap[i] for i in common]], rtol=0, atol=0, equal_nan=True)
        for key in ('fixed_nfc', 'fixed_nfd'):
            ii, xx, pp = outputs[key]
            assert ii == a and pp == ap
            np.testing.assert_array_equal(xx, af)
        details.append(dict(sample_id=row['sample_id'], target=row['target_vessel_id'], variant=row['variant'], candidate_set_changed=set(a) != set(b), candidates_only_nfc=len(set(a) - set(b)), candidates_only_nfd=len(set(b) - set(a)), shared_candidate_feature_cells_changed=int((~eq).sum()), prediction_changed=ap != bp, nfc_correct=ap == row['target_vessel_id'], nfd_correct=bp == row['target_vessel_id']))
        if (number + 1) % len(va) == 0:
            print('DONE', row['variant'], number + 1, round(time.time() - started, 1), flush=True)
    predictions = pd.DataFrame(rows); changes = pd.DataFrame(details)
    predictions.to_csv(OUT / 'predictions.csv', index=False)
    changes.to_csv(OUT / 'differences.csv', index=False)
    # The baseline predictions must reproduce the already stored fold evaluation.
    prior = pd.read_csv(ROOT / f'astra/rankless87_fold{fold}/predictions.csv')
    prior = prior[prior.variant.isin(['orig', 'trunc_noise'])].copy()
    prior['pred'] = prior.pred.map(lambda i: ix.ids[int(i)])
    match = predictions[predictions.encoding.eq('nfc')].merge(prior[['sample_id', 'variant', 'pred']], on=['sample_id', 'variant'], suffixes=('', '_prior'), validate='one_to_one')
    assert len(match) == len(probes)
    assert (match.pred == match.pred_prior).all()
    report = {
        'fold': fold, 'validation_ships': len(va), 'query_count': len(probes),
        'train_vessel_disjoint': True, 'ko_dict': 0,
        'prior_nfc_baseline_reproduced': True,
        'candidate_set_changed': int(changes.candidate_set_changed.sum()),
        'queries_shared_feature_changes': int(changes.shared_candidate_feature_cells_changed.gt(0).sum()),
        'prediction_changes': int(changes.prediction_changed.sum()),
        'nfd_regressions': int((changes.nfc_correct & ~changes.nfd_correct).sum()),
        'nfd_corrections': int((~changes.nfc_correct & changes.nfd_correct).sum()),
        'fixed_nfc_changes': 0, 'fixed_nfd_vs_nfc_candidate_feature_prediction_mismatches': 0,
        'summary': {f'{variant}/{encoding}': {'accuracy': float(group.correct.mean()), 'recall': float(group.recall.mean())} for (variant, encoding), group in predictions.groupby(['variant', 'encoding'])},
        'proposed_fix': "Normalize a copy of each row's vessel_name to NFC at the query boundary before both candidate generation and scoring.",
        'scope': 'Canonical NFC/NFD only; compatibility Jamo or spaces between Jamo are not assumed equivalent.',
        'seconds': time.time() - started,
    }
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
