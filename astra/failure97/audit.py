"""Astra: forensic audit of submission 91's Public failure; no training or edits."""
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['KO_DICT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import ast
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import pickle
import sys
import tempfile
import unicodedata
import zipfile
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def main():
    report = {}
    score = Decimal('0.9023454095'); precision = Decimal('0.0000000001')
    matches = []
    for ko in range(533):
        # Only the nearest integer English counts can possibly round to this score.
        ideal = (score - Decimal(7) * ko / Decimal(5320)) * Decimal(35680) / Decimal(3)
        for en in range(max(0, int(ideal) - 1), min(3568, int(ideal) + 1) + 1):
            actual = Decimal(7) * ko / Decimal(5320) + Decimal(3) * en / Decimal(35680)
            if actual.quantize(precision, rounding=ROUND_HALF_EVEN) == score:
                matches.append({'korean_correct': ko, 'english_correct': en, 'exact_score_decimal': str(actual), 'rounding_residual': str(actual - score)})
    report['score_decode'] = {'public_lb': str(score), 'all_integer_solutions_given_532_3568': matches, 'not_unique_from_score_alone': len(matches) != 1,
                              'if_english_unchanged_3502': {'korean_correct': 462, 'korean_change_from_79': -30, 'english_change_from_79': 0, 'score_delta_from_79': str(score - Decimal('0.9418190937'))}}
    with tempfile.TemporaryDirectory(prefix='astra_failure97_') as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(ROOT / 'submissions/91_submit.zip') as z:
            z.extractall(tmp)
            new_contents = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
        with zipfile.ZipFile(ROOT / 'submissions/79_submit.zip') as z:
            old_contents = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
        report['integrity'] = {'all_79_model_files_byte_identical': all(new_contents[n] == b for n, b in old_contents.items() if n.startswith('model/')),
                               'old_korean_models': sorted(n for n in old_contents if n.startswith('model/ranker_') and n.endswith('.txt')),
                               'new_korean_models_actually_used': ['model/ko_ranker88.txt'],
                               'english_model_files': sorted(n for n in old_contents if n.startswith('model/en65_') and n.endswith('.txt'))}
        sys.path.insert(0, str(tmp / 'model'))
        import opus_lib as L, opus_lib_65 as L65, astra_lib as A
        universal = list(L.KA._UNIVERSAL_KO_SUBWORDS)
        train = pd.read_csv(ROOT / 'data/open/train.csv'); cat = pd.read_csv(ROOT / 'data/open/vessel_catalog.csv')
        old = pickle.loads(new_contents['model/stats.pkl'])
        portable = pickle.loads(new_contents['model/ko_stats89.pkl']); new = {}
        for key, payload in portable.items():
            frame = pd.DataFrame(payload['data'], columns=payload['columns'], dtype=object)
            for column, dtype in payload['numeric_dtypes'].items():
                frame[column] = frame[column].astype(dtype)
            new[key] = frame
        ix = L.Index(cat, None)
        ix.cs_map = {k: ix.id2i[v] for k, v in old['cs_map'].items() if v in ix.id2i}
        ix.flag_p, ix.type_p = old['flag_p'], old['type_p']
        ix.train_targets = set(ix.cs_map.values())
        ix.ko = L.KA.KoMatcher(cat, old['ko_pairs']); ix.fit_alias(old['alias_rows'])
        old_fingerprint_before = digest((ix.cs_map, ix.flag_p, ix.type_p, ix.ko.exact_word_dict, ix.ko.learned_subwords))
        try:
            L.KA._UNIVERSAL_KO_SUBWORDS = []
            koix = A.Index(cat, new['fit'])
            koix.ko = L.KA.KoMatcher(cat, new['ko_pairs'])
        finally:
            L.KA._UNIVERSAL_KO_SUBWORDS = universal
        assert old_fingerprint_before == digest((ix.cs_map, ix.flag_p, ix.type_p, ix.ko.exact_word_dict, ix.ko.learned_subwords))
        assert list(ix.ids) == list(koix.ids)
        all_train = {ix.id2i[v] for v in train.target_vessel_id}
        assert ix.train_targets == all_train
        assert dict(ix.flag_p) == dict(koix.flag_p) and dict(ix.type_p) == dict(koix.type_p)
        lib65 = ast.parse(new_contents['model/opus_lib_65.py'])
        normalization = {}
        for name in ('query_candidates', 'query_features'):
            fn = next(n for n in lib65.body if isinstance(n, ast.FunctionDef) and n.name == name)
            assignments = [n for n in fn.body if isinstance(n, ast.Assign)]
            q_assignment = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == 'q' for t in n.targets))
            normalization[name] = ast.unparse(q_assignment)
            assert "normalize('NFC'" in ast.unparse(q_assignment)
        report['english_structural_equivalence'] = {
            'libraries_models_and_stats_bytes_equal': True,
            'catalog_ids_order_equal': True,
            'known_call_sign_lookup_equal': True,
            'final_train_exclusion_set_equal': True,
            'english_matcher_state_unchanged_by_new_korean_construction': True,
            'temporary_universal_dictionary_restored': L.KA._UNIVERSAL_KO_SUBWORDS == universal,
            'query_functions_already_normalize_NFC': normalization,
            'argument': 'For the same parsed row/catalog and unset KO_DICT at server launch, the English branch receives identical normalized names/other fields, same index state, same candidate/feature code, same six model files, same score summation order, same tie break and same excluded IDs. The new upfront NFC is idempotent.',
            'remaining_native_code_assumption': 'LightGBM per-row predictions are deterministic between default thread count and num_threads=1; no algorithmic score or model change.'}
        dictionary = {}
        for field, oldd, newd in [('exact_words', ix.ko.exact_word_dict, koix.ko.exact_word_dict), ('subwords', dict(ix.ko.learned_subwords), dict(koix.ko.learned_subwords))]:
            only_old = sorted(set(oldd) - set(newd)); only_new = sorted(set(newd) - set(oldd)); changed = sorted(k for k in set(oldd) & set(newd) if oldd[k] != newd[k])
            dictionary[field] = {'old_count': len(oldd), 'new_count': len(newd), 'only_old_count': len(only_old), 'only_new_count': len(only_new), 'shared_changed_count': len(changed)}
            pd.DataFrame([{'key': k, 'old': oldd.get(k), 'new': newd.get(k), 'change': 'old_only' if k in only_old else 'new_only' if k in only_new else 'value_changed'} for k in only_old + only_new + changed]).to_csv(OUT / f'dictionary_{field}_diff.csv', index=False)
        report['dictionary'] = dictionary
        report['metadata'] = {'old_and_new_flag_p_identical': True, 'old_and_new_type_p_identical': True, 'old_and_new_train_target_set_identical': True,
                               'universal_pairs_old': len(universal), 'universal_pairs_new': 0, 'old_ko_pair_rows': len(old['ko_pairs']), 'new_all_pair_rows': len(new['ko_pairs']), 'new_effective_korean_pair_rows': int(new['ko_pairs'].vessel_name.map(L.is_hangul).sum()),
                               'universal_dictionary_claim_scope': 'Its exclusion is a design change from79, not an unresolved serialization/runtime bug in91.'}
        old_models = [lgb.Booster(model_file=str(tmp / p)) for p in report['integrity']['old_korean_models']]
        model = lgb.Booster(model_file=str(tmp / 'model/ko_ranker88.txt'))
        importance = pd.DataFrame({'feature': model.feature_name(), 'split': model.feature_importance('split'), 'gain': model.feature_importance('gain')}).sort_values('gain', ascending=False)
        importance.to_csv(OUT / 'new_model_importance.csv', index=False)
        report['models'] = {'old': [{'trees': m.num_trees(), 'features': m.num_feature()} for m in old_models], 'new': {'trees': model.num_trees(), 'features': model.num_feature()}, 'new_top10_gain': importance.head(10).to_dict('records')}
        # Reconstruct training queries only to measure recipe/data coverage, never retrain.
        llm_old = L.llm_korean_rows(train, ROOT / 'shared/experiments/pinned/build70/llm_korean_scored.csv')
        llm_new = L.llm_korean_rows(train, ROOT / 'astra/full84/llm_fit.csv')
        ko = train[train.vessel_name.map(L.is_hangul)]
        llm_fit = llm_new.sample(min(2500, len(llm_new)), random_state=84)
        names = pd.concat([ko, llm_fit], ignore_index=True)
        noisy = L.augment_corrupt(names, 184, frac=.3)
        scale = np.random.default_rng(84).uniform(.75, 1.3, len(noisy))
        for c in ['gross_tonnage', 'international_gross_tonnage']:
            noisy[c] *= scale
        new_queries = pd.concat([L.augment_korean(ko, 84), llm_fit, L.augment_corrupt(names, 84, frac=.7), L.augment_rename(ko, 84, frac=.2), noisy], ignore_index=True)
        old_queries = pd.concat([L.augment_korean(train, 0), L.augment_corrupt(train, 0), L.augment_rename(train, 0), llm_old], ignore_index=True)
        def coverage(frame):
            return {'rows': len(frame), 'vessels': frame.target_vessel_id.nunique(), 'korean_named_rows_including_llm': int(frame.vessel_name.map(L.is_hangul).sum()), 'gt_missing_fraction': float(frame.gross_tonnage.isna().mean()), 'flags': int(frame.flag_name.nunique()), 'types': int(frame.vessel_type.nunique())}
        report['training_recipe'] = {'old_per_seed': coverage(old_queries), 'new_single_seed': coverage(new_queries), 'old_llm_rows': len(llm_old), 'new_llm_rows_available': len(llm_new), 'new_llm_rows_used_in_ranker': len(llm_fit), 'new_llm_unique_vessels_used': llm_fit.target_vessel_id.nunique(), 'new_real_korean_unique_vessels': ko.target_vessel_id.nunique(), 'caveat': 'Old recipe reconstructed for augmentation seed0; the submitted ensemble contains six independently augmented models.'}
        # Counterfactual diagnostics have no accuracy claim: true training IDs are
        # deliberately excluded exactly as in real inference, so labels are unusable.
        selected = ko.drop_duplicates('target_vessel_id').sample(40, random_state=97)
        query_rows = []
        for variant in ('orig', 'trunc_noise', 'rename'):
            for n, row in enumerate(selected.to_dict('records')):
                row = dict(row); row['call_sign'] = 'UNSEEN'; row['variant'] = variant
                if variant == 'trunc_noise':
                    row['vessel_name'] = row['vessel_name'][:max(3, int(.6 * len(row['vessel_name'])))]
                    for c in ['gross_tonnage', 'international_gross_tonnage']:
                        row[c] *= 1.2
                if variant == 'rename':
                    row['vessel_name'] = selected.iloc[(n + 1) % len(selected)].vessel_name
                query_rows.append(row)
        traces = []; keep = [A.FEATURES.index(n) for n in model.feature_name()]
        for row in query_rows:
            oldcand = L.query_candidates(ix, row); oldidx, oldF = L.query_features(ix, row, oldcand)
            newcand = A.query_candidates(koix, row); newidx, newF = A.query_features(koix, row, newcand)
            scores_old = sum(m.predict(oldF, num_threads=1) for m in old_models)
            scores_new = model.predict(newF[:, keep], num_threads=1)
            def choose(idx, scores):
                pool = [j for j, i in enumerate(idx) if i not in ix.train_targets] or list(range(len(idx)))
                win = min(pool, key=lambda j: (-float(scores[j]), idx[j]))
                return idx[win]
            oldwin, newwin = choose(oldidx, scores_old), choose(newidx, scores_new)
            # Exact full-data index, but temporary old dictionary isolates only this
            # representation change under the91 ranker; still diagnostic, no CV.
            actual_matcher = koix.ko
            koix.ko = ix.ko
            try:
                altidx, altF = A.query_features(koix, row, A.query_candidates(koix, row))
                altwin = choose(altidx, model.predict(altF[:, keep], num_threads=1))
            finally:
                koix.ko = actual_matcher
            traces.append({'sample_id': row['sample_id'], 'variant': row['variant'], 'old79_pred': ix.ids[oldwin], 'new91_pred': ix.ids[newwin], 'counterfactual_old_dictionary_pred': ix.ids[altwin], 'old_name_translation': ix.ko.translate_ko(row['vessel_name'])['trans'], 'new_name_translation': koix.ko.translate_ko(row['vessel_name'])['trans'], 'old_candidates': len(oldidx), 'new_candidates': len(newidx), 'old_winner_in_new_candidates': oldwin in newcand, 'new_winner_in_old_candidates': newwin in oldcand})
        traces = pd.DataFrame(traces); traces.to_csv(OUT / 'counterfactual_traces.csv', index=False)
        report['counterfactual_no_accuracy'] = {'rows': len(traces), '79_vs91_disagreements': int(traces.old79_pred.ne(traces.new91_pred).sum()), 'old_dictionary_swap_changes_91': int(traces.new91_pred.ne(traces.counterfactual_old_dictionary_pred).sum()), 'translation_changed': int(traces.old_name_translation.ne(traces.new_name_translation).sum()), 'old_winner_absent_from91_candidates': int((~traces.old_winner_in_new_candidates).sum()), 'caveat': 'Train-derived original/corrupt/renamed inputs with UNSEEN callsign and every train target excluded. This is pipeline sensitivity only; no accuracy or heldout generalization estimate.'}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
