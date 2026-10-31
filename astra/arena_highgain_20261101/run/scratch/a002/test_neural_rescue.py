import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock
sys.dont_write_bytecode = True
import numpy as np
import neural_rescue as nr

SOURCE = Path('/Users/sunjae/Desktop/항만/astra/arena_20261101/baseline/model/ko/model/src/encoder_views.py')


class Scores:
    def __init__(self, winner='C064', second=None, margin=9):
        self.winner, self.second, self.margin = winner, second, margin
        self.calls = []

    def score(self, row, candidates, beams):
        self.calls.append((copy.deepcopy(row), copy.deepcopy(candidates), list(beams)))
        return [self.margin if c['vessel_name'] == self.winner else
                0.5 if c['vessel_name'] == self.second else 0 for c in candidates]


def fixture(n=100):
    ids = [f'C{i:03}' for i in range(n)]
    p = dict(sample_id='S1', row={'sample_id': 'S1', 'vessel_name': '스타 오션'},
             ids=ids, base=[0.9]+[0.1/(n-1)]*(n-1), post=[1.0]+[0.0]*(n-1),
             signals={k: [1.0 if i == 'C064' else 0.0 for i in ids] for k in nr.SIGNALS},
             beams=['STAR OCEAN'], excluded=[], v20='C000', en='C001', hs='C002',
             gap=0.3, components_ok=True)
    cat = {i: {'vessel_name': i, 'gross_tonnage': 1200} for i in ids}
    return p, cat


class Checks(unittest.TestCase):
    def test_positive_outside_top20_and_skipped_base(self):
        p, cat = fixture()
        out = nr.run_one(p, cat, Scores(), Scores())
        self.assertEqual(out['candidate'], 'C064')
        self.assertTrue(out['changed'])
        self.assertGreater(out['winner_base_rank'], 20)
        self.assertLessEqual(len(out['pool']), 97)
        self.assertLessEqual(len(out['shortlist']), 29)

    def test_view_disagreement(self):
        p, cat = fixture()
        r = nr.run_one(p, cat, Scores(), Scores('C001'))
        self.assertEqual(r['reason'], 'view_disagreement')
        self.assertFalse(r['changed'])

    def test_weak_margin(self):
        p, cat = fixture()
        r = nr.run_one(p, cat, Scores(margin=0.99), Scores())
        self.assertEqual(r['reason'], 'weak_separation')

    def test_english_and_gap(self):
        for key, val in [('gap', 4), ('gap', 8), ('vessel_name', 'STAR OCEAN')]:
            p, cat = fixture()
            if key == 'vessel_name':
                p['row'][key] = val
            else:
                p[key] = val
            e, q = Scores(), Scores()
            r = nr.run_one(p, cat, e, q)
            self.assertEqual(r['candidate'], p['en'])
            self.assertFalse(e.calls or q.calls)

    def test_majority_fixed(self):
        p, cat = fixture()
        p['hs'] = p['en']
        e, q = Scores(), Scores()
        r = nr.run_one(p, cat, e, q)
        self.assertEqual(r['reason'], 'fixed_majority')
        self.assertFalse(e.calls or q.calls)

    def test_excluded_neural_favourite(self):
        p, cat = fixture()
        p['excluded'] = ['C064']
        out = nr.run_one(p, cat, Scores(), Scores())
        self.assertNotIn('C064', out['pool'])
        self.assertFalse(out['changed'])

    def test_packet_candidate_order(self):
        p, cat = fixture()
        ref = nr.run_one(p, cat, Scores(), Scores())
        order = list(range(len(p['ids'])))
        random.Random(2026).shuffle(order)
        for key in ('ids', 'base', 'post'):
            p[key] = [p[key][j] for j in order]
        p['signals'] = {k: [v[j] for j in order] for k, v in p['signals'].items()}
        self.assertEqual(ref, nr.run_one(p, cat, Scores(), Scores()))

    def test_row_order_and_subset(self):
        p, cat = fixture()
        rows = [copy.deepcopy(p) for _ in range(4)]
        for i, row in enumerate(rows):
            row['sample_id'] = row['row']['sample_id'] = str(i)
            row['gap'] = i*2
        run = lambda pp: {p['sample_id']: nr.run_one(p, cat, Scores(), Scores()) for p in pp}
        full = run(rows)
        self.assertEqual(full, run(list(reversed(rows))))
        self.assertEqual({i: full[i] for i in ['0', '2']}, run([rows[0], rows[2]]))

    def test_labels_cannot_enter_inference(self):
        for where in ('packet', 'row'):
            p, cat = fixture()
            (p if where == 'packet' else p['row'])['truth'] = 'C064'
            with self.assertRaises(ValueError):
                nr.run_one(p, cat, Scores(), Scores())

    def test_failure_aborts(self):
        p, cat = fixture()
        p['components_ok'] = False
        with self.assertRaises(ValueError):
            nr.run_one(p, cat, Scores(), Scores())
        p['components_ok'] = True
        with self.assertRaises(ValueError):
            nr.run_one(p, cat, Scores(margin=float('nan')), Scores())

    def test_all_voter_ids_required(self):
        for field in ('v20', 'en', 'hs'):
            for bad in (None, '', '  ', 0, []):
                p, cat = fixture()
                p[field] = bad
                e, q = Scores(), Scores()
                with self.assertRaises(ValueError):
                    nr.run_one(p, cat, e, q)
                self.assertFalse(e.calls or q.calls)

    def test_unknown_catalog_voters_abort_even_when_protected_or_disabled(self):
        for field in ('v20', 'en', 'hs'):
            for gap, enabled in ((0.3, True), (4, True), (0.3, False)):
                p, cat = fixture()
                p[field], p['gap'] = 'NO_SUCH_CATALOG_VESSEL', gap
                e, q = Scores(), Scores()
                with self.assertRaises(ValueError):
                    nr.run_one(p, cat, e, q, enabled=enabled)
                self.assertFalse(e.calls or q.calls)

    def test_zero_probability_mass_rejected(self):
        for field in ('base', 'post'):
            p, cat = fixture()
            p[field] = [0.0]*len(p['ids'])
            e, q = Scores(), Scores()
            with self.assertRaisesRegex(ValueError, 'probability mass'):
                nr.run_one(p, cat, e, q)
            self.assertFalse(e.calls or q.calls)
        # Positive mass on an excluded vessel cannot validate an empty allowed pool.
        p, cat = fixture()
        p['excluded']=['C064']; p['base']=[float(i=='C064') for i in p['ids']]
        with self.assertRaisesRegex(ValueError, 'probability mass'):
            nr.run_one(p, cat, Scores(), Scores())

    def test_diagnosis_rejects_zero_mass_before_arithmetic(self):
        import diagnose
        p, cat = fixture()
        result=nr.run_one(p,cat,Scores(),Scores())
        p['base']=[0.0]*len(p['ids'])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'tmp') as tmp:
            root=Path(tmp)
            (root/'packets.jsonl').write_text(json.dumps(p)+'\n')
            (root/'results.jsonl').write_text(json.dumps(result)+'\n')
            (root/'truth.csv').write_text('sample_id,target_vessel_id\nS1,C064\n')
            argv=['diagnose.py','--packets',str(root/'packets.jsonl'),'--results',str(root/'results.jsonl'),
                  '--truth',str(root/'truth.csv'),'--output',str(root/'diagnosis.json')]
            with mock.patch.object(sys,'argv',argv):
                with self.assertRaisesRegex(ValueError,'probability mass'):
                    diagnose.main()
            self.assertFalse((root/'diagnosis.json').exists())

    def test_disabled_no_change(self):
        p, cat = fixture()
        e, q = Scores(), Scores()
        r = nr.run_one(p, cat, e, q, enabled=False)
        self.assertEqual(r['baseline'], r['candidate'])
        self.assertFalse(e.calls or q.calls)

    def test_batch_bound_no_identity_fields(self):
        p, cat = fixture()
        e, q = Scores(), Scores()
        nr.run_one(p, cat, e, q)
        for _, docs, _ in e.calls+q.calls:
            self.assertLessEqual(len(docs), 8)
            self.assertTrue(all('sample_id' not in c and 'target_vessel_id' not in c for c in docs))

    def test_no_safety_claim_from_unanimity(self):
        # An adversarial confident joint error is accepted. This is the central
        # empirical risk, which tests cannot remove or rebrand as calibrated.
        p, cat = fixture()
        r = nr.run_one(p, cat, Scores(), Scores())
        self.assertNotEqual(r['candidate'], 'C000')  # if original were truth, harm

    def test_capture_executes_actual_class_unchanged(self):
        original_bytes = SOURCE.read_bytes()
        cls = next(n for n in ast.parse(original_bytes).body if isinstance(n, ast.ClassDef) and n.name == 'EncoderViewsEnsemble')
        def mix(bp, values, weights, weight):
            z = np.zeros(len(bp))
            for score, w in zip(values, weights):
                x = np.exp(np.array(score)-max(score)); z += w*x/x.sum()
            return (1-weight)*bp+weight*z/sum(weights)
        ns = dict(np=np, Path=Path, is_hangul=lambda name: nr.hangul({'vessel_name':name}), mix_views=mix)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), ns)
        Owner = ns['EncoderViewsEnsemble']
        class Base:
            def __init__(self, arr): self.arr=arr; self.calls=0
            def score_groups(self, blocks): self.calls+=1; return [self.arr.copy()]
        class Cell:
            def __init__(self, i): self.i=i
            def to_dict(self): return {'vessel_name': f'C{self.i:03}'}
        class Iloc:
            def __getitem__(self, i): return Cell(i)
        class Catalog: iloc=Iloc()
        class Retriever: ids=np.array(['C000','C001','C002']); catalog=Catalog()
        for scores in ([.9,.05,.05], [.34,.33,.33]):
            base=Base(np.array(scores)); obj=Owner(base, [(Scores('C001'),1)], Retriever())
            rows=[{'sample_id':'S','vessel_name':'스타'}]
            signals=[{'cand':np.arange(3), 'llm_beams':[], **{k:np.ones(3) for k in nr.SIGNALS}}]
            reference=obj.score_records(rows,signals,[None])
            saved=[]; restore=nr.install_capture(Owner, set(), saved.append)
            before=base.calls
            try: actual=obj.score_records(rows,signals,[None])
            finally: restore()
            np.testing.assert_array_equal(reference[0],actual[0])
            self.assertEqual(base.calls-before,1)
            self.assertEqual(saved[0]['base'],list(scores))
            self.assertEqual(saved[0]['v20'],Retriever.ids[np.argmax(reference[0])])
        self.assertEqual(hashlib.sha256(original_bytes).hexdigest(), hashlib.sha256(SOURCE.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main(verbosity=2)
