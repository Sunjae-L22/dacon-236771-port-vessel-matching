import copy
import json
import unittest
from challenge import challenge, route, select_pool, install_instance


class Scorer:
    """Known-logit fixture, never a GPU/model-quality measurement."""
    def __init__(self, scores):
        self.scores, self.calls = scores, []
    def score(self, row, records, beams):
        self.calls.append((copy.deepcopy(row), copy.deepcopy(records)))
        return [self.scores[r["vessel_name"]] for r in records]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.row = {"vessel_name": "가나다", "target": "NEVER", "sample_id": "row1"}
        self.ids = [f"V{i:03}" for i in range(160)]
        self.records = [{"vessel_name": x, "target_vessel_id": "SECRET"} for x in self.ids]
        self.base = [1/(i+1) for i in range(160)]
        self.sig = {k: [0.0]*160 for k in ("sim_llm", "sim_key", "sim_compact", "sim_key_word")}
        self.sig["sim_llm"][100] = 1.0
        self.votes = {"v20": "V000", "hsw": "V100", "p112": "V001", "gap110": 1.0}
        self.scores = {x: (4.0 if x == "V100" else 0.0) for x in self.ids}
    def call(self, **extra):
        args = dict(row=self.row, ids=self.ids, records=self.records, base=self.base,
                    signals=self.sig, votes=self.votes, encoder=Scorer(self.scores), qwen=Scorer(self.scores))
        args.update(extra)
        return challenge(**args)
    def test_positive_beyond_top20_and_vote_changes(self):
        x, t = self.call()
        self.assertEqual(x, "V100")
        self.assertNotIn(x, t["base_top20"])
        self.assertNotEqual(t["baseline_final"], t["candidate_final"])
        self.assertLessEqual(len(t["pool_ids"]), 128)
        self.assertLessEqual(len(t["finalist_ids"]), 9)
    def test_abstain_disagreement_and_margin(self):
        q = dict(self.scores, V000=5.0)
        self.assertEqual(self.call(qwen=Scorer(q))[1]["reason"], "view_disagreement")
        q = dict(self.scores, V100=0.5)
        self.assertEqual(self.call(qwen=Scorer(q))[1]["reason"], "insufficient_margin")
    def test_protected_english_router_pinned_disabled_no_model_calls(self):
        for row, votes, enabled in [(dict(vessel_name="English"), self.votes, True),
                                  (self.row, dict(self.votes, gap110=4), True),
                                  (self.row, dict(self.votes, hsw="V001"), True),
                                  (self.row, self.votes, False)]:
            s = Scorer(self.scores)
            self.call(row=row, votes=votes, encoder=s, qwen=s, enabled=enabled)
            self.assertEqual(s.calls, [])
    def test_labels_never_enter_inference(self):
        s = Scorer(self.scores)
        a = self.call(encoder=s, qwen=s)
        b = self.call(row=dict(self.row, target="OTHER", label="V100", truth="V100"))
        self.assertEqual(a, b)
        for row, records in s.calls:
            self.assertEqual(set(row), {"vessel_name", "gross_tonnage", "international_gross_tonnage", "vessel_type", "flag_name"})
            self.assertTrue(all("target_vessel_id" not in r for r in records))
    def test_candidate_order_invariance(self):
        p = list(reversed(range(160)))
        sig = {k: [v[i] for i in p] for k, v in self.sig.items()}
        a = self.call()
        b = self.call(ids=[self.ids[i] for i in p], records=[self.records[i] for i in p],
                      base=[self.base[i] for i in p], signals=sig)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1]["pool_ids"], b[1]["pool_ids"])
        self.assertEqual(a[1]["encoder_logits"], b[1]["encoder_logits"])
    def test_row_order_and_subset_invariance(self):
        rows = [self.row, dict(self.row, sample_id="row2", vessel_name="English")]
        def run(rs):
            return {r["sample_id"]: self.call(row=r) for r in rs}
        self.assertEqual(run(rows), run(list(reversed(rows))))
        self.assertEqual(run(rows)["row1"], run(rows[:1])["row1"])
    def test_exclusions_and_failure_are_closed(self):
        self.assertNotEqual(self.call(excluded={"V100"})[0], "V100")
        for kwargs in ({"excluded": {"V000"}}, {"votes": dict(self.votes, hsw=None)},
                       {"votes": dict(self.votes, gap110=float("nan"))},
                       {"encoder": Scorer(dict(self.scores, V100=float("nan")))}):
            with self.assertRaises(ValueError):
                self.call(**kwargs)
    def test_router_all_patterns(self):
        cases = [("A", "B", "C", "A"), ("A", "B", "B", "B"),
                 ("A", "A", "C", "A"), ("A", "B", "A", "A")]
        for a, b, c, expected in cases:
            self.assertEqual(route(self.row, dict(v20=a, hsw=b, p112=c, gap110=0)), expected)

    def test_baseline_top20_stable_ties_are_not_experimental_id_ties(self):
        from evaluate_trace import classify
        ids = [f"V{i:02}" for i in range(20, -1, -1)]
        scores = {i: (4.0 if i == 'V20' else 0.0) for i in ids}
        sig = {k: [0.0]*21 for k in self.sig}
        _, trace = self.call(ids=ids, records=[{'vessel_name': i} for i in ids],
                             base=[1/21]*21, signals=sig,
                             votes=dict(v20='V20', hsw='V01', p112='V02', gap110=1),
                             encoder=Scorer(scores), qwen=Scorer(scores))
        self.assertEqual(trace['base_top20'], ids[:20])
        self.assertNotIn('V00', trace['base_top20'])
        self.assertEqual(classify(trace, 'V00')['stage'], 'rank_below20')

    def test_install_actual_class_post_neural_exclusion_nochange_and_activation(self):
        import ast
        from pathlib import Path
        import numpy as np
        import pandas as pd
        src = Path('/Users/sunjae/Desktop/항만/astra/arena_20261101/baseline/model/ko/model/src')
        ns = {'np': np, 'Path': Path, 'is_hangul': lambda x: True}
        for filename, typename, name in [('ensemble.py', ast.FunctionDef, 'normalize'),
                                        ('pair_views.py', ast.FunctionDef, 'mix_views'),
                                        ('encoder_views.py', ast.ClassDef, 'EncoderViewsEnsemble')]:
            node = next(n for n in ast.parse((src/filename).read_text()).body
                        if isinstance(n, typename) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(src/filename), 'exec'), ns)
        class Base:
            def score_groups(self, blocks):
                return [np.array([.3,.1,.3,.3,0.0]) for _ in blocks]
        class View:
            def __init__(self, name): self.name = name
            def score(self, row, records, beams):
                # Real baseline includes X with zero base, then outer mask
                # must remove its positive neural score. Challenge excludes X.
                return np.array([20.0 if r['vessel_name'] == 'X' else
                                 4.0 if r['vessel_name'] == 'B' else 0.0 for r in records])
        class R:
            ids = np.array(['A','B','C','D','X'])
            catalog = pd.DataFrame({'vessel_name': ids})
        rows = [dict(sample_id='s', vessel_name='가나다')]
        signals = [dict(cand=np.arange(5), llm_beams=[],
                        **{k: np.zeros(5) for k in self.sig})]
        for enabled in (False, True):
            obj = ns['EncoderViewsEnsemble'](Base(), [(View('encoder:phonetic'),1),
                                                     (View('qwen:evidence'),1)], R())
            original = obj.score_records(rows, signals, [None])[0]
            self.assertEqual(R.ids[np.argmax(original)], 'X')
            masked = np.where(R.ids == 'X', -np.inf, original)
            incumbent = str(R.ids[np.argmax(masked)])
            self.assertEqual(incumbent, 'A')
            emitted = []
            install_instance(obj, {'s': dict(v20=incumbent, hsw='C', p112='D', gap110=1)},
                             {'X'}, emitted.append, enabled=enabled)
            actual = obj.score_records(rows, signals, [None])[0]
            if not enabled:
                np.testing.assert_array_equal(actual, original)
            else:
                self.assertNotIn('X', emitted[0]['pool_ids'])
                self.assertEqual(emitted[0]['candidate_v20'], 'B')
                self.assertNotEqual(emitted[0]['candidate_final'], emitted[0]['baseline_final'])
            self.assertEqual(emitted[0]['baseline_v20'], incumbent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
