import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit_baseline", ROOT / "package/audit_baseline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.payload, self.baseline = self.root/"payload", self.root/"baseline"
        (self.payload/"model/ko/model").mkdir(parents=True)
        (self.baseline/"internal").mkdir(parents=True)
        self.test, self.cat = self.root/"test.csv", self.root/"vessel_catalog.csv"
        self.csv(self.test, [dict(sample_id="K", vessel_name="가나"), dict(sample_id="E", vessel_name="English")])
        self.csv(self.cat, [dict(vessel_id=i) for i in ("A", "B", "X")])
        self.excluded = self.payload/"model/ko/model/train_vessels.json"
        self.excluded.write_text('["X"]')
        self.csv(self.baseline/"en_pred.csv", [dict(sample_id="K", target_vessel_id="A", ko_gap="1.5"),
                                               dict(sample_id="E", target_vessel_id="B", ko_gap="")])
        for tag in ("ko", "hs"):
            self.csv(self.baseline/f"{tag}_pred.csv", [dict(sample_id="K", target_vessel_id="A")])
        self.records = {
            "en": dict(sample_id="K", candidate_ids=["A", "B"], scores=[2, 1], top20_ids=["A", "B"], gap110=1.5),
            "hs": dict(sample_id="K", candidate_ids=["A", "B"], scores=[2, 1], top20_ids=["A", "B"]),
            "v20": dict(sample_id="K", candidate_ids=["A", "B", "X"], base_prob=[.5, .5, 0],
                        final_scores=[.3, .2, .5], top20_ids=["A", "B", "X"], views=[]),
        }
        for tag, record in self.records.items(): self.jl(tag, [record])

    def tearDown(self): self.tmp.cleanup()
    def csv(self, path, rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def jl(self, tag, rows):
        (self.baseline/f"internal/{tag}_candidates.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    def call(self): return module.audit(self.payload, self.test, self.baseline, self.cat)
    def test_valid_nfc_and_masked_raw_winner(self):
        self.assertEqual(self.call()["trace_rows"], dict(en=1, hs=1, v20=1))
        self.assertTrue((self.baseline/"TRACE_AUDIT.json").exists())
    def test_missing_trace_file(self):
        (self.baseline/"internal/hs_candidates.jsonl").unlink()
        with self.assertRaises(ValueError): self.call()
    def test_missing_trace_row(self):
        self.jl("v20", [])
        with self.assertRaises(ValueError): self.call()
    def test_duplicate_trace(self):
        self.jl("en", [self.records["en"]]*2)
        with self.assertRaises(ValueError): self.call()
    def test_wrong_en_winner(self):
        self.jl("en", [dict(self.records["en"], top20_ids=["B", "A"])])
        with self.assertRaises(ValueError): self.call()
    def test_wrong_hs_winner(self):
        self.jl("hs", [dict(self.records["hs"], top20_ids=["B", "A"])])
        with self.assertRaises(ValueError): self.call()
    def test_wrong_gap(self):
        self.jl("en", [dict(self.records["en"], gap110=1.6)])
        with self.assertRaises(ValueError): self.call()
    def test_wrong_v20_winner(self):
        self.jl("v20", [dict(self.records["v20"], final_scores=[.1, .4, .5])])
        with self.assertRaises(ValueError): self.call()
    def test_every_candidate_excluded_fallback(self):
        self.excluded.write_text('["A","B","X"]')
        self.csv(self.baseline/"ko_pred.csv", [dict(sample_id="K", target_vessel_id="X")])
        self.assertEqual(self.call()["state"], "PASS")
    def test_component_exact_ids_and_duplicates(self):
        for rows in ([dict(sample_id="BAD", target_vessel_id="A")],
                     [dict(sample_id="K", target_vessel_id="A")]*2):
            self.csv(self.baseline/"hs_pred.csv", rows)
            with self.assertRaises(ValueError): self.call()
    def test_v20_array_finiteness_shapes_and_sign(self):
        for key, value in (("base_prob", [1,0]), ("base_prob", [0,0,0]),
                           ("final_scores", [float("nan"), .2, .5]),
                           ("final_scores", [-.1,.6,.5])):
            self.jl("v20", [dict(self.records["v20"], **{key:value})])
            with self.assertRaises(ValueError): self.call()
    def test_v20_duplicate_and_unknown_candidates(self):
        for ids in (["A","A","X"], ["A","B","UNKNOWN"]):
            self.jl("v20", [dict(self.records["v20"], candidate_ids=ids)])
            with self.assertRaises(ValueError): self.call()
    def test_stable_top20(self):
        self.jl("v20", [dict(self.records["v20"], top20_ids=["B","A","X"])])
        with self.assertRaises(ValueError): self.call()


if __name__ == "__main__": unittest.main(verbosity=2)
