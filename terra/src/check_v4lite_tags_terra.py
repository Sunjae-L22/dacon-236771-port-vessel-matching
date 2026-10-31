# author: terra (Sol)
"""Tiny source-routing check, without catalog indexing, learning, or inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PIN = ROOT / "shared/experiments/pinned/build49"
OUT = ROOT / "terra/experiments/20261027_v4lite_tags"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = PIN / "ko_translit_argon.py"
    code_sha = digest(p)
    spec = importlib.util.spec_from_file_location("v4lite_check", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    class Spy(mod.KoMatcher):
        def _count_pairs(self, records):
            self.seen.append([r["sample_id"] for r in records])
            return {}, {}
    records = [{"sample_id": f"TRAIN_{i:05d}"} for i in range(950)] + [
        {"sample_id": f"LLM_{i:05d}"} for i in range(302)]
    random.Random(42).shuffle(records)
    records = pickle.loads(pickle.dumps(records))
    spy = Spy.__new__(Spy)
    spy.seen = []
    spy._fit_train_pairs(records)
    assert len(spy.seen[0]) == 950 and len(spy.seen[1]) == 302
    assert all(s.startswith("TRAIN_") for s in spy.seen[0])
    assert all(s.startswith("LLM_") for s in spy.seen[1])
    make = (PIN / "make_submit.py").read_text()
    assert '["sample_id", "vessel_name", "target_vessel_id"]].to_dict("records")' in make
    assert digest(p) == code_sha
    paths = [ROOT / "submissions/36_submit.zip", ROOT / "submissions/36_submit_argon_v4.zip"]
    with zipfile.ZipFile(paths[0]) as a, zipfile.ZipFile(paths[1]) as b:
        assert set(a.namelist()) == set(b.namelist())
        changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
        assert changed == ["model/ko_translit_argon.py"]
        zip_report = {x.name: digest(x) for x in paths}
        zip_report["patched_matcher_sha256"] = hashlib.sha256(b.read(changed[0])).hexdigest()
    report = {"boundary": "Source routing and archive equality only; no accuracy, full matcher fit, GPU, or submission",
              "matcher_sha256": code_sha, "make_submit_sha256": digest(PIN / "make_submit.py"),
              "shuffled_pickle_roundtrip": {"real": len(spy.seen[0]), "llm": len(spy.seen[1])},
              "changed_members_36_patch": changed, "zip_checks": zip_report}
    OUT.mkdir(exist_ok=True)
    (OUT / "proof.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
