# author: Sol
"""Check feature invariance using public train-derived, vessel-held-out queries.

No ranker training or performance estimate. No official test access.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ["KO_DICT"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "shared/experiments/pinned/build43"
NEW = ROOT / "shared/experiments/pinned/build56"
OUT = ROOT / "terra/experiments/gt_window56"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    t = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE))
    old = load("base43_contract_sol", BASE / "opus_lib.py")
    new = load("build56_contract_sol", NEW / "opus_lib.py")
    def definitions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    a, b = definitions(BASE / "opus_lib.py"), definitions(NEW / "opus_lib.py")
    assert set(b) - set(a) == {"_sol_window_features"}
    assert all(a[k] == b[k] for k in a if k != "query_features")
    assert new.FEATURES[:-2] == old.FEATURES
    assert new.FEATURES[-2:] == ["ko_gt_win_diff_max", "ko_gt_win_margin"]
    train = pd.read_csv(ROOT / "data/open/train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "data/open/vessel_catalog.csv", encoding="utf-8-sig")
    roles = pd.read_csv(ROOT / "terra/byt5_pilot/split_v1/rows.csv").set_index("sample_id").role
    tr = train[train.sample_id.map(roles).eq("train")].copy()
    ix = old.Index(catalog, tr)
    llm = old.llm_korean_rows(tr, ROOT / "shared/experiments/byt5_baseline/llm_scored_trainrole.csv")
    ix.ko = old.KA.KoMatcher(catalog, pd.concat([tr, llm], ignore_index=True))
    queries = pd.read_csv(ROOT / "terra/byt5_pilot/queries_v1/queries.csv")
    original_ids = queries.orig_sample_id.drop_duplicates().head(4)
    chosen = queries[queries.orig_sample_id.isin(original_ids)].copy()
    latin = train[train.sample_id.map(roles).ne("train") & ~train.vessel_name.map(old.is_hangul)].head(8)
    records = chosen.to_dict("records") + latin.to_dict("records")
    missing = dict(records[0])
    missing.update(gross_tonnage=np.nan, international_gross_tonnage=np.nan)
    records.append(missing)
    saved = []
    for row in records:
        q = {k: v for k, v in row.items() if k not in ("target_vessel_id", "role")}
        c0, c1 = old.query_candidates(ix, q), new.query_candidates(ix, q)
        assert c0 == c1, "candidate retrieval changed"
        ids0, f0 = old.query_features(ix, q, c0)
        ids1, f1 = new.query_features(ix, q, c1)
        assert ids0 == ids1
        np.testing.assert_array_equal(f0, f1[:, :-2])
        if not old.is_hangul(q["vessel_name"]):
            assert np.isnan(f1[:, -2:]).all()
        if pd.isna(q.get("gross_tonnage")) and pd.isna(q.get("international_gross_tonnage")):
            assert (f1[:, new.FEATURES.index("best_gap")] > .12).all()
            assert np.isnan(f1[:, -2:]).all(), "missing tonnage must not create a false zero-gap window"
        saved.append((q, ids1, f1))
    # Reordering/subsetting queries must not change the per-query features.
    for j in list(range(len(saved)))[::-3]:
        q, expected_ids, expected_f = saved[j]
        ids, f = new.query_features(ix, q, new.query_candidates(ix, q))
        assert ids == expected_ids
        np.testing.assert_array_equal(f, expected_f)
    edge_cases = [
        ([.9, .6, .3], [.5, .01, .02], [[np.nan, np.nan], [0, .3], [-.3, -.3]]),
        ([.7, .8], [0, .9], [[0, np.nan], [np.nan, np.nan]]),
        ([.6, .6], [.01, .02], [[0, 0], [0, 0]]),
        ([-.08, -.02], [.01, .02], [[-.06, -.06], [0, .06]]),
        ([1.1, 1.05], [.01, .02], [[0, .05], [-.05, -.05]]),
    ]
    for scores, gaps, expected in edge_cases:
        np.testing.assert_allclose(new._sol_window_features(scores, gaps), expected, equal_nan=True, atol=1e-7)
    assert new._sol_window_features([], []).shape == (0, 2)
    report = {"status": "PASS_IMPLEMENTATION_CONTRACT_ONLY", "queries": len(records),
              "public_held_out_korean_queries": len(chosen), "latin_queries": len(latin),
              "original_feature_count": len(old.FEATURES), "new_feature_count": len(new.FEATURES),
              "existing_values_and_candidate_ids_identical": True, "query_order_subset_invariant": True,
              "all_other_functions_and_classes_ast_identical": True,
              "both_tonnages_missing_produces_no_window": True,
              "edge_conditions": "outside winner, singleton, tie, negative scores, scores above one, empty candidates",
              "ranker_training": False, "CV_or_LB_accuracy_measured": False, "official_test_access": False,
              "seconds": round(time.monotonic() - t, 2)}
    (OUT / "feature_contract_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
