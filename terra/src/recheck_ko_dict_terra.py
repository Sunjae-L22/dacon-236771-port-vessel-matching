# author: terra
"""Recheck selected dictionary failures with the actual submission defaults.

This is a dictionary-only mechanism check, not ranker accuracy or Public CV.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/experiments/pinned/build31"))
os.environ["KO_DICT"] = "1"
import cv
import opus_lib as L

UNIVERSAL = list(L.KA._UNIVERSAL_KO_SUBWORDS)
assert UNIVERSAL, "universal dictionary was unexpectedly disabled"
OUT = ROOT / "terra/experiments/20261027_ko_dict_recheck"
OLD = ROOT / "terra/experiments/20261027_llm_audit"


def build_rows(train, gen):
    base = train.drop_duplicates("target_vessel_id").set_index("target_vessel_id")
    g = gen[gen.vessel_id.isin(base.index)]
    rows = base.loc[g.vessel_id].reset_index()
    rows["vessel_name"] = g.ko.to_numpy()
    return rows[rows.gross_tonnage.notna()]


def inspect_zip(n):
    path = ROOT / f"submissions/{n:02d}_submit.zip"
    with ZipFile(path) as z:
        lib = z.read("model/opus_lib.py")
        ka = z.read("model/ko_translit_argon.py")
        default = re.search(r'os\.environ\.get\("KO_DICT",\s*"([01])"\)', lib.decode()).group(1)
        return {"submission": n, "KO_DICT_default": int(default),
                "library_sha256": hashlib.sha256(lib).hexdigest(),
                "matcher_sha256": hashlib.sha256(ka).hexdigest()}, pickle.loads(z.read("model/stats.pkl"))


def evaluate(ix, km, r):
    pool = set()
    for v in (L.positive(r.gross_tonnage), L.positive(r.international_gross_tonnage)):
        if np.isfinite(v):
            lo, hi = np.searchsorted(ix.gt_sorted, [v * .88, v * 1.12])
            pool.update(ix.gt_sorted_idx[lo:hi].tolist())
    pool = sorted(pool)
    score_col = L.KA.KoMatcher.FEATURE_NAMES.index("ko_combined_score")
    scores = km.score_batch(r.query_v1, pool)[:, score_col]
    ranked = [pool[j] for j in sorted(range(len(pool)), key=lambda j: (-scores[j], pool[j]))]
    target = ix.id2i[r.vessel_id]
    return {"translation": km.translate_ko(r.query_v1)["trans"],
            "rank": ranked.index(target) + 1 if target in ranked else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, ca = cv.load()
    evidence, stats = [], {}
    for n in (8, 18, 27, 29, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42):
        item, s = inspect_zip(n)
        evidence.append(item)
        if n in (33, 36, 42):
            stats[n] = s
    pd.DataFrame(evidence).to_csv(OUT / "zip_defaults.csv", index=False)
    expected_ka = hashlib.sha256((ROOT / "shared/experiments/pinned/build31/ko_translit_argon.py").read_bytes()).hexdigest()
    assert all(x["matcher_sha256"] == expected_ka for x in evidence if x["submission"] != 8)
    print("ZIP defaults checked", flush=True)

    keys = ("씨", "에스에이치", "이하이")
    fullrows = []
    for n, setting in ((33, 1), (36, 1), (42, 0), (42, 1)):
        L.KA._UNIVERSAL_KO_SUBWORDS = UNIVERSAL if setting else []
        km = L.KA.KoMatcher(ca, stats[n]["ko_pairs"])
        for q in ("씨 울프", "에스에이치 프리지아", "에스아이티씨 이하이"):
            fullrows.append({"submission_stats": n, "KO_DICT": setting, "query": q,
                             "translation": km.translate_ko(q)["trans"],
                             **{f"dictionary_{key}": km.exact_word_dict.get(key) for key in keys}})
    pd.DataFrame(fullrows).to_csv(OUT / "production_dictionary_projections.csv", index=False)
    print("production dictionary projections checked", flush=True)

    L.KA._UNIVERSAL_KO_SUBWORDS = UNIVERSAL
    gen = pd.read_csv(ROOT / "shared/llm_aug/llm_korean_scored_v12.csv")
    fs = set(pd.read_csv(ROOT / "shared/llm_aug/llm_fewshot.csv").vessel_id)
    fs |= set(pd.read_csv(ROOT / "shared/llm_aug/llm_fewshot_sets.csv").vessel_id)
    f = cv.folds(train)
    changed = pd.read_csv(OLD / "paired_changed_predictions.csv")
    cases = changed[changed.query_v1.str.contains("씨|에스에이치", regex=True)]
    cases = cases.drop_duplicates(["fold", "vessel_id", "scenario", "query_v1"])
    cases = cases.merge(train[["sample_id", "gross_tonnage", "international_gross_tonnage"]], on="sample_id", validate="many_to_one")
    results, counts = [], []
    for k, part in cases.groupby("fold"):
        trf = train[~((f == k) & ~train.target_vessel_id.isin(fs))]
        assert not set(trf.target_vessel_id) & set(part.vessel_id)
        assert not fs & set(part.vessel_id)
        ix = L.Index(ca, trf)
        g = gen[gen.vessel_id.isin(set(trf.target_vessel_id))].copy()
        g["fold_score"] = [ix.ko.score(n, ix.id2i[v])["ko_combined_score"] for v, n in zip(g.vessel_id, g.ko)]
        selected = g[g.fold_score >= .70]
        # Setting=1 and fold-local rescore throughout. The old setting=0 runs are
        # retained in OLD; they are not compared as a final score or effect size.
        for variant in ("v1", "v12"):
            selected_variant = selected[selected.src.str.startswith("v1_")] if variant == "v1" else selected
            rows = build_rows(trf, selected_variant)
            km = L.KA.KoMatcher(ca, pd.concat([trf, rows], ignore_index=True))
            counts.append({"fold": int(k), "variant": variant, "llm_rows": len(rows), "KO_DICT": 1})
            for r in part.itertuples():
                results.append({"fold": int(k), "vessel_id": r.vessel_id, "scenario": r.scenario,
                                "query": r.query_v1, "variant": variant, "KO_DICT": 1,
                                **evaluate(ix, km, r),
                                **{f"dictionary_{key}": km.exact_word_dict.get(key) for key in keys}})
        print(f"fold {k} targeted checks complete", flush=True)
    pd.DataFrame(results).to_csv(OUT / "holdout_targeted_cases.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "holdout_augmentation_counts.csv", index=False)
    summary = {"status": "COMPLETE", "boundary": "selected dictionary mechanisms only; no ranker/Public performance",
               "universal_entries": len(UNIVERSAL), "targeted_queries": len(cases),
               "heldout_vessels": cases.vessel_id.nunique(), "folds": sorted(cases.fold.unique().tolist()),
               "checks": ["packaged defaults inspected", "matcher code hash matches",
                          "KO_DICT=1 nonempty", "heldout vessels excluded from train/prompt examples",
                          "selection scores recomputed using train-fold and KO_DICT=1"],
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(results)[["fold", "query", "variant", "translation", "rank"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
