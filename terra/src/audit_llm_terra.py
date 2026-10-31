# author: terra
"""Public-train-only diagnosis of LLM v1/v2; never evaluates a submitted model.

Uses the frozen build31 library. No shared source, data, or submission is changed.
The dictionary experiment holds out vessels and keeps every prompt-example vessel
in training; augmentation scores are recomputed from this training partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[2]
PIN = ROOT / "shared/experiments/pinned/build31"
sys.path.insert(0, str(PIN))
os.environ["KO_DICT"] = "0"  # production entry point's default; keep it explicit
import cv
import opus_lib as L


def norm(s):
    return "".join(str(s).split())


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def llm_rows(train, gen):
    base = train.drop_duplicates("target_vessel_id").set_index("target_vessel_id")
    g = gen[gen.vessel_id.isin(base.index)]
    rows = base.loc[g.vessel_id].reset_index()
    rows["vessel_name"] = g.ko.to_numpy()
    rows = rows[rows.gross_tonnage.notna()].reset_index(drop=True)
    rows["sample_id"] = [f"TERRA_LLM_{i:06d}" for i in range(len(rows))]
    return rows


def recipe(train, llm):
    return pd.concat([L.augment_korean(train, 0), L.augment_corrupt(train, 0),
                      L.augment_rename(train, 0), llm], ignore_index=True)


def count_profile(df):
    c = df.groupby("target_vessel_id").size()
    return {"rows": len(df), "korean_rows": int(df.vessel_name.map(L.is_hangul).sum()),
            "korean_share": float(df.vessel_name.map(L.is_hangul).mean()),
            "vessels": len(c), "rows_per_vessel_median": float(c.median()),
            "rows_per_vessel_p90": float(c.quantile(.9)), "rows_per_vessel_max": int(c.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--stress", action="store_true", help="Also use frozen pre-existing corruption families")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train, catalog = cv.load()
    augdir = ROOT / "shared/llm_aug"
    old = pd.read_csv(augdir / "llm_korean_scored.csv")
    old["src"] = "v1_" + old.src
    new = pd.read_csv(augdir / "llm_korean_scored_v12.csv")
    assert not new.duplicated(["vessel_id", "ko"]).any()
    assert set(map(tuple, old[["vessel_id", "ko"]].values)) <= set(map(tuple, new[["vessel_id", "ko"]].values))
    fs1 = pd.read_csv(augdir / "llm_fewshot.csv")
    fs2 = pd.read_csv(augdir / "llm_fewshot_sets.csv")
    prompt_ids = set(fs1.vessel_id) | set(fs2.vessel_id)
    f = cv.folds(train)
    rows1 = llm_rows(train, old[old.score >= .70])
    rows12 = llm_rows(train, new[new.score >= .70])
    full1, full12 = recipe(train, rows1), recipe(train, rows12)
    summary = {"boundary": "Public train diagnostics only; no submitted-model accuracy or Public estimate",
               "fold": args.fold, "KO_DICT": 0,
               "data": {"train_rows": len(train), "catalog_rows": len(catalog),
                        "korean_vessels": int(train[train.vessel_name.map(L.is_hangul)].target_vessel_id.nunique())},
               "profiles": {"recipe_v1": count_profile(full1), "recipe_v12": count_profile(full12),
                            "llm_v1": count_profile(rows1), "llm_v12": count_profile(rows12)},
               "fewshot_overlap_original_fold": {
                   "v1": int(len(set(train.loc[f == args.fold, "target_vessel_id"]) & set(fs1.vessel_id))),
                   "v2": int(len(set(train.loc[f == args.fold, "target_vessel_id"]) & set(fs2.vessel_id)))},
               "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [PIN / "opus_lib.py", PIN / "ko_translit_argon.py",
                                            augdir / "llm_korean_scored.csv", augdir / "llm_korean_scored_v12.csv"]}}
    per_ship = pd.concat([full1.groupby("target_vessel_id").size().rename("v1_recipe_rows"),
                          full12.groupby("target_vessel_id").size().rename("v12_recipe_rows"),
                          rows1.groupby("target_vessel_id").size().rename("v1_llm_rows"),
                          rows12.groupby("target_vessel_id").size().rename("v12_llm_rows")], axis=1).fillna(0).astype(int)
    per_ship.to_csv(args.out / "per_vessel_counts.csv")
    source = new.assign(accepted=new.score >= .70).groupby("src").agg(
        generated_unique=("ko", "size"), mean_matcher_score=("score", "mean"),
        accepted=("accepted", "sum"), acceptance_rate=("accepted", "mean"))
    source.to_csv(args.out / "source_profile.csv")
    # Actual train spellings are a proxy, not ground truth about every generated variant.
    refs = train[train.vessel_name.map(L.is_hangul)].groupby("target_vessel_id").vessel_name.agg(list).to_dict()
    similar = new[new.vessel_id.isin(set(refs) - prompt_ids)].copy()
    similar["actual_name_similarity"] = [max(fuzz.ratio(norm(n), norm(r)) / 100 for r in refs[v])
                                          for v, n in zip(similar.vessel_id, similar.ko)]
    similar["accepted"] = similar.score >= .70
    similar.to_csv(args.out / "actual_name_similarity_rows.csv", index=False)
    proxy = similar.groupby(["src", "accepted"]).agg(rows=("ko", "size"),
          mean_similarity=("actual_name_similarity", "mean"), median_similarity=("actual_name_similarity", "median"))
    proxy.to_csv(args.out / "actual_name_similarity_summary.csv")
    print(json.dumps(summary["profiles"], ensure_ascii=False), flush=True)
    save_json(args.out / "summary.json", summary)
    if args.static_only:
        return

    # Force all prompt-example vessels into training. No held-out vessel was a prompt example.
    val_mask = (f == args.fold) & ~train.target_vessel_id.isin(prompt_ids)
    trf, va = train[~val_mask], train[val_mask & train.vessel_name.map(L.is_hangul)]
    assert not set(trf.target_vessel_id) & set(va.target_vessel_id)
    assert not set(va.target_vessel_id) & prompt_ids
    t0 = time.time()
    ix = L.Index(catalog, trf)
    print(f"index ready {time.time() - t0:.1f}s; validation {len(va)} rows", flush=True)
    g = new[new.vessel_id.isin(set(trf.target_vessel_id))].copy()
    g["fold_score"] = [float(ix.ko.score(n, ix.id2i[v])["ko_combined_score"])
                       for v, n in zip(g.vessel_id, g.ko)]
    g.to_csv(args.out / "fold_rescored_augmentation.csv", index=False)
    selected12 = g[g.fold_score >= .70]
    selected1 = selected12[selected12.src.str.startswith("v1_")]
    llm1 = llm_rows(trf, selected1)
    llm12 = llm_rows(trf, selected12)
    matchers = {"real_only": ix.ko,
                "v1": L.KA.KoMatcher(catalog, pd.concat([trf, llm1], ignore_index=True)),
                "v12": L.KA.KoMatcher(catalog, pd.concat([trf, llm12], ignore_index=True))}
    # Dictionary changes under the actual full-training recipes (diagnostic, no validation score).
    fullkm = {"real": L.KA.KoMatcher(catalog, train),
              "v1": L.KA.KoMatcher(catalog, pd.concat([train, rows1], ignore_index=True)),
              "v12": L.KA.KoMatcher(catalog, pd.concat([train, rows12], ignore_index=True))}
    dictrows = []
    keys = set().union(*(set(km.exact_word_dict) for km in fullkm.values()))
    for key in sorted(keys):
        vals = {n: km.exact_word_dict.get(key) for n, km in fullkm.items()}
        if vals["v1"] != vals["v12"]:
            dictrows.append({"korean_key": key, **vals,
                             "overrides_v1": vals["v1"] is not None,
                             "overrides_real": vals["real"] is not None and vals["v12"] != vals["real"]})
    pd.DataFrame(dictrows).to_csv(args.out / "dictionary_changes.csv", index=False)
    summary["dictionary_changes"] = {"new_or_changed": len(dictrows),
        "changed_existing_v1_keys": sum(r["overrides_v1"] for r in dictrows),
        "changed_real_keys": sum(r["overrides_real"] for r in dictrows)}
    summary["holdout"] = {"rows": len(va), "vessels": va.target_vessel_id.nunique(),
        "training_rows": len(trf), "prompt_example_vessels_forced_into_training": len(prompt_ids),
        "fold_llm_v1_rows": len(llm1), "fold_llm_v12_rows": len(llm12),
        "metric": "name-only rank inside GT/IGT +/-12% union, includes catalog distractors; not final ranker",
        "limitations": "original real names only; does not reproduce hidden test corruption; one fold"}
    result = []
    score_col = L.KA.KoMatcher.FEATURE_NAMES.index("ko_combined_score")
    if args.stress:
        import opus_stress2 as S2
        import opus_stress4 as S4
    for original in va.to_dict("records"):
        scenarios = {"orig": original}
        if args.stress:
            # Seed by sample ID, not evaluation order, and build once for all variants.
            sample_seed = int(hashlib.sha256(original["sample_id"].encode()).hexdigest()[:8], 16)
            S2.R.seed(sample_seed)
            S4.R.seed(sample_seed)
            cname = ix.names_raw[ix.id2i[original["target_vessel_id"]]]
            scenarios.update({"typo1": S2.V["typo1"](original),
                              "jamo1": S4.V["jamo1"](original, cname),
                              "reorder": S4.V["reorder"](original, cname),
                              "drop_last": S2.V["drop_last_word"](original),
                              "trunc60": S2.V["trunc60%"](original)})
        for scenario, r in scenarios.items():
            result.extend(evaluate_row(ix, matchers, r, original, scenario, score_col))
    result = pd.DataFrame(result)
    result.to_csv(args.out / "holdout_name_ranks.csv", index=False)
    summary["holdout_metrics"] = {}
    for (scenario, name), part in result.groupby(["scenario", "variant"]):
        summary["holdout_metrics"][f"{scenario}/{name}"] = {"n": len(part),
            "changed_queries": int(part.changed_query.sum()), "pool_recall": float(part.target_in_pool.mean()),
            **{f"top{k}": float((part["rank"] <= k).mean()) for k in (1, 5, 40)},
            "ship_macro_top1": float(part.assign(hit=part["rank"] <= 1).groupby("vessel_id").hit.mean().mean())}
    summary["elapsed_seconds"] = time.time() - t0
    save_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def evaluate_row(ix, matchers, r, original, scenario, score_col):
        result = []
        target = ix.id2i[r["target_vessel_id"]]
        pool = set()
        for v in (L.positive(r["gross_tonnage"]), L.positive(r["international_gross_tonnage"])):
            if np.isfinite(v):
                lo, hi = np.searchsorted(ix.gt_sorted, [v * .88, v * 1.12])
                pool.update(ix.gt_sorted_idx[lo:hi].tolist())
        pool = sorted(pool)
        for name, km in matchers.items():
            if pool:
                sc = km.score_batch(r["vessel_name"], pool)[:, score_col]
                order = sorted(range(len(pool)), key=lambda j: (-float(sc[j]), pool[j]))
                ranked = [pool[j] for j in order]
                rank = ranked.index(target) + 1 if target in ranked else None
                pred = ix.ids[ranked[0]]
            else:
                rank, pred = None, None
            result.append({"sample_id": r["sample_id"], "vessel_id": r["target_vessel_id"],
                           "query": r["vessel_name"], "scenario": scenario,
                           "changed_query": r["vessel_name"] != original["vessel_name"],
                           "variant": name, "pool_size": len(pool),
                           "target_in_pool": target in pool, "rank": rank, "prediction": pred,
                           "translation": km.translate_ko(r["vessel_name"])["trans"]})
        return result


if __name__ == "__main__":
    main()
