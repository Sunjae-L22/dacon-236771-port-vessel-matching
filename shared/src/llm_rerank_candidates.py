# author: sol
"""Reuse the train-role baseline to export natural top-20 LLM reranking pairs.

No fitting occurs here. The existing ranker and its train-role-only LLM dictionary
are reused. Stress functions come from opus_stress2/3/4; target labels are used
only after retrieval/scoring to write evaluation columns.

Run: PYTHONHASHSEED=0 python3 shared/src/llm_rerank_candidates.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KO_DICT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "shared" / "experiments" / "pinned" / "build31"
BASE = ROOT / "shared" / "experiments" / "byt5_baseline"
OUT = ROOT / "shared" / "experiments" / "llm_rerank"
sys.path.insert(0, str(CODE))
os.environ.setdefault("CACHE_DIR", str(ROOT / "shared" / "experiments" / "cache"))
import cv
import opus_lib as L
import opus_cv  # pin dependencies before importing the shared stress generators


def stress_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "shared" / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S2 = stress_module("opus_stress2")
S3 = stress_module("opus_stress3")
S4 = stress_module("opus_stress4")

VARIANTS = ["orig", "edit1", "edit2", "jamo1", "jamo2", "trunc8", "trunc60%",
            "drop_last_word", "drop_first_word", "reorder", "rename"]


def variant_rows(row, row_number, pool_en, pool_ko, seed):
    # Reset generator state per query; output does not depend on processing order.
    row_seed = seed + row_number * 1009
    edit = S3.variants(pool_en, pool_ko, seed=row_seed)
    for variant in VARIANTS:
        S2.R = random.Random(row_seed)
        S4.R = random.Random(row_seed)
        if variant in ("orig", "edit1", "edit2", "rename"):
            transformed = edit[variant](dict(row))
        elif variant in ("jamo1", "jamo2", "trunc8", "reorder"):
            # Selected operations do not use catalog name. Never give them labels.
            transformed = S4.V[variant](dict(row), "")
        else:
            transformed = S2.V[variant](dict(row))
        yield variant, transformed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT / "candidates_top20.csv")
    p.add_argument("--limit", type=int, default=0, help="Smoke check: at most this many original queries")
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--variants", nargs="*", default=VARIANTS)
    args = p.parse_args()
    if os.environ.get("KO_DICT") != "1":
        raise SystemExit("The saved baseline uses KO_DICT=1")
    if set(args.variants) - set(VARIANTS):
        raise SystemExit("Unknown variant")
    start = time.perf_counter()
    roles = pd.read_csv(ROOT / "terra" / "byt5_pilot" / "split_v1" / "rows.csv")
    train, catalog = cv.load()
    train = train.merge(roles[["sample_id", "role"]], on="sample_id", how="left", validate="one_to_one")
    assert train["role"].notna().all()
    assert train.groupby("target_vessel_id")["role"].nunique().max() == 1
    tr = train[train["role"] == "train"].drop(columns="role").reset_index(drop=True)
    ix = L.Index(catalog, tr)
    llm = L.llm_korean_rows(tr, BASE / "llm_scored_trainrole.csv")
    assert set(llm["target_vessel_id"]).issubset(set(tr["target_vessel_id"]))
    ix.ko = L.KA.KoMatcher(catalog, pd.concat([tr, llm], ignore_index=True))
    model = lgb.Booster(model_file=str(BASE / "ranker.txt"))
    assert model.feature_name() == L.FEATURES
    queries = train[train["role"].isin(["tune", "eval"]) & train["vessel_name"].map(L.is_hangul)]
    queries = queries.sort_values("sample_id").reset_index(drop=True)
    if args.limit:
        queries = queries.head(args.limit)
    pool_en = [str(n).upper() for n in catalog["vessel_name"].sample(5000, random_state=1)]
    pool_ko = sorted(set(tr.loc[tr["vessel_name"].map(L.is_hangul), "vessel_name"]))
    train_targets = set(tr["target_vessel_id"])
    rows = []
    n_queries = len(queries) * len(args.variants)
    print(f"index ready {time.perf_counter()-start:.1f}s; train={len(tr)} llm={len(llm)}; "
          f"queries={n_queries}, natural top20", flush=True)
    for row_number, original in enumerate(queries.to_dict("records")):
        for variant, row in variant_rows(original, row_number, pool_en, pool_ko, args.seed):
            if variant not in args.variants:
                continue
            # Labels are removed before feature extraction, as in real inference.
            query = {k: v for k, v in row.items() if k not in ("target_vessel_id", "role")}
            idx, features = L.query_features(ix, query, L.query_candidates(ix, query))
            scores = model.predict(features, num_threads=1)
            order = sorted(range(len(idx)), key=lambda j: (-scores[j], idx[j]))[:20]
            for rank, j in enumerate(order, 1):
                c = idx[j]
                rows.append(dict(
                    query_sample_id=f"{original['sample_id']}__{variant}",
                    orig_sample_id=original["sample_id"], variant=variant,
                    role=original["role"], vessel_name=row["vessel_name"],
                    original_vessel_name=original["vessel_name"],
                    target_vessel_id=original["target_vessel_id"], rank=rank,
                    candidate_vessel_id=ix.ids[c], candidate_vessel_name=ix.names_raw[c],
                    ranker_score=float(scores[j]), is_target=ix.ids[c] == original["target_vessel_id"],
                    cand_is_train_role_target=ix.ids[c] in train_targets,
                    name_changed=row["vessel_name"] != original["vessel_name"], n_candidates=len(idx),
                ))
        if (row_number + 1) % 20 == 0 or row_number + 1 == len(queries):
            elapsed = time.perf_counter() - start
            print(f"original queries {row_number+1}/{len(queries)}; {len(rows)} pairs; {elapsed:.1f}s", flush=True)
    result = pd.DataFrame(rows)
    assert not result.duplicated(["query_sample_id", "rank"]).any()
    assert not result.duplicated(["query_sample_id", "candidate_vessel_id"]).any()
    assert (result.groupby("query_sample_id").size() == 20).all()
    summary = result.groupby(["role", "variant"], sort=False).apply(
        lambda d: pd.Series({
            "queries": d["query_sample_id"].nunique(),
            "changed_queries": d[d["rank"] == 1]["name_changed"].sum(),
            "baseline_acc1": d.loc[d["rank"] == 1, "is_target"].mean(),
            "baseline_acc1_excluding_train_role_targets": d.loc[~d["cand_is_train_role_target"]]
                .sort_values(["query_sample_id", "rank"]).groupby("query_sample_id").head(1)["is_target"].mean(),
            "recall20": d.groupby("query_sample_id")["is_target"].any().mean(),
        }), include_groups=False).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary_path = args.output.with_name(args.output.stem + "_baseline.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"saved {args.output}; {len(rows)} pairs; total {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
