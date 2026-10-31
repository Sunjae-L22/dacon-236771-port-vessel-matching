"""Verify GT-Window Rescue on held-out queries and build submissions/66_submit.zip from 36_submit.zip."""
from __future__ import annotations
import bisect
import csv
import math
import os
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(unicodedata.normalize("NFD", "/Users/sunjae/Desktop/항만"))
sys.path.insert(0, str(ROOT / "shared/experiments/pinned/build31"))
import ko_translit_argon as KA


def rel_gap(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        return 2.0
    return min(abs(a - b) / max(a, b, 1.0), 2.0)


def ko_rescue_score(km, qinfo: dict, qinfo_rev: dict | None, idx: int) -> float:
    sc = km._score_from_qinfo(qinfo, idx)
    comb = sc["ko_combined_score"]
    comb_rev = km._score_from_qinfo(qinfo_rev, idx)["ko_combined_score"] if qinfo_rev is not None else comb
    part_tok = (
        0.35 * sc["ko_partial_sim"]
        + 0.50 * sc["ko_token_cov"]
        + 0.15 * max(sc["ko_trans_sim"], sc["ko_en2ko_sim"])
    )
    if sc["ko_digit_match"] < 0:
        part_tok += 0.10 * sc["ko_digit_match"]
    return max(comb, comb_rev, part_tok)


def verify_heldout():
    with open(ROOT / "data/open/vessel_catalog.csv", encoding="utf-8-sig") as f:
        cat_rows = list(csv.DictReader(f))
    id2idx = {r["vessel_id"]: i for i, r in enumerate(cat_rows)}
    gt_arr = [
        float(r["gross_tonnage"]) if r["gross_tonnage"] and float(r["gross_tonnage"]) > 0 else float("nan")
        for r in cat_rows
    ]
    gt_pairs = sorted((g, i) for i, g in enumerate(gt_arr) if math.isfinite(g))
    gt_sorted = [x[0] for x in gt_pairs]
    gt_sorted_idx = [x[1] for x in gt_pairs]

    with open(ROOT / "data/open/train.csv", encoding="utf-8-sig") as f:
        tr_rows = list(csv.DictReader(f))
    id2tr = {r["sample_id"]: r for r in tr_rows}

    with open(ROOT / "shared/experiments/llm_rerank/candidates_top20.csv", encoding="utf-8-sig") as f:
        c_rows = list(csv.DictReader(f))

    heldout_sids = {r["orig_sample_id"] for r in c_rows}
    train_only = [r for r in tr_rows if r["sample_id"] not in heldout_sids]
    train_target_idxs = {id2idx[r["target_vessel_id"]] for r in train_only if r["target_vessel_id"] in id2idx}
    km_oof = KA.KoMatcher(cat_rows, train_only)

    by_q = {}
    for r in c_rows:
        qid = (
            r["role"],
            r["variant"],
            r["query_sample_id"],
            r["orig_sample_id"],
            r["vessel_name"],
            r["original_vessel_name"],
            r["target_vessel_id"],
            int(r["n_candidates"]),
        )
        by_q.setdefault(qid, []).append(r)

    base_ok = 0
    new_ok = 0
    helped = []
    hurt = []

    for qid, rlist in by_q.items():
        role, var, qsid, osid, qname, oname, tvid, ncand = qid
        if var == "rename":
            continue
        rlist.sort(key=lambda x: int(x["rank"]))
        filt = [r for r in rlist if r["cand_is_train_role_target"] != "True"] or rlist
        b_pick = filt[0]["candidate_vessel_id"]
        b_win = b_pick == tvid
        base_ok += int(b_win)

        tr = id2tr[osid]
        qgt = float(tr["gross_tonnage"] or 0)
        qigt = float(tr["international_gross_tonnage"] or 0)

        valid = [
            r
            for r in filt
            if min(rel_gap(qgt, gt_arr[id2idx[r["candidate_vessel_id"]]]), rel_gap(qigt, gt_arr[id2idx[r["candidate_vessel_id"]]]))
            <= 0.0885
        ]
        pick_r = valid[0] if valid else filt[0]
        pick_vid = pick_r["candidate_vessel_id"]
        pick_idx = id2idx[pick_vid]
        pick_lgb = float(pick_r["ranker_score"])

        qinfo = km_oof.translate_ko(qname)
        toks = qname.split()
        qinfo_rev = km_oof.translate_ko(" ".join(toks[::-1])) if len(toks) > 1 else None
        w_sc = ko_rescue_score(km_oof, qinfo, qinfo_rev, pick_idx)

        if w_sc < 0.48 and pick_lgb < -2.0:
            pool = set()
            for v in (qgt, qigt):
                if math.isfinite(v) and v > 0:
                    lo = bisect.bisect_left(gt_sorted, v * (1 - 0.0885))
                    hi = bisect.bisect_right(gt_sorted, v * (1 + 0.0885))
                    pool.update(gt_sorted_idx[lo:hi])
            pool -= train_target_idxs
            best_pool_sc = -1.0
            best_pool_idx = None
            for idx in sorted(pool):
                cgt = gt_arr[idx]
                g = min(rel_gap(qgt, cgt), rel_gap(qigt, cgt))
                if g > 0.0885:
                    continue
                psc = ko_rescue_score(km_oof, qinfo, qinfo_rev, idx) - 0.15 * g
                if psc > best_pool_sc:
                    best_pool_sc = psc
                    best_pool_idx = idx
            if best_pool_idx is not None and best_pool_sc >= 0.78 and (best_pool_sc - w_sc) >= 0.35:
                pick_vid = cat_rows[best_pool_idx]["vessel_id"]

        n_win = pick_vid == tvid
        new_ok += int(n_win)
        if n_win and not b_win:
            helped.append((var, qname, oname, cat_rows[id2idx[b_pick]]["vessel_name"], cat_rows[id2idx[pick_vid]]["vessel_name"]))
        elif b_win and not n_win:
            hurt.append((var, qname, oname, cat_rows[id2idx[b_pick]]["vessel_name"], cat_rows[id2idx[pick_vid]]["vessel_name"]))

    total_non_rename = sum(1 for qid in by_q if qid[1] != "rename")
    print(f"Non-rename queries: {total_non_rename} | base={base_ok} -> new={new_ok} | +{len(helped)} / -{len(hurt)}")
    for x in helped:
        print("  HELPED:", x)
    for x in hurt:
        print("  HURT:", x)


SCRIPT_66 = '''"""DACON inference entry point (36_submit 12-model ensemble + Tonnage Bound Guard & GT-Window KoMatcher Rescue)."""
from pathlib import Path
import os
import pickle
import sys
import unicodedata

if os.environ.get("PYTHONHASHSEED") is None:  # same predictions in every process
    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED="0"))

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
import opus_lib as L


def _ko_rescue_score(km, qinfo: dict, qinfo_rev: dict | None, idx: int) -> float:
    sc = km._score_from_qinfo(qinfo, int(idx))
    comb = sc["ko_combined_score"]
    comb_rev = km._score_from_qinfo(qinfo_rev, int(idx))["ko_combined_score"] if qinfo_rev is not None else comb
    part_tok = (
        0.35 * sc["ko_partial_sim"]
        + 0.50 * sc["ko_token_cov"]
        + 0.15 * max(sc["ko_trans_sim"], sc["ko_en2ko_sim"])
    )
    if sc["ko_digit_match"] < 0:
        part_tok += 0.10 * sc["ko_digit_match"]
    return max(comb, comb_rev, part_tok)


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    with open(HERE / "model" / "stats.pkl", "rb") as f:
        stats = pickle.load(f)
    ix = L.Index(catalog, None)
    ix.cs_map = {k: ix.id2i[v] for k, v in stats["cs_map"].items() if v in ix.id2i}
    ix.flag_p, ix.type_p = stats["flag_p"], stats["type_p"]
    ix.train_targets = set(ix.cs_map.values())
    ix.ko = L.KA.KoMatcher(catalog, stats["ko_pairs"])
    ix.fit_alias(stats["alias_rows"])
    models = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ranker*.txt"))]
    n_models = max(len(models), 1)
    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}
    preds = []
    for row in test.to_dict("records"):
        cs = ix.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix.ids[cs])
            continue
        idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
        score = sum(m.predict(F) for m in models)
        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
        non_train_j = [j for j in order if idx[j] not in train_targets] or order

        gt = L.positive(row.get("gross_tonnage"))
        igt = L.positive(row.get("international_gross_tonnage"))
        if np.isfinite(gt) or np.isfinite(igt):
            valid_gt_j = [
                j for j in non_train_j
                if min(L.rel_gap(gt, ix.gt[idx[j]]), L.rel_gap(igt, ix.gt[idx[j]])) <= 0.0885
            ]
            if valid_gt_j:
                non_train_j = valid_gt_j

        best_j = non_train_j[0]
        best = idx[best_j]
        best_avg_score = float(score[best_j]) / n_models

        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        if L.is_hangul(q) and (np.isfinite(gt) or np.isfinite(igt)):
            qinfo = ix.ko.translate_ko(q)
            toks = q.split()
            qinfo_rev = ix.ko.translate_ko(" ".join(toks[::-1])) if len(toks) > 1 else None
            w_sc = _ko_rescue_score(ix.ko, qinfo, qinfo_rev, best)
            if w_sc < 0.48 and best_avg_score < -2.0:
                pool = set()
                for v in dict.fromkeys((gt, igt)):
                    if np.isfinite(v):
                        lo, hi = np.searchsorted(ix.gt_sorted, [v * (1.0 - 0.0885), v * (1.0 + 0.0885)])
                        pool.update(ix.gt_sorted_idx[lo:hi].tolist())
                pool -= train_targets
                best_pool_sc = -1.0
                best_pool_idx = None
                for ci in sorted(pool):
                    g = min(L.rel_gap(gt, ix.gt[ci]), L.rel_gap(igt, ix.gt[ci]))
                    if g > 0.0885:
                        continue
                    psc = _ko_rescue_score(ix.ko, qinfo, qinfo_rev, ci) - 0.15 * g
                    if psc > best_pool_sc:
                        best_pool_sc = psc
                        best_pool_idx = ci
                if best_pool_idx is not None and best_pool_sc >= 0.78 and (best_pool_sc - w_sc) >= 0.35:
                    best = best_pool_idx

        preds.append(ix.ids[best])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(out, index=False)
    print(f"Saved {len(preds)} predictions to {out}")


if __name__ == "__main__":
    run()
'''


def build_66():
    src = ROOT / "submissions/36_submit.zip"
    dst = ROOT / "submissions/66_submit.zip"
    with zipfile.ZipFile(src, "r") as zin:
        items = {name: zin.read(name) for name in zin.namelist() if not name.endswith("/")}
    items["script.py"] = SCRIPT_66.encode("utf-8")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(items.keys()):
            zout.writestr(name, items[name])
    print(f"Built {dst} ({dst.stat().st_size / 1e6:.2f} MB) with {len([k for k in items if k.startswith('model/ranker')])} models.")


if __name__ == "__main__":
    verify_heldout()
    build_66()
