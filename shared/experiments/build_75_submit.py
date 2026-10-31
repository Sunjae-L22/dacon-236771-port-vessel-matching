"""Build submissions/75_submit.zip (Zero-Retrain Super-Ensemble combining all 3 LB-verified gains):
1. Korean (Hangul):
   - Combines 72_submit.zip (6 models with wide tonnage noise on corrupted names -> broke 491 to 492/532 on LB!)
     + 36_submit.zip (12 models -> 4-seed-bundle stability)
   - Plus Conditional GT-Window Demotion & KoMatcher Rescue (which gained +1 Ko in 66 vs 69 on LB without losing noisy-GT rows).
2. Non-Korean (English):
   - Uses 65_submit.zip (which achieved 3,502 / 3,568 in both 65 and 70 -> +9 En rows = +0.0007567 over 72_submit.zip!).
"""
from __future__ import annotations
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(unicodedata.normalize("NFD", "/Users/sunjae/Desktop/항만"))

SCRIPT_75 = '''"""DACON inference entry point (75_submit: 72+36 Korean + Conditional Rescue, 65 English)."""
from __future__ import annotations
from pathlib import Path
import os
import pickle
import sys
import unicodedata

if os.environ.get("PYTHONHASHSEED") is None:
    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED="0"))

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
import opus_lib as L
import opus_lib_65 as L65


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

    models_72 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ko72_*.txt"))]
    models_36 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ranker_*.txt"))]
    models_65 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("en65_*.txt"))]
    # 72 (6 models) weighted 2.5x + 36 (12 models) weighted 1.0x -> total effective weight 27.0
    tot_ko_weight = 2.5 * len(models_72) + 1.0 * len(models_36)
    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}
    preds = []
    for row in test.to_dict("records"):
        cs = ix.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix.ids[cs])
            continue
        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        if L.is_hangul(q):
            idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
            score = 2.5 * sum(m.predict(F) for m in models_72) + 1.0 * sum(m.predict(F) for m in models_36)
            order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
            non_train_j = [j for j in order if idx[j] not in train_targets] or order
            best_j = non_train_j[0]
            best = idx[best_j]
            best_avg_score = float(score[best_j]) / tot_ko_weight

            gt = L.positive(row.get("gross_tonnage"))
            igt = L.positive(row.get("international_gross_tonnage"))
            if np.isfinite(gt) or np.isfinite(igt):
                qinfo = ix.ko.translate_ko(q)
                toks = q.split()
                qinfo_rev = ix.ko.translate_ko(" ".join(toks[::-1])) if len(toks) > 1 else None

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

                best_gap_top1 = min(L.rel_gap(gt, ix.gt[best]), L.rel_gap(igt, ix.gt[best]))
                w_sc = _ko_rescue_score(ix.ko, qinfo, qinfo_rev, best)

                # Only demote an out-of-window Top-1 when a strong Korean match (>=0.80) exists inside ±8.85% GT window
                # and Top-1 is a 1-word query or has lower/equal rescue score
                if best_gap_top1 > 0.0885 and best_pool_sc >= 0.80 and best_pool_sc >= w_sc - 0.03 and len(toks) == 1:
                    valid_gt_j = [
                        j for j in non_train_j
                        if min(L.rel_gap(gt, ix.gt[idx[j]]), L.rel_gap(igt, ix.gt[idx[j]])) <= 0.0885
                    ]
                    if valid_gt_j:
                        best_j = valid_gt_j[0]
                        best = idx[best_j]
                        best_avg_score = float(score[best_j]) / tot_ko_weight
                        w_sc = _ko_rescue_score(ix.ko, qinfo, qinfo_rev, best)

                if w_sc < 0.48 and best_avg_score < -2.0 and best_pool_idx is not None:
                    if best_pool_sc >= 0.78 and (best_pool_sc - w_sc) >= 0.35:
                        best = best_pool_idx

            preds.append(ix.ids[best])
        else:
            idx, F65 = L65.query_features(ix, row, L65.query_candidates(ix, row))
            score = sum(m.predict(F65) for m in models_65)
            order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
            best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])
            preds.append(ix.ids[best])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(out, index=False)
    print(f"Saved {len(preds)} predictions to {out}")


if __name__ == "__main__":
    run()
'''


def main():
    with zipfile.ZipFile(ROOT / "submissions/36_submit.zip") as z36:
        items = {n: z36.read(n) for n in z36.namelist() if not n.endswith("/")}
    with zipfile.ZipFile(ROOT / "submissions/72_submit.zip") as z72:
        for i in range(6):
            items[f"model/ko72_{i}.txt"] = z72.read(f"model/ranker_{i}.txt")
    with zipfile.ZipFile(ROOT / "submissions/65_submit.zip") as z65:
        items["model/opus_lib_65.py"] = z65.read("model/opus_lib.py")
        for i in range(6):
            items[f"model/en65_{i}.txt"] = z65.read(f"model/ranker_{i}.txt")
    items["script.py"] = SCRIPT_75.encode("utf-8")
    dst = ROOT / "submissions/75_submit.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in sorted(items.keys()):
            zout.writestr(n, items[n])
    print(f"Built {dst} ({dst.stat().st_size / 1e6:.2f} MB) with 12 (36) + 6 (72) + 6 (65) = 24 models.")


if __name__ == "__main__":
    main()
