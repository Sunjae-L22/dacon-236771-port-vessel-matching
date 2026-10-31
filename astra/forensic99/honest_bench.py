"""Public-LB-Calibrated Honest Outer-Fold-2 Benchmark (ko_honest & en_honest).

Validates BOTH Korean and English on 100% unseen fold-2 vessels (fit_targets excluded at inference,
matching test.csv disjointness) with the exact Public LB corruption & wide-tonnage-noise distribution
proven by LB probes 08, 09, 10, 11, 13, 14, 69, 72, 79, 92 and our train.csv forensic audit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = "/Users/sunjae/Desktop/항만"
sys.path.insert(0, f"{ROOT}/astra/incumbent97/src")
sys.path.insert(0, f"{ROOT}/astra/super98")

import ko_translit_argon as KA
import opus_lib as L
import super_features_92 as SF92
from cv import folds

OUT_DIR = f"{ROOT}/astra/forensic99"
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = [0, 5000, 1000, 2000, 6000, 7000]


def apply_organizer_typo(s: str, n_edits: int, rng: np.random.Generator) -> str:
    """Exact 1/3 duplication, 1/3 deletion, 1/3 adjacent swap discovered in train.csv."""
    chars = list(s)
    for _ in range(n_edits):
        alpha_pos = [i for i, c in enumerate(chars) if not c.isspace()]
        if len(alpha_pos) < 3:
            break
        op = float(rng.random())
        if op < 0.34:  # char duplication (c -> cc)
            i = int(rng.choice(alpha_pos))
            chars.insert(i, chars[i])
        elif op < 0.67:  # char deletion
            i = int(rng.choice(alpha_pos))
            del chars[i]
        else:  # adjacent swap
            valid_swaps = [i for i in range(len(chars) - 1) if not chars[i].isspace() and not chars[i + 1].isspace() and chars[i] != chars[i + 1]]
            if valid_swaps:
                i = int(rng.choice(valid_swaps))
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            else:
                i = int(rng.choice(alpha_pos))
                chars.insert(i, chars[i])
    out = "".join(chars).strip()
    return out if len(out) >= 2 else s


def corrupt_ko_name(s: str, mode: str, rng: np.random.Generator) -> str:
    w = s.split()
    if mode == "typo1":
        return apply_organizer_typo(s, 1, rng)
    if mode == "typo2":
        return apply_organizer_typo(s, 2, rng)
    if mode == "trunc":
        frac = float(rng.uniform(0.42, 0.75))
        k = max(2, int(round(len(s) * frac)))
        return s[:k].strip() or s[:2]
    if mode == "drop_last":
        if len(w) > 1:
            return " ".join(w[:-1])
        return s[:max(2, len(s) - 2)]
    if mode == "drop_first":
        if len(w) > 1:
            return " ".join(w[1:])
        return s[min(len(s) - 2, 1):]
    if mode == "reorder":
        if len(w) > 1:
            return " ".join(w[::-1])
        return apply_organizer_typo(s, 1, rng)
    return s


def corrupt_en_severe(s: str, mode: str, rng: np.random.Generator) -> str:
    w = s.split()
    if mode == "typo2_3":
        k = 2 if len(s) < 12 else int(rng.choice([2, 3]))
        return apply_organizer_typo(s, k, rng)
    if mode == "mv_plus_typo":
        pref = str(rng.choice(["M/V ", "MV ", "M/T "]))
        return pref + apply_organizer_typo(s, 1, rng)
    if mode == "trunc":
        frac = float(rng.uniform(0.45, 0.72))
        return s[:max(3, int(round(len(s) * frac)))].strip()
    if mode == "drop_last":
        if len(w) > 1:
            return " ".join(w[:-1])
        return s[:max(3, len(s) - 3)]
    if mode == "drop_first":
        if len(w) > 1:
            return " ".join(w[1:])
        return apply_organizer_typo(s, 2, rng)
    if mode == "initial":
        if len(w) > 1 and w[0][0].isalpha():
            return w[0][0] + ". " + apply_organizer_typo(" ".join(w[1:]), int(rng.choice([0, 1])), rng)
        return apply_organizer_typo(s, 2, rng)
    return apply_organizer_typo(s, 2, rng)


def perturb_tonnage_wide(row: dict, rng: np.random.Generator) -> dict:
    r = dict(row)
    # Outside +-12%: either [0.76, 0.87] or [1.13, 1.27]
    if rng.random() < 0.5:
        scale = float(rng.uniform(0.76, 0.87))
    else:
        scale = float(rng.uniform(1.13, 1.27))
    # 70% common scale on gt & igt, 30% independent wide noise
    if rng.random() < 0.70:
        if pd.notna(r.get("gross_tonnage")):
            r["gross_tonnage"] = round(float(r["gross_tonnage"]) * scale, 1)
        if pd.notna(r.get("international_gross_tonnage")):
            r["international_gross_tonnage"] = round(float(r["international_gross_tonnage"]) * scale, 1)
    else:
        scale2 = float(rng.uniform(0.78, 0.88) if rng.random() < 0.5 else rng.uniform(1.12, 1.26))
        if pd.notna(r.get("gross_tonnage")):
            r["gross_tonnage"] = round(float(r["gross_tonnage"]) * scale, 1)
        if pd.notna(r.get("international_gross_tonnage")):
            r["international_gross_tonnage"] = round(float(r["international_gross_tonnage"]) * scale2, 1)
    return r


def build_benchmarks(tr: pd.DataFrame, cat: pd.DataFrame):
    rng = np.random.default_rng(20261030)
    val = tr[tr["fold"] == 2].copy().reset_index(drop=True)
    cat_by_id = cat.set_index("vessel_id")

    val_ko = val[val["vessel_name"].map(L.is_hangul)].reset_index(drop=True)
    val_en = val[~val["vessel_name"].map(L.is_hangul)].reset_index(drop=True)

    # Load unseen fold2 LLM Korean queries
    with open(f"{ROOT}/astra/incumbent97/fold2/eval_llm_holdout_queries.json") as f:
        llm_q_raw = json.load(f)
    llm_rows = []
    for x in llm_q_raw:
        d = dict(x["row"]) if "row" in x else dict(x)
        # Korean rows in test.csv never have missing GT (Probe 14 = 0)
        if pd.isna(d.get("gross_tonnage")) or not np.isfinite(float(d.get("gross_tonnage") or np.nan)):
            cg = float(cat_by_id.loc[d["target_vessel_id"], "gross_tonnage"])
            noise = float(rng.uniform(0.96, 1.04))
            d["gross_tonnage"] = round(cg * noise, 0)
            d["international_gross_tonnage"] = round(cg * noise, 0)
        d["call_sign"] = f"UNSEEN_CS_{len(llm_rows)}"
        d["sample_id"] = f"HON_KO_LLM_{len(llm_rows)}"
        llm_rows.append(d)

    # Pool of Korean source rows across unseen fold2 ships (76 native ships + ~741 llm ships)
    ko_base_ships = []
    for _, r in val_ko.drop_duplicates("target_vessel_id").iterrows():
        d = r.to_dict()
        d["call_sign"] = f"UNSEEN_KO_{r.target_vessel_id}"
        ko_base_ships.append(d)
    seen_vids = {d["target_vessel_id"] for d in ko_base_ships}
    for d in llm_rows:
        if d["target_vessel_id"] not in seen_vids:
            seen_vids.add(d["target_vessel_id"])
            ko_base_ships.append(dict(d))

    ko_queries = []
    # 1. Clean Korean (weight 0.50): all 237 native rows + 600 unseen LLM rows
    for i, r in val_ko.iterrows():
        d = r.to_dict()
        d["call_sign"] = f"UNSEEN_NAT_{i}"
        d["sample_id"] = f"KO_CLN_NAT_{i}"
        ko_queries.append({"qid": d["sample_id"], "slice": "ko_clean_native", "bucket": "clean", "row": d})
    llm_idx = rng.choice(len(llm_rows), size=min(600, len(llm_rows)), replace=False)
    for k, idx_i in enumerate(sorted(llm_idx)):
        d = dict(llm_rows[int(idx_i)])
        d["sample_id"] = f"KO_CLN_LLM_{k}"
        ko_queries.append({"qid": d["sample_id"], "slice": "ko_clean_llm", "bucket": "clean", "row": d})

    # 2. Corrupted Korean (weight 0.35): 300 normal-tonnage + 300 wide-tonnage-noise
    modes_ko = ["typo1", "typo2", "trunc", "drop_last", "drop_first", "reorder"]
    chosen_norm = rng.choice(len(ko_base_ships), size=300, replace=False)
    for k, idx_i in enumerate(chosen_norm):
        d = dict(ko_base_ships[int(idx_i)])
        mode = modes_ko[k % len(modes_ko)]
        d["vessel_name"] = corrupt_ko_name(str(d["vessel_name"]), mode, rng)
        d["sample_id"] = f"KO_COR_NORM_{k}"
        ko_queries.append({"qid": d["sample_id"], "slice": f"ko_corrupt_norm_{mode}", "bucket": "corrupt_norm", "row": d})

    chosen_wide = rng.choice(len(ko_base_ships), size=300, replace=False)
    for k, idx_i in enumerate(chosen_wide):
        d = perturb_tonnage_wide(ko_base_ships[int(idx_i)], rng)
        mode = modes_ko[k % len(modes_ko)]
        d["vessel_name"] = corrupt_ko_name(str(d["vessel_name"]), mode, rng)
        d["sample_id"] = f"KO_COR_WIDE_{k}"
        ko_queries.append({"qid": d["sample_id"], "slice": f"ko_corrupt_wide_{mode}", "bucket": "corrupt_wide", "row": d})

    # 3. Renamed Korean (weight 0.15): 130 owner-prefix renames + 110 full renames
    ko_prefixes = ["글로비스", "하프니아", "케이엠티씨", "팬오션", "에스티아이", "현대", "동방", "써니", "골든", "퍼시픽", "오션", "씨"]
    chosen_ren = rng.choice(len(ko_base_ships), size=240, replace=False)
    for k, idx_i in enumerate(chosen_ren):
        d = dict(ko_base_ships[int(idx_i)])
        # Real PORT-MIS renames have exact or very tight GT (step 578: 10/12 exact GT!)
        cg = float(cat_by_id.loc[d["target_vessel_id"], "gross_tonnage"])
        if rng.random() < 0.72:
            d["gross_tonnage"] = cg
            d["international_gross_tonnage"] = cg
        else:
            sc = float(rng.uniform(0.96, 1.04))
            d["gross_tonnage"] = round(cg * sc, 0)
            d["international_gross_tonnage"] = round(cg * sc, 0)
        w = str(d["vessel_name"]).split()
        if k < 130 and len(w) > 1:
            new_p = str(rng.choice([p for p in ko_prefixes if p != w[0]]))
            d["vessel_name"] = new_p + " " + " ".join(w[1:])
            sl = "ko_rename_owner"
        else:
            other = ko_base_ships[int(rng.integers(len(ko_base_ships)))]
            while other["target_vessel_id"] == d["target_vessel_id"]:
                other = ko_base_ships[int(rng.integers(len(ko_base_ships)))]
            d["vessel_name"] = str(other["vessel_name"])
            sl = "ko_rename_full"
        d["sample_id"] = f"KO_REN_{k}"
        ko_queries.append({"qid": d["sample_id"], "slice": sl, "bucket": "rename", "row": d})

    # ---------------- ENGLISH BENCHMARK ----------------
    en_queries = []
    # 1. Real fold2 English rows (weight 0.78)
    for i, r in val_en.iterrows():
        d = r.to_dict()
        d["call_sign"] = f"UNSEEN_EN_REAL_{i}"
        d["sample_id"] = f"EN_REAL_{i}"
        en_queries.append({"qid": d["sample_id"], "slice": "en_real", "bucket": "clean", "row": d})

    en_base_ships = [r.to_dict() for _, r in val_en.drop_duplicates("target_vessel_id").iterrows()]
    for i, d in enumerate(en_base_ships):
        d["call_sign"] = f"UNSEEN_EN_SHIP_{i}"
        # Use clean catalog name as starting point for controlled severe corruption
        d["clean_cat_name"] = str(cat_by_id.loc[d["target_vessel_id"], "vessel_name"]).upper()

    # 2. Severe corrupted English (ratio < 90, weight 0.19): 500 queries (40% wide tonnage noise, 22% gt=NaN)
    modes_en = ["typo2_3", "mv_plus_typo", "trunc", "drop_last", "drop_first", "initial"]
    chosen_en_cor = rng.choice(len(en_base_ships), size=500, replace=False)
    for k, idx_i in enumerate(chosen_en_cor):
        d = dict(en_base_ships[int(idx_i)])
        mode = modes_en[k % len(modes_en)]
        d["vessel_name"] = corrupt_en_severe(d["clean_cat_name"], mode, rng)
        r_draw = float(rng.random())
        if r_draw < 0.22:
            d["gross_tonnage"] = np.nan
            d["international_gross_tonnage"] = np.nan
            sub_b = "corrupt_nogt"
        elif r_draw < 0.62:
            d = perturb_tonnage_wide(d, rng)
            sub_b = "corrupt_wide"
        else:
            sub_b = "corrupt_norm"
        d["sample_id"] = f"EN_COR_{k}"
        en_queries.append({"qid": d["sample_id"], "slice": f"en_{sub_b}_{mode}", "bucket": "corrupt", "row": d})

    # 3. Renamed English (weight 0.03): 120 queries
    en_owners = ["HAFNIA", "BOW", "FAIRCHEM", "GINGA", "SWAN", "NAVIG8", "STI", "TORM", "ARDMORE", "PACIFIC", "OCEAN"]
    chosen_en_ren = rng.choice(len(en_base_ships), size=120, replace=False)
    for k, idx_i in enumerate(chosen_en_ren):
        d = dict(en_base_ships[int(idx_i)])
        cg = float(cat_by_id.loc[d["target_vessel_id"], "gross_tonnage"])
        if rng.random() < 0.75:
            d["gross_tonnage"] = cg
            d["international_gross_tonnage"] = cg
        else:
            sc = float(rng.uniform(0.96, 1.04))
            d["gross_tonnage"] = round(cg * sc, 0)
            d["international_gross_tonnage"] = round(cg * sc, 0)
        w = d["clean_cat_name"].split()
        if k < 66 and len(w) > 1:
            npref = str(rng.choice([p for p in en_owners if p != w[0]]))
            d["vessel_name"] = npref + " " + " ".join(w[1:])
            sl = "en_rename_owner"
        else:
            other = en_base_ships[int(rng.integers(len(en_base_ships)))]
            while other["target_vessel_id"] == d["target_vessel_id"]:
                other = en_base_ships[int(rng.integers(len(en_base_ships)))]
            d["vessel_name"] = other["clean_cat_name"]
            sl = "en_rename_full"
        d["sample_id"] = f"EN_REN_{k}"
        en_queries.append({"qid": d["sample_id"], "slice": sl, "bucket": "rename", "row": d})

    return ko_queries, en_queries


def evaluate_baseline_vs_92(tr: pd.DataFrame, cat: pd.DataFrame, ko_queries: list, en_queries: list):
    t0 = time.time()
    tr_fit = tr[tr["fold"] != 2].reset_index(drop=True)
    fit_target_ids = set(tr_fit["target_vessel_id"].unique())

    ix = L.Index(cat, tr_fit)
    # Also fit LLM v1 pairs into KoMatcher just like incumbent97 fold2!
    llm_fit = pd.read_csv(f"{ROOT}/astra/incumbent97/fold2/llm_fit.csv")
    tr_plus_llm = pd.concat([tr_fit, llm_fit], ignore_index=True)
    ix.ko = KA.KoMatcher(cat, tr_plus_llm)

    sfe92 = SF92.SuperFeatureExtractor(ix, cat, tr_fit, L.FEATURES, stateless=True)

    # Load 6 fold2 models of incumbent97 (79_submit), Super98, and Clean18
    base_models = [lgb.Booster(model_file=f"{ROOT}/astra/incumbent97/fold2/seed{s}/ranker.txt") for s in SEEDS]
    sup98_models = [lgb.Booster(model_file=f"{ROOT}/astra/super98/fold2/seed{s}/ranker.txt") for s in SEEDS]
    cln18_models = [lgb.Booster(model_file=f"{ROOT}/astra/super98/fold2_clean18/seed{s}/ranker.txt") for s in SEEDS]

    keep_cols_cln18 = list(range(60)) + [60 + i for i in SF92.CLEAN18_KEEP_IDX]

    def run_ko(queries):
        res = []
        for qobj in queries:
            row = qobj["row"]
            target_id = row["target_vessel_id"]
            cand = L.query_candidates(ix, row)
            c_idx, F60 = L.query_features(ix, row, cand)
            c_vids = [ix.ids[i] for i in c_idx]
            in_cand = target_id in c_vids

            # Score 79_submit (6-model mean)
            sc79 = sum(m.predict(F60) for m in base_models) / len(base_models)

            # Score 92_submit (0.34*BASE6 + 0.33*SUP98_6 + 0.33*CLN18_6)
            row92 = dict(row)
            row92["vessel_name"] = SF92.compose_compat_jamo(str(row.get("vessel_name") or ""))
            F24 = sfe92.compute_query(row92, c_idx, F60)
            F84 = np.hstack([F60, F24])
            F78 = F84[:, keep_cols_cln18]
            sc_sup = sum(m.predict(F84) for m in sup98_models) / len(sup98_models)
            sc_cln = sum(m.predict(F78) for m in cln18_models) / len(cln18_models)
            sc92 = 0.34 * sc79 + 0.33 * sc_sup + 0.33 * sc_cln

            # Exclude fit_targets (train_targets post-exclusion!)
            allowed = np.array([v not in fit_target_ids for v in c_vids], dtype=bool)
            sc79_m = np.where(allowed, sc79, -1e9)
            sc92_m = np.where(allowed, sc92, -1e9)

            pred79 = c_vids[int(np.argmax(sc79_m))] if len(c_vids) else ""
            pred92 = c_vids[int(np.argmax(sc92_m))] if len(c_vids) else ""
            res.append({
                "qid": qobj["qid"],
                "bucket": qobj["bucket"],
                "slice": qobj["slice"],
                "recall": in_cand,
                "ok79": pred79 == target_id,
                "ok92": pred92 == target_id,
                "pred79": pred79,
                "pred92": pred92,
                "target": target_id,
            })
        return pd.DataFrame(res)

    def run_en(queries):
        res = []
        for qobj in queries:
            row = qobj["row"]
            target_id = row["target_vessel_id"]
            cand = L.query_candidates(ix, row)
            c_idx, F60 = L.query_features(ix, row, cand)
            c_vids = [ix.ids[i] for i in c_idx]
            in_cand = target_id in c_vids
            sc79 = sum(m.predict(F60) for m in base_models) / len(base_models)
            allowed = np.array([v not in fit_target_ids for v in c_vids], dtype=bool)
            sc79_m = np.where(allowed, sc79, -1e9)
            pred79 = c_vids[int(np.argmax(sc79_m))] if len(c_vids) else ""
            res.append({
                "qid": qobj["qid"],
                "bucket": qobj["bucket"],
                "slice": qobj["slice"],
                "recall": in_cand,
                "ok79": pred79 == target_id,
                "pred79": pred79,
                "target": target_id,
            })
        return pd.DataFrame(res)

    print(f"Scoring {len(ko_queries)} KO honest queries and {len(en_queries)} EN honest queries...")
    df_ko = run_ko(ko_queries)
    df_en = run_en(en_queries)

    print(f"\n=== KO HONEST BENCHMARK ({len(df_ko)} queries, elapsed {time.time()-t0:.1f}s) ===")
    ko_weights = {"clean": 0.50, "corrupt_norm": 0.175, "corrupt_wide": 0.175, "rename": 0.15}
    w79_ko, w92_ko, wrec_ko = 0.0, 0.0, 0.0
    for b, w in ko_weights.items():
        g = df_ko[df_ko["bucket"] == b]
        a79, a92, rec = g["ok79"].mean(), g["ok92"].mean(), g["recall"].mean()
        w79_ko += w * a79
        w92_ko += w * a92
        wrec_ko += w * rec
        diff_n = int(g["ok92"].sum()) - int(g["ok79"].sum())
        print(f"  [{b:13s}] w={w:.3f} n={len(g):4d} | recall={rec:.4f} | 79={a79:.4f} ({g.ok79.sum()}) | 92={a92:.4f} ({g.ok92.sum()}) | delta={diff_n:+d}")

    print(f"  --> WEIGHTED KO ACCURACY (LB-calibrated): 79={w79_ko:.5f} vs 92={w92_ko:.5f} (delta={w92_ko-w79_ko:+.5f})")

    print(f"\n=== EN HONEST BENCHMARK ({len(df_en)} queries) ===")
    en_weights = {"clean": 0.78, "corrupt": 0.19, "rename": 0.03}
    w79_en, wrec_en = 0.0, 0.0
    for b, w in en_weights.items():
        g = df_en[df_en["bucket"] == b]
        a79, rec = g["ok79"].mean(), g["recall"].mean()
        w79_en += w * a79
        wrec_en += w * rec
        print(f"  [{b:13s}] w={w:.3f} n={len(g):4d} | recall={rec:.4f} | 79={a79:.4f} ({g.ok79.sum()}/{len(g)})")
    for sl, g in df_en[df_en["bucket"] != "clean"].groupby("slice"):
        print(f"    slice {sl:28s}: n={len(g):3d} | recall={g.recall.mean():.3f} | 79={g.ok79.mean():.3f}")

    lb_est_79 = 0.7 * w79_ko + 0.3 * w79_en
    lb_est_92 = 0.7 * w92_ko + 0.3 * w79_en
    print(f"\n==> CALIBRATED PUBLIC LB ESTIMATE: 79={lb_est_79:.5f} (actual LB: 0.94182) | 92_ko={lb_est_92:.5f} (actual LB: 0.93558)")

    df_ko.to_csv(f"{OUT_DIR}/honest_ko_79_vs_92.csv", index=False)
    df_en.to_csv(f"{OUT_DIR}/honest_en_79.csv", index=False)


def main():
    tr = pd.read_csv(f"{ROOT}/data/open/train.csv")
    cat = pd.read_csv(f"{ROOT}/data/open/vessel_catalog.csv")
    tr["fold"] = folds(tr)
    ko_q, en_q = build_benchmarks(tr, cat)
    with open(f"{OUT_DIR}/honest_ko_queries.json", "w", encoding="utf-8") as f:
        json.dump(ko_q, f, ensure_ascii=False)
    with open(f"{OUT_DIR}/honest_en_queries.json", "w", encoding="utf-8") as f:
        json.dump(en_q, f, ensure_ascii=False)
    evaluate_baseline_vs_92(tr, cat, ko_q, en_q)


if __name__ == "__main__":
    main()
