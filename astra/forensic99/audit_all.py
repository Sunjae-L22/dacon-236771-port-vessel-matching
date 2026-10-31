"""Comprehensive First-Principles Forensic Audit of train.csv, vessel_catalog.csv, and 79_submit errors."""
from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein, Levenshtein

ROOT = "/Users/sunjae/Desktop/항만"
sys.path.insert(0, f"{ROOT}/astra/incumbent97/src")
import ko_translit_argon as KT
import opus_lib as L
from cv import folds

OUT_DIR = f"{ROOT}/astra/forensic99"
os.makedirs(OUT_DIR, exist_ok=True)


def dedup_chars(s: str) -> str:
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def classify_en_diff(q_raw: str, c_raw: str) -> str:
    q = re.sub(r"\s+", " ", q_raw.upper().strip())
    c = re.sub(r"\s+", " ", c_raw.upper().strip())
    if q == c:
        return "exact_case_only"
    q_al = re.sub(r"[^A-Z0-9 ]", " ", q)
    q_al = re.sub(r"\s+", " ", q_al).strip()
    c_al = re.sub(r"[^A-Z0-9 ]", " ", c)
    c_al = re.sub(r"\s+", " ", c_al).strip()
    if q_al == c_al:
        return "punct_only"
    # Check M/V, MV, M/T, MT prefix
    q_nop = re.sub(r"^(M/V|MV|M/T|MT|S/S|SS)\s+", "", q).strip()
    if re.sub(r"[^A-Z0-9]", "", q_nop) == re.sub(r"[^A-Z0-9]", "", c):
        return "vessel_prefix_mv_mt"
    # Check token reorder
    if sorted(q_al.split()) == sorted(c_al.split()):
        return "token_reorder_exact"
    # Check pure character duplication
    if dedup_chars(q_al) == dedup_chars(c_al):
        return "char_duplication_only"
    # Check pure adjacent swap(s) (same character multiset!)
    if Counter(q_al) == Counter(c_al):
        dl = DamerauLevenshtein.distance(q_al, c_al)
        lv = Levenshtein.distance(q_al, c_al)
        if dl < lv:
            return f"adjacent_swap_dl{dl}"
        return "anagram_other"
    # Check pure deletion (q_al is subsequence of c_al)
    it = iter(c_al)
    if all(ch in it for ch in q_al):
        diff_len = len(c_al) - len(q_al)
        if c_al.startswith(q_al) or c_al.endswith(q_al):
            return f"trunc_or_word_drop_len{diff_len}"
        return f"char_deletion_only_k{diff_len}"
    # Check combination of deletion + duplication + swap
    dl = DamerauLevenshtein.distance(q_al, c_al)
    lv = Levenshtein.distance(q_al, c_al)
    sim = fuzz.ratio(q_al, c_al)
    if dl <= 2:
        if Counter(dedup_chars(q_al)) == Counter(dedup_chars(c_al)):
            return "swap_plus_dup"
        if dl < lv:
            return f"combo_with_swap_dl{dl}"
        return f"edit_dl{dl}_lv{lv}"
    if dl <= 4 and sim >= 75:
        return f"multi_typo_dl{dl}"
    # Check shared word (shipping line rename!)
    qt = set(q_al.split())
    ct = set(c_al.split())
    if qt & ct:
        return "partial_rename_shared_word"
    return "full_rename"


def main():
    tr = pd.read_csv(f"{ROOT}/data/open/train.csv")
    cat = pd.read_csv(f"{ROOT}/data/open/vessel_catalog.csv")
    tr["fold"] = folds(tr)
    m = tr.merge(cat, left_on="target_vessel_id", right_on="vessel_id", suffixes=("", "_cat"))
    m["is_ko"] = m["vessel_name"].map(lambda s: any("\uac00" <= c <= "\ud7a3" for c in str(s)))

    report = {}

    # 1. English edit operator breakdown
    en = m[~m["is_ko"]].copy()
    en["edit_cls"] = [classify_en_diff(str(a), str(b)) for a, b in zip(en["vessel_name"], en["vessel_name_cat"])]
    cls_counts_rows = en["edit_cls"].value_counts().to_dict()
    en_ships = en.drop_duplicates("target_vessel_id")
    cls_counts_ships = en_ships["edit_cls"].value_counts().to_dict()
    report["en_edit_classes_rows"] = cls_counts_rows
    report["en_edit_classes_ships"] = cls_counts_ships

    # Check correlation between edit_cls and gt_na / flag_na / gt_noise / arrival_year
    en["gt_na"] = en["gross_tonnage"].isna()
    en["fl_na"] = en["flag_name"].isna()
    en["gt_ratio"] = en["gross_tonnage"] / en["gross_tonnage_cat"]
    en["gt_exact"] = (en["gross_tonnage"] == en["gross_tonnage_cat"]) | (en["international_gross_tonnage"] == en["gross_tonnage_cat"])
    cls_meta = {}
    for cls, grp in en.groupby("edit_cls"):
        cls_meta[cls] = {
            "rows": int(len(grp)),
            "ships": int(grp["target_vessel_id"].nunique()),
            "gt_na_rate": round(float(grp["gt_na"].mean()), 4),
            "fl_na_rate": round(float(grp["fl_na"].mean()), 4),
            "gt_exact_rate": round(float(grp["gt_exact"].mean()), 4),
            "examples": [
                f"{r.vessel_name!r} -> {r.vessel_name_cat!r} (yr={r.arrival_year}, gt={r.gross_tonnage}/{r.gross_tonnage_cat})"
                for _, r in grp.drop_duplicates("target_vessel_id").head(5).iterrows()
            ],
        }
    report["en_class_metadata"] = cls_meta

    # 2. Check Korean train rows in detail: flag & vessel_type mapping coverage vs catalog
    ko = m[m["is_ko"]].copy()
    ko["gt_exact"] = (ko["gross_tonnage"] == ko["gross_tonnage_cat"]) | (ko["international_gross_tonnage"] == ko["gross_tonnage_cat"])
    report["ko_summary"] = {
        "rows": int(len(ko)),
        "ships": int(ko["target_vessel_id"].nunique()),
        "gt_exact_rate": round(float(ko["gt_exact"].mean()), 4),
        "gt_only_exact_rate": round(float((ko["gross_tonnage"] == ko["gross_tonnage_cat"]).mean()), 4),
        "igt_exact_when_gt_diff": round(
            float((ko.loc[ko["gross_tonnage"] != ko["gross_tonnage_cat"], "international_gross_tonnage"]
                   == ko.loc[ko["gross_tonnage"] != ko["gross_tonnage_cat"], "gross_tonnage_cat"]).mean()), 4
        ),
    }

    # Metadata coverage audit across ALL train.csv (note: BOTH Korean and English train rows have Korean flag_name & Korean vessel_type!)
    tr_flags = set(tr["flag_name"].dropna().unique())
    cat_flags = set(cat["flag_name"].dropna().unique())
    tr_types = set(tr["vessel_type"].dropna().unique())
    cat_types = set(cat["vessel_type"].dropna().unique())

    # Co-occurrence matrix in train.csv
    flag_pairs = m.dropna(subset=["flag_name", "flag_name_cat"]).groupby(["flag_name", "flag_name_cat"]).size()
    covered_cat_flags = set(m["flag_name_cat"].dropna().unique())
    uncovered_cat_flags = sorted(cat_flags - covered_cat_flags)

    covered_cat_types = set(m["vessel_type_cat"].dropna().unique())
    uncovered_cat_types = sorted(cat_types - covered_cat_types)

    report["metadata_coverage"] = {
        "train_ko_flags": len(tr_flags),
        "catalog_en_flags": len(cat_flags),
        "catalog_flags_seen_in_train_targets": len(covered_cat_flags),
        "catalog_flags_UNSEEN_in_train_targets": len(uncovered_cat_flags),
        "unseen_cat_flags_vessel_count": int(cat["flag_name"].isin(uncovered_cat_flags).sum()),
        "unseen_cat_flags_top20": cat[cat["flag_name"].isin(uncovered_cat_flags)]["flag_name"].value_counts().head(20).to_dict(),
        "train_ko_types": len(tr_types),
        "catalog_en_types": len(cat_types),
        "catalog_types_seen_in_train_targets": len(covered_cat_types),
        "catalog_types_UNSEEN_in_train_targets": len(uncovered_cat_types),
        "unseen_cat_types_vessel_count": int(cat["vessel_type"].isin(uncovered_cat_types).sum()),
        "unseen_cat_types_top25": cat[cat["vessel_type"].isin(uncovered_cat_types)]["vessel_type"].value_counts().head(25).to_dict(),
    }

    # Print sparse train vessel_types (like 여객선, 견인용예선, etc.)
    vt_dist = {}
    for vt, grp in m.groupby("vessel_type"):
        vt_dist[vt] = {
            "rows": int(len(grp)),
            "ships": int(grp["target_vessel_id"].nunique()),
            "top_cat_types": grp["vessel_type_cat"].value_counts().head(6).to_dict(),
        }
    report["train_vessel_type_to_cat"] = vt_dist

    # 3. Candidate recall audit of opus_lib.query_candidates under:
    # (a) Real train rows (with fold-isolated KoMatcher)
    # (b) Wide tonnage noise (best_gap in [0.13, 0.25]) + 2-char corruption / truncation / word-drop / rename
    tr_fit = tr[tr["fold"] != 2].reset_index(drop=True)
    val_f2 = tr[tr["fold"] == 2].reset_index(drop=True)
    idx = L.Index(cat, tr_fit)
    vid_to_cidx = idx.id2i

    miss_real_ko = 0
    miss_real_en = 0
    miss_wide20_ko = 0
    miss_wide20_en = 0
    miss_wide20_trunc_ko = 0
    miss_gtna_en = 0

    val_ko = val_f2[val_f2["vessel_name"].map(L.is_hangul)].reset_index(drop=True)
    val_en = val_f2[~val_f2["vessel_name"].map(L.is_hangul)].reset_index(drop=True)

    wide_ko_miss_examples = []
    for _, r in val_ko.iterrows():
        tid = vid_to_cidx[r["target_vessel_id"]]
        d = r.to_dict()
        cands = L.query_candidates(idx, d)
        if tid not in cands:
            miss_real_ko += 1
        # Wide tonnage noise (+18%)
        d_wide = dict(d)
        d_wide["gross_tonnage"] = float(d["gross_tonnage"]) * 1.18
        d_wide["international_gross_tonnage"] = float(d["international_gross_tonnage"]) * 1.18
        cands_w = L.query_candidates(idx, d_wide)
        if tid not in cands_w:
            miss_wide20_ko += 1
            if len(wide_ko_miss_examples) < 15:
                wide_ko_miss_examples.append((d["vessel_name"], cat.iloc[tid]["vessel_name"], float(d["gross_tonnage"])))
        # Wide tonnage noise (+18%) + last-2-char truncation
        d_wt = dict(d_wide)
        nm = str(d["vessel_name"]).strip()
        if len(nm) >= 4:
            d_wt["vessel_name"] = nm[:-2]
        cands_wt = L.query_candidates(idx, d_wt)
        if tid not in cands_wt:
            miss_wide20_trunc_ko += 1

    wide_en_miss_examples = []
    for _, r in val_en.iloc[::3].iterrows():  # ~730 rows sample for speed
        tid = vid_to_cidx[r["target_vessel_id"]]
        d = r.to_dict()
        cands = L.query_candidates(idx, d)
        if tid not in cands:
            miss_real_en += 1
        d_wide = dict(d)
        if pd.notna(d["gross_tonnage"]):
            d_wide["gross_tonnage"] = float(d["gross_tonnage"]) * 1.18
            d_wide["international_gross_tonnage"] = float(d["international_gross_tonnage"]) * 1.18
        cands_w = L.query_candidates(idx, d_wide)
        if tid not in cands_w:
            miss_wide20_en += 1
            if len(wide_en_miss_examples) < 15:
                wide_en_miss_examples.append((d["vessel_name"], cat.iloc[tid]["vessel_name"]))
        d_na = dict(d)
        d_na["gross_tonnage"] = np.nan
        d_na["international_gross_tonnage"] = np.nan
        cands_na = L.query_candidates(idx, d_na)
        if tid not in cands_na:
            miss_gtna_en += 1

    n_en_sample = len(val_en.iloc[::3])
    report["fold2_candidate_recall_audit"] = {
        "val_ko_rows": len(val_ko),
        "miss_real_ko": miss_real_ko,
        "miss_wide18pct_ko": miss_wide20_ko,
        "miss_wide18pct_trunc2_ko": miss_wide20_trunc_ko,
        "wide_ko_miss_examples": wide_ko_miss_examples,
        "val_en_sample_rows": n_en_sample,
        "miss_real_en": miss_real_en,
        "miss_wide18pct_en": miss_wide20_en,
        "miss_gtna_en": miss_gtna_en,
        "wide_en_miss_examples": wide_en_miss_examples,
    }

    # 4. Inspect 80_submit.zip and 83_submit.zip script.py
    sub_scripts = {}
    for sname in ["79_submit.zip", "80_submit.zip", "83_submit.zip", "92_submit.zip"]:
        zp = f"{ROOT}/submissions/{sname}"
        if os.path.exists(zp):
            with zipfile.ZipFile(zp) as zf:
                sub_scripts[sname] = zf.read("script.py").decode("utf-8", errors="replace")
    report["submission_scripts"] = sub_scripts

    out_fp = f"{OUT_DIR}/audit_report.json"
    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Audit saved to {out_fp}")
    print("EN edit classes (ships):", cls_counts_ships)
    print("Candidate recall audit:", json.dumps(report["fold2_candidate_recall_audit"], ensure_ascii=False, indent=2))
    print("Metadata unseen:", {
        k: report["metadata_coverage"][k]
        for k in ["unseen_cat_flags_vessel_count", "unseen_cat_types_vessel_count"]
    })


if __name__ == "__main__":
    main()
