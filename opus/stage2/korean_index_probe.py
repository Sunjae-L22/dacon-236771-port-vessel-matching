# author: opus
"""Feasibility of a Hangul index of catalog names (paradigm A), using spellings we already have.

Queries: the real Korean names of train's 382 Korean vessels (few-shot vessels removed).
Index: one LLM Korean spelling (greedy + samples) per train vessel, 5,851 vessels, built only from
the English catalog name, i.e. every vessel is "unseen" for the matcher. Retrieval by jamo-level
similarity, optionally restricted by tonnage (+-15%). Reports top-1 / top-5 vessel accuracy.
The real catalog has ~13x more vessels, so this is an upper bound on name-only retrieval.

usage: python3 opus/stage2/korean_index_probe.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "shared/llm_aug"
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def jamo(s):
    out = []
    for c in str(s).replace(" ", ""):
        o = ord(c) - 0xAC00
        if 0 <= o < 11172:
            out += [CHO[o // 588], JUNG[o % 588 // 28], JONG[o % 28].strip()]
        else:
            out.append(c)
    return "".join(out)


def main():
    tr = pd.read_csv(ROOT / "data/open/train.csv", encoding="utf-8-sig")
    cat = pd.read_csv(ROOT / "data/open/vessel_catalog.csv", encoding="utf-8-sig").set_index("vessel_id")
    fs = set(pd.read_csv(A / "llm_fewshot.csv").vessel_id)
    ko = tr[tr.vessel_name.map(lambda s: any("가" <= c <= "힣" for c in str(s)))]
    ko = ko[~ko.target_vessel_id.isin(fs)]
    for name in ("llm_korean_aug.csv", "llm_korean_aug_kanana.csv"):
        g = pd.read_csv(A / name, encoding="utf-8-sig")
        rows = pd.concat([g[["vessel_id", c]].rename(columns={c: "ko"}) for c in ("ko_greedy", "ko_sample1", "ko_sample2")])
        rows = rows.dropna().drop_duplicates()
        keys = rows.ko.map(jamo).tolist()
        vids = rows.vessel_id.to_numpy()
        gt = cat.gross_tonnage.reindex(vids).to_numpy(dtype=float)
        top1 = top5 = top1_t = 0
        for r in ko.itertuples():
            res = process.extract(jamo(r.vessel_name), keys, scorer=fuzz.ratio, limit=200)
            ranked = list(dict.fromkeys(vids[i] for _, _, i in res))
            top1 += ranked[0] == r.target_vessel_id
            top5 += r.target_vessel_id in ranked[:5]
            q = r.gross_tonnage
            if pd.notna(q):
                ok = [vids[i] for _, _, i in res if not np.isfinite(gt[i]) or abs(gt[i] / q - 1) <= 0.15]
                ranked = list(dict.fromkeys(ok)) or ranked
            top1_t += ranked[0] == r.target_vessel_id
        n = len(ko)
        print(f"{name:28s} rows {n}  top1 {top1 / n:.3f}  top5 {top5 / n:.3f}  top1+tonnage {top1_t / n:.3f}")


if __name__ == "__main__":
    main()
