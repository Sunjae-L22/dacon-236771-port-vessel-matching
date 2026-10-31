# author: opus
"""Korean-side retrieval: catalog English names rendered into Hangul offline by a large public LLM
(catalog_ko/4_catalog_ko.ipynb: Qwen2.5-72B-Instruct-AWQ, greedy + 3 samples per name), so a Korean
record name is compared with Korean strings at the jamo level (robust to syllable typos, vowel/final
consonant confusions, spacing). The table is derived only from the official catalog.

KoCat(catalog, table_csv).search(name, k) -> [(catalog_index, similarity)], best rendering per vessel.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
        "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def jamo(s) -> str:
    s = unicodedata.normalize("NFC", str(s or ""))
    out = []
    for ch in s:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out += [CHO[o // 588], JUNG[(o % 588) // 28], JONG[o % 28]]
        elif ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


class KoCat:
    def __init__(self, catalog: pd.DataFrame, table_csv):
        t = pd.read_csv(table_csv, encoding="utf-8-sig", dtype=str).fillna("")
        by_name = {}
        for r in t.itertuples(index=False):
            vs = [v for v in (r.ko0, r.ko1, r.ko2, r.ko3) if v and any("가" <= c <= "힣" for c in v)]
            by_name[r.vessel_name] = list(dict.fromkeys(jamo(v) for v in vs if jamo(v)))
        self.strings, self.owner = [], []
        for i, n in enumerate(catalog["vessel_name"].astype(str)):
            for s in by_name.get(n, []):
                self.strings.append(s)
                self.owner.append(i)
        self.owner = np.asarray(self.owner, dtype=np.int64)
        self.by_vessel = {}
        for j, i in enumerate(self.owner.tolist()):
            self.by_vessel.setdefault(i, []).append(self.strings[j])

    def sim(self, name, idx):
        """Best jamo similarity (0..1) between the record name and each candidate's renderings (0 if none)."""
        q = jamo(name)
        out = np.zeros(len(idx), dtype=np.float32)
        if not q:
            return out
        for n, i in enumerate(idx):
            ss = self.by_vessel.get(int(i))
            if ss:
                out[n] = max(fuzz.ratio(q, s) for s in ss) / 100.0
        return out

    def search(self, name, k=100):
        q = jamo(name)
        if not q:
            return []
        sc = process.cdist([q], self.strings, scorer=fuzz.ratio, workers=1)[0]
        top = np.argpartition(-sc, min(len(sc) - 1, 4 * k))[:4 * k]
        top = top[np.lexsort((top, -sc[top]))]
        seen, out = set(), []
        for j in top:
            i = int(self.owner[j])
            if i not in seen:
                seen.add(i)
                out.append((i, float(sc[j]) / 100.0))
                if len(out) == k:
                    break
        return out
