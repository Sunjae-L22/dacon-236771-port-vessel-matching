# author: opus
"""Port of the user's v15-v18 idea (github ksh1004, Public Korean 471 -> 500 from retrieval alone) into the
93 Korean path: does adding wide edit-distance candidates lift recall / accuracy of our ranker?

93 already runs one global fuzzy search (consonant key vs catalog ckey_acr, top 40). Variants add, without
retraining (as the user did):
  F200 / F500 : the same Indel (fuzz.ratio) key search widened to top 200 / 500
  +rev        : the user's typo reversals (adjacent syllables swapped, doubled syllable dropped) whose key
                exactly matches a catalog key (closest tonnage, 10)
  +ksh        : the user's own phonetic key (src.hangul.hangul_key vs english_key of catalog names), top 500
Held-out Korean rows of fold FOLD, seed-0 79-recipe ranker, train targets of the fit split excluded.
Conditions include the user's noise estimate (syllable typos with tonnage removed).

usage: python3 fuzzy_probe.py FOLD   -> fuzzy/fold{K}.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "external" / "ksh"))
import incumbent97 as I  # noqa: E402
from jamo_fingerprint import name as jname  # noqa: E402
from src.hangul import english_key, hangul_key  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from collections import defaultdict  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

B = I.B
CONDS = [("clean", "keep"), ("corrupt", "keep"), ("corrupt", "null"), ("corrupt", "s15"), ("jamo2", "null")]
VARIANTS = ["base", "F200", "F500", "F500+rev", "F500+rev+ksh"]


def typo_reversals(name):
    out = []
    for i in range(len(name) - 1):
        a, b = name[i], name[i + 1]
        if a.isspace() or b.isspace():
            continue
        out.append(name[:i] + name[i + 1:] if a == b else name[:i] + b + a + name[i + 2:])
    return list(dict.fromkeys(out))


class Extra:
    def __init__(self, ix, cat):
        self.ix = ix
        self.exact = defaultdict(list)
        for i, k in enumerate(ix.ckey_acr):
            if k:
                self.exact[k].append(i)
        self.kkeys = [english_key(v).replace(" ", "") for v in cat.vessel_name]
        self.gt = cat.gross_tonnage.to_numpy(float)

    def closest(self, hits, q, k=10):
        hits = np.asarray(hits, int)
        refs = [v for v in (B.positive(q.get("gross_tonnage")), B.positive(q.get("international_gross_tonnage"))) if np.isfinite(v)]
        if len(hits) <= k or not refs:
            return hits[:k]
        gap = np.min([np.abs(self.gt[hits] - r) for r in refs], axis=0)
        return hits[np.lexsort((hits, np.where(np.isnan(gap), np.inf, gap)))[:k]]

    def add(self, cand, q, variant):
        if variant == "base":
            return cand
        cand = {k: list(v) for k, v in cand.items()}

        def put(i, r):
            e = cand.setdefault(int(i), [999, 999, 999])
            e[0] = min(e[0], r)
        name = str(q["vessel_name"])
        _, ck, _ = B.ko_keys(name)
        kk = 500 if variant.startswith("F500") else 200
        if ck:
            for r, (_, _, i) in enumerate(process.extract(ck, self.ix.ckey_acr, scorer=fuzz.ratio, limit=kk)):
                put(i, r)
        if "+rev" in variant:
            for v in typo_reversals(name):
                vk = B.ko_keys(v)[1]
                if vk and vk != ck and vk in self.exact:
                    for i in self.closest(self.exact[vk], q):
                        put(i, 0)
        if "+ksh" in variant:
            hk = hangul_key(name).replace(" ", "")
            if hk:
                for r, (_, _, i) in enumerate(process.extract(hk, self.kkeys, scorer=fuzz.ratio, limit=500)):
                    put(i, r)
        return cand


def main(fold):
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    model = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    ex = Extra(ix, cat)
    rows = va[va.vessel_name.map(B.is_hangul)]
    rng = np.random.default_rng(900 + fold)
    res = {f"{c}+{t}|{v}": [] for c, t in CONDS for v in VARIANTS}
    secs = defaultdict(float)
    for r in rows.to_dict("records"):
        base = {k: r.get(k) for k in I.MODEL_FIELDS}
        base["call_sign"] = "UNSEEN"
        t = ix.id2i[r["target_vessel_id"]]
        for nk, tk in CONDS:
            s0 = str(r["vessel_name"])
            q = dict(base, vessel_name=B._corrupt_name(s0, rng) if nk == "corrupt" else jname(nk, s0, rng))
            if tk == "s15":
                s = rng.uniform(0.85, 1.15)
                q["gross_tonnage"] = r["gross_tonnage"] * s
                q["international_gross_tonnage"] = r["international_gross_tonnage"] * s
            elif tk == "null":
                q["gross_tonnage"] = q["international_gross_tonnage"] = np.nan
            c0 = B.query_candidates(ix, q)
            for v in VARIANTS:
                t0 = time.time()
                cand = ex.add(c0, q, v)
                idx, F = B.query_features(ix, q, cand)
                sc = model.predict(F, num_threads=1)
                secs[v] += time.time() - t0
                order = sorted(range(len(idx)), key=lambda j: (-float(sc[j]), idx[j]))
                allowed = [j for j in order if idx[j] not in fit_targets] or order
                res[f"{nk}+{tk}|{v}"].append([int(t in cand), int(idx[allowed[0]] == t), len(idx)])
    dst = Path(__file__).parent / "fuzzy"
    dst.mkdir(exist_ok=True)
    (dst / f"fold{fold}.json").write_text(json.dumps({"res": res, "secs": secs, "n": len(rows)}))


if __name__ == "__main__":
    main(int(sys.argv[1]))
