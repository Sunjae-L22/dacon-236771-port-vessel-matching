# author: opus
"""Fingerprint test regimes against what the Public score already told us.

Public Korean (532 rows): 93 acc .925; 102 says ~30% of rows are hard (gap<4) with ~.76 accuracy;
105 (switch hard rows to an exact-tonnage top-3 candidate when the leader is >5% off) cost 27 rows (-5.1%).
For each synthetic regime on vessel-disjoint fold data (seed-0 ranker of the 79 recipe), report the same
four numbers; a regime that matches all four is a candidate description of the test.

usage: python3 regime_fingerprint.py FOLD   (writes regime_fp_foldK.json next to this file)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
import incumbent97 as I  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

B = I.B
NAN = float("nan")
BG = B.FEATURES.index("best_gap")


def tonnage(kind, r, rng):
    gt, igt = r["gross_tonnage"], r["international_gross_tonnage"]
    if kind == "dom":
        return {"international_gross_tonnage": gt}
    if kind == "s15":
        s = rng.uniform(0.85, 1.15)
        return {"gross_tonnage": gt * s, "international_gross_tonnage": igt * s}
    if kind == "s30":
        s = rng.uniform(0.7, 1.3)
        return {"gross_tonnage": gt * s, "international_gross_tonnage": igt * s}
    if kind == "null":
        return {"gross_tonnage": NAN, "international_gross_tonnage": NAN}
    return {}


def name(kind, r, rng):
    n = str(r["vessel_name"])
    if kind == "corrupt":
        return {"vessel_name": B._corrupt_name(n, rng)}
    if kind == "typo":
        return {"vessel_name": B._typo(n, 1, rng)}
    if kind == "half" and rng.random() < 0.5:
        return {"vessel_name": B._corrupt_name(n, rng)}
    return {}


REGIMES = [(nk, tk) for nk in ("clean", "typo", "half", "corrupt") for tk in ("keep", "dom", "s15", "s30", "null")]


def main(fold):
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    model = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    rows = va[va.vessel_name.map(B.is_hangul)]
    rng = np.random.default_rng(fold)
    res = {f"{a}+{b}": [] for a, b in REGIMES}
    for r in rows.to_dict("records"):
        base = {k: r.get(k) for k in I.MODEL_FIELDS}
        base["call_sign"] = "UNSEEN"
        t = ix.id2i[r["target_vessel_id"]]
        for nk, tk in REGIMES:
            q = dict(base, **name(nk, r, rng), **tonnage(tk, r, rng))
            idx, F = B.query_features(ix, q, B.query_candidates(ix, q))
            s = model.predict(F, num_threads=1)
            order = sorted(range(len(idx)), key=lambda j: (-float(s[j]), idx[j]))
            allowed = [j for j in order if idx[j] not in fit_targets] or order
            gap = float(s[allowed[0]] - s[allowed[1]]) if len(allowed) > 1 else 99.0
            pick = allowed[0]
            if gap < 4:
                top = allowed[:3]
                bg = np.nan_to_num(F[top, BG].astype(float), nan=9.0)
                i = int(np.argmin(bg))
                if bg[i] < 0.02 and bg[0] > 0.05:
                    pick = top[i]
            res[f"{nk}+{tk}"].append([int(idx[allowed[0]] == t), int(gap < 4), int(idx[pick] == t)])
    (Path(__file__).parent / f"regime_fp_fold{fold}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(int(sys.argv[1]))
