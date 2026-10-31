# author: opus
"""After 112 (Public Korean 501): which no-retrain setting of the kc ranker is best?

Per held-out Korean row (fuzzy_probe.py conditions and rng), with the fold kc ranker from train_kocat.py and the
fold seed-0 base ranker: 110's answer and gap, and the kc ranker's answer with KoCat top-50 / top-200 candidates.
Gate variants (keep 110 when its gap >= G) are computed from these columns afterwards.

usage: python3 kocat_variants.py FOLD TABLE.csv   -> kocat_var/fold{K}.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
sys.path.insert(0, str(ROOT / "shared/src"))
sys.path.insert(0, str(Path(__file__).parent))
import incumbent97 as I  # noqa: E402
from fuzzy_probe import CONDS, Extra  # noqa: E402
from jamo_fingerprint import name as jname  # noqa: E402
from kocat import KoCat, kc_features  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

B = I.B


def pick(idx, sc, fit_targets):
    order = sorted(range(len(idx)), key=lambda j: (-float(sc[j]), idx[j]))
    allowed = [j for j in order if idx[j] not in fit_targets] or order
    gap = float(sc[allowed[0]] - sc[allowed[1]]) if len(allowed) > 1 else 99.0
    return idx[allowed[0]], gap


def main(fold, table):
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    base = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    kcm = lgb.Booster(model_file=str(Path(__file__).parent / "kocat_train" / f"fold{fold}_ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    ex, kc = Extra(ix, cat), KoCat(cat, table)
    shots = {en for en, _ in json.load(open(Path(table).parent.parent / "fewshot_pairs.json"))}
    rows = va[va.vessel_name.map(B.is_hangul)]
    rng = np.random.default_rng(900 + fold)
    res = {f"{c}+{t}": [] for c, t in CONDS}
    for r in rows.to_dict("records"):
        keep = cat.vessel_name[ix.id2i[r["target_vessel_id"]]] not in shots
        b0 = {k: r.get(k) for k in I.MODEL_FIELDS}
        b0["call_sign"] = "UNSEEN"
        t = ix.id2i[r["target_vessel_id"]]
        for nk, tk in CONDS:
            s0 = str(r["vessel_name"])
            q = dict(b0, vessel_name=B._corrupt_name(s0, rng) if nk == "corrupt" else jname(nk, s0, rng))
            if tk == "s15":
                s = rng.uniform(0.85, 1.15)
                q["gross_tonnage"] = r["gross_tonnage"] * s
                q["international_gross_tonnage"] = r["international_gross_tonnage"] * s
            elif tk == "null":
                q["gross_tonnage"] = q["international_gross_tonnage"] = np.nan
            if not keep:
                continue
            c110 = ex.add(B.query_candidates(ix, q), q, "F500+rev+ksh")
            idx, F = B.query_features(ix, q, c110)
            p110, g110 = pick(idx, base.predict(F, num_threads=1), fit_targets)
            hits = kc.search(q["vessel_name"], 200)
            row = [int(p110 == t), round(g110, 3)]
            for k in (50, 200):
                cand = {kk: list(e) for kk, e in c110.items()}
                for rk, (i, _) in enumerate(hits[:k]):
                    e = cand.setdefault(int(i), [999, 999, 999])
                    e[0] = min(e[0], rk)
                idx, F = B.query_features(ix, q, cand)
                pk, _ = pick(idx, kcm.predict(np.hstack([F, kc_features(kc, q["vessel_name"], idx)]), num_threads=1), fit_targets)
                row.append(int(pk == t))
            res[f"{nk}+{tk}"].append(row)
    dst = Path(__file__).parent / "kocat_var"
    dst.mkdir(exist_ok=True)
    (dst / f"fold{fold}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
