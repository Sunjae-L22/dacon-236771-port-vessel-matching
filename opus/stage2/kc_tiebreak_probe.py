# author: opus
"""Proxy check for build_vote TIE="kc<M>": among three close candidates, does the 72B Hangul-rendering
similarity pick the answer more often than the first-ranked one?

Held-out Korean rows of fold FOLD (fuzzy_probe.py conditions and rng), 110 candidates, seed-0 base ranker. When
the answer is among the top-3 allowed candidates: ranker = top-1; kc<M> = the most similar of the three if it beats
top-1's similarity by >= M, else top-1. (In the vote the three come from different models, so this only measures
how well the similarity separates close candidates.)

usage: python3 kc_tiebreak_probe.py FOLD TABLE.csv   -> kc_tie/fold{K}.json
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
from kocat import KoCat  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

B = I.B
MS = (0.0, 0.05, 0.1, 0.2)


def main(fold, table):
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    base = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
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
            idx, F = B.query_features(ix, q, ex.add(B.query_candidates(ix, q), q, "F500+rev+ksh"))
            sc = base.predict(F, num_threads=1)
            order = sorted(range(len(idx)), key=lambda j: (-float(sc[j]), idx[j]))
            allowed = [idx[j] for j in order if idx[j] not in fit_targets] or [idx[j] for j in order]
            top = allowed[:3]
            if len(top) < 3 or t not in top:
                continue
            sims = kc.sim(q["vessel_name"], top)
            j = int(sims.argmax())
            row = [int(top[0] == t)]
            for m in MS:
                row.append(int((top[j] if j and sims[j] - sims[0] >= m else top[0]) == t))
            res[f"{nk}+{tk}"].append(row)
    dst = Path(__file__).parent / "kc_tie"
    dst.mkdir(exist_ok=True)
    (dst / f"fold{fold}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
