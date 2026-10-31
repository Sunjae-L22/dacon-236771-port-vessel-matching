# author: opus
"""Inference-time blend: s = s_full + lam * s_nogt, where s_nogt scores the same candidates with the
query's tonnage removed (the ranker was trained on no-tonnage copies). 105 lost 27 Public Korean rows
by trusting tonnage on hard rows, so this softens tonnage reliance without retraining.

Fold-2 held-out Korean rows (vessel-disjoint), seed-0 ranker, under the regimes of regime_probe.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import incumbent97 as I  # noqa: E402
from regime_probe import regimes, NAN  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

B = I.B
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 2
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0)
DROP = {"gt_gap", "igt_gap", "best_gap", "gt_exact", "dwt_gap", "gt_exact_n", "rank_gt"}


def main():
    tr, cat, fit, va, ix, llm, out = I.context(FOLD)
    model = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    rows = va[va.vessel_name.map(B.is_hangul)]
    rng = np.random.default_rng(0)
    res = {}
    for r in rows.to_dict("records"):
        base = {k: r.get(k) for k in I.MODEL_FIELDS}
        base["call_sign"] = "UNSEEN"
        t = ix.id2i[r["target_vessel_id"]]
        for name, upd in regimes(r, rng):
            q = dict(base, **upd)
            cand = B.query_candidates(ix, q)
            idx, F = B.query_features(ix, q, cand)
            idx2, F2 = B.query_features(ix, dict(q, gross_tonnage=NAN, international_gross_tonnage=NAN), cand)
            pos = {c: k for k, c in enumerate(idx2)}
            s1 = model.predict(F, num_threads=2)
            s2 = model.predict(F2, num_threads=2)[[pos[c] for c in idx]]
            keep = [k for k in range(len(idx)) if idx[k] not in fit_targets] or list(range(len(idx)))
            for lam in LAMS:
                s = s1 + lam * s2
                top = min(keep, key=lambda k: (-float(s[k]), idx[k]))
                res.setdefault((name, lam), []).append(idx[top] == t)
    print(f"fold {FOLD}  Korean rows {len(rows)}")
    for name in dict.fromkeys(n for n, _ in res):
        print(f"  {name:12s} " + "  ".join(f"lam{lam}: {np.mean(res[(name, lam)]):.4f}" for lam in LAMS))


if __name__ == "__main__":
    main()
