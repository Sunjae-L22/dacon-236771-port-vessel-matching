# author: opus
"""Which input regime reproduces the Public Korean profile (102: ~30% hard rows, hard acc ~0.76, easy ~0.995)?

Fold-2 held-out vessels (vessel-disjoint), seed-0 ranker of the 79 recipe. Every Korean row of the held-out
vessels is scored under several input regimes; reports accuracy, recall@candidates, hard share (gap<4) and
hard accuracy. 105 (Public Korean -27) says the true ship's tonnage is often >5% off in test hard rows,
while in train Korean rows the international tonnage is within 5% for 99.2% of rows.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
import incumbent97 as I  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

B = I.B
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NAN = float("nan")


def regimes(r, rng):
    gt, igt = r["gross_tonnage"], r["international_gross_tonnage"]
    yield "clean", {}
    yield "igt_null", {"international_gross_tonnage": NAN}
    yield "gt_only_dom", {"international_gross_tonnage": gt, "gross_tonnage": gt}
    yield "both_null", {"gross_tonnage": NAN, "international_gross_tonnage": NAN}
    s = rng.uniform(0.85, 1.15)
    yield "scale15", {"gross_tonnage": gt * s, "international_gross_tonnage": igt * s}
    yield "corrupt", {"vessel_name": B._corrupt_name(str(r["vessel_name"]), rng)}


def main():
    tr, cat, fit, va, ix, llm, out = I.context(FOLD)
    model = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    rows = va[va.vessel_name.map(B.is_hangul)]
    rng = np.random.default_rng(0)
    stats = {}
    for r in rows.to_dict("records"):
        base = {k: r.get(k) for k in I.MODEL_FIELDS}
        base["call_sign"] = "UNSEEN"
        t = ix.id2i[r["target_vessel_id"]]
        for name, upd in regimes(r, rng):
            q = dict(base, **upd)
            idx, F = B.query_features(ix, q, B.query_candidates(ix, q))
            s = model.predict(F, num_threads=2)
            keep = [k for k in range(len(idx)) if idx[k] not in fit_targets] or list(range(len(idx)))
            order = sorted(keep, key=lambda k: (-float(s[k]), idx[k]))
            gap = float(s[order[0]] - s[order[1]]) if len(order) > 1 else 99.0
            stats.setdefault(name, []).append((idx[order[0]] == t, t in idx, gap < 4))
    print(f"fold {FOLD}  Korean rows {len(rows)}")
    for name, v in stats.items():
        a = np.array(v)
        ok, rec, hard = a[:, 0], a[:, 1], a[:, 2]
        print(f"  {name:12s} acc {ok.mean():.4f}  recall {rec.mean():.4f}  hard share {hard.mean():.3f}"
              f"  hard acc {ok[hard].mean() if hard.any() else 0:.3f}  easy acc {ok[~hard].mean():.4f}")


if __name__ == "__main__":
    main()
