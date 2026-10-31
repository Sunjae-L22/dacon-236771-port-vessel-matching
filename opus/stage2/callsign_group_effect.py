# author: opus
"""LOCAL MEASUREMENT ONLY — not a submission method.

How much would pooling predictions across test rows that share a call_sign change Korean accuracy?
The competition rules (평가 행 독립) forbid this at inference; this script only measures the effect on a
vessel-disjoint holdout so the team can judge the leaderboard gap.

Holdout: fold2 held-out vessels (astra/incumbent97 fit-only dictionary, seed-0 ranker of the 79 recipe).
Rows keep their real call_sign. Each Korean row is name-corrupted with probability P (opus_lib._corrupt_name),
tonnage kept. Per-row prediction = allowed top-1. Grouped prediction = argmax over candidates of the
summed per-row softmax probabilities of all rows with the same call_sign (any language).

usage: python3 opus/stage2/callsign_group_effect.py
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
import incumbent97 as I  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

B = I.B
FOLD, P = 2, 0.5


def main():
    tr, cat, fit, va, ix, llm, out = I.context(FOLD)
    model = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    ko_cs = set(va.call_sign[va.vessel_name.map(B.is_hangul)])
    rows = va[va.call_sign.isin(ko_cs)]  # every row of vessels that have a Korean row
    print("held-out vessels with Korean rows", len(ko_cs), "rows", len(rows),
          "Korean rows", int(rows.vessel_name.map(B.is_hangul).sum()))
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        preds = []
        for r in rows.to_dict("records"):
            q = {k: r.get(k) for k in I.MODEL_FIELDS}
            is_ko = B.is_hangul(str(q["vessel_name"]))
            if is_ko and rng.random() < P:
                q["vessel_name"] = B._corrupt_name(str(q["vessel_name"]), rng)
            q["call_sign"] = "UNSEEN"  # no train lookup; grouping uses the real call_sign below
            idx, F = B.query_features(ix, q, B.query_candidates(ix, q))
            s = model.predict(F, num_threads=2)
            keep = [k for k in range(len(idx)) if idx[k] not in fit_targets] or list(range(len(idx)))
            sk = s[keep]
            p = np.exp(sk - sk.max()); p /= p.sum()
            order = sorted(range(len(keep)), key=lambda k: (-float(sk[k]), idx[keep[k]]))
            gap = float(sk[order[0]] - sk[order[1]]) if len(order) > 1 else 99.0
            preds.append(dict(cs=r["call_sign"], ko=is_ko and B.is_hangul(str(q["vessel_name"])),
                              target=ix.id2i[r["target_vessel_id"]], top=idx[keep[order[0]]], gap=gap,
                              prob={idx[keep[k]]: float(p[k]) for k in range(len(keep))}))
        pool = defaultdict(lambda: defaultdict(float))
        for d in preds:
            for c, pr in d["prob"].items():
                pool[d["cs"]][c] += pr
        ko = [d for d in preds if d["ko"]]
        row_ok = np.array([d["top"] == d["target"] for d in ko])
        grp_ok = np.array([max(pool[d["cs"]].items(), key=lambda kv: (kv[1], -kv[0]))[0] == d["target"] for d in ko])
        hard = np.array([d["gap"] < 4 for d in ko])
        print(f"seed {seed}: Korean rows {len(ko)}  per-row acc {row_ok.mean():.4f}  grouped acc {grp_ok.mean():.4f}"
              f"  | hard share {hard.mean():.3f}  hard acc per-row {row_ok[hard].mean():.3f} grouped {grp_ok[hard].mean():.3f}"
              f"  | fixed {int((~row_ok & grp_ok).sum())} broken {int((row_ok & ~grp_ok).sum())}")


if __name__ == "__main__":
    main()
