# author: opus
"""Does training on the test-like regime (one typo + tonnage x0.8-1.25) fix it?

regime_fingerprint.py: the Public Korean profile (acc .925, hard .30, hard acc .76, 105 -5.1%) is matched by
"one name typo + tonnage scaled U(0.85,1.15)" (.934/.255/.750/-3.6%); clean names score .997 locally.
Per fold: the 79-recipe seed-0 ranker (astra/incumbent97 cached blocks, read only) retrained with an extra
block of Korean fit rows: COPIES copies, each with one typo (prob TYPO_P) and tonnage scaled U(0.8,1.25).
Both rankers are scored on the held-out Korean rows under the fingerprint regimes (same rng draws).

usage: python3 train_typo.py FOLD [COPIES] [TYPO_P]     -> opus/stage2/typo/foldK/
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import incumbent97 as I  # noqa: E402
from regime_fingerprint import name, tonnage  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

B = I.B
FOLD = int(sys.argv[1])
COPIES = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TYPO_P = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
OUT = Path(__file__).parent / "typo" / f"fold{FOLD}_c{COPIES}_p{TYPO_P}"
EVAL = [("clean", "keep"), ("clean", "s15"), ("typo", "keep"), ("typo", "s15"), ("typo", "s30"), ("half", "keep"), ("corrupt", "keep")]


def features(ix, rows):
    X, y, g = [], [], []
    for row in rows:
        idx, F = B.query_features(ix, row, B.query_candidates(ix, row))
        t = ix.id2i[row["target_vessel_id"]]
        X.append(F); y.append(np.array([i == t for i in idx], np.int8)); g.append(len(idx))
    return np.concatenate(X).astype(np.float32), np.concatenate(y), g


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tr, cat, fit, va, ix, llm, out = I.context(FOLD)
    plan = json.loads((out / "plan.json").read_text())
    order = plan["seeds"]["0"]["order"]
    rng = np.random.default_rng(10_000 + FOLD)
    ko = fit[fit.vessel_name.map(B.is_hangul)]
    extra = []
    for c in range(COPIES):
        for r in ko.to_dict("records"):
            q = {k: r.get(k) for k in I.MODEL_FIELDS + ["target_vessel_id"]}
            if rng.random() < TYPO_P:
                q["vessel_name"] = B._typo(str(q["vessel_name"]), 1, rng)
            s = rng.uniform(0.8, 1.25)
            q["gross_tonnage"] = q["gross_tonnage"] * s
            q["international_gross_tonnage"] = q["international_gross_tonnage"] * s
            extra.append(q)
    Xe, ye, ge = features(ix, extra)
    print("extra block", len(extra), "queries", round(time.time() - t0), "s", flush=True)
    Xs, ys, gs = [], [], []
    for nm in order:
        n = len(json.loads((out / "blocks" / f"{nm}.json").read_text()))
        for lo in range(0, n, 500):
            with np.load(out / "features" / nm / f"{lo:06d}.npz") as z:
                Xs.append(z["X"].astype(np.float32)); ys.append(z["y"]); gs.extend(z["groups"].tolist())
    X = np.concatenate(Xs + [Xe]); y = np.concatenate(ys + [ye]); g = gs + ge
    del Xs
    params = dict(I.PARAMS, num_threads=3)
    model = lgb.train(params, lgb.Dataset(X, y, group=g, feature_name=B.FEATURES), 600)
    model.save_model(str(OUT / "ranker.txt"))
    print("trained", round(time.time() - t0), "s", flush=True)
    base = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    rows = va[va.vessel_name.map(B.is_hangul)]
    erng = np.random.default_rng(FOLD)
    res = {f"{a}+{b}": {"base": [], "new": []} for a, b in EVAL}
    for r in rows.to_dict("records"):
        b0 = {k: r.get(k) for k in I.MODEL_FIELDS}
        b0["call_sign"] = "UNSEEN"
        t = ix.id2i[r["target_vessel_id"]]
        for nk, tk in EVAL:
            q = dict(b0, **name(nk, r, erng), **tonnage(tk, r, erng))
            idx, F = B.query_features(ix, q, B.query_candidates(ix, q))
            for key, m in (("base", base), ("new", model)):
                s = m.predict(F, num_threads=1)
                allowed = [j for j in range(len(idx)) if idx[j] not in fit_targets] or list(range(len(idx)))
                top = min(allowed, key=lambda j: (-float(s[j]), idx[j]))
                res[f"{nk}+{tk}"][key].append(int(idx[top] == t))
    (OUT / "eval.json").write_text(json.dumps(res))
    for k, v in res.items():
        print(f"{k:14s} base {np.mean(v['base']):.4f}  new {np.mean(v['new']):.4f}  ({sum(v['new']) - sum(v['base']):+d}/{len(v['base'])})", flush=True)


if __name__ == "__main__":
    main()
