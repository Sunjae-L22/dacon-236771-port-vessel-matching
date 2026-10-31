# author: opus
"""Retrain the Korean ranker with LLM-Hangul catalog features (shared/src/kocat.py) and test on held-out rows.

The 72B Hangul table covers every catalog vessel, test vessels included, so a record can be compared with the
Hangul rendering of each candidate directly. Two features are appended to opus_lib's 60:
  kc_sim    best jamo similarity between the record name and the candidate's renderings (NaN if not Korean)
  kc_margin kc_sim minus the best kc_sim among the other candidates of the query
Per fold: astra/incumbent97 cached seed-0 blocks (79 recipe) + the two columns; the 'llm' block gets NaN
(its Korean names come from an LLM too, which would make the feature look stronger than on real records).
Eval: held-out Korean rows, fuzzy_probe.py conditions and rng (same names), rows whose vessel is a few-shot
example of the table excluded. Baseline = seed-0 ranker on 110's candidates; new = retrained ranker on 110's
candidates + KoCat top-K.

usage: python3 train_kocat.py FOLD TABLE.csv [K=50]   -> kocat_train/fold{K}.json
"""
import json
import os
import sys
import time
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
KC_FEATURES = ["kc_sim", "kc_margin"]


def kc_cols(kc, name, idx):
    if not B.is_hangul(str(name)):
        return np.full((len(idx), 2), np.nan, np.float32)
    s = kc.sim(name, idx)
    if len(s) > 1:
        top2 = np.sort(s)[-2:]
        other = np.where(s >= top2[1], top2[0], top2[1])
    else:
        other = np.zeros_like(s)
    return np.column_stack([s, s - other]).astype(np.float32)


def main(fold, table, k=50):
    t0 = time.time()
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    kc = KoCat(cat, table)
    plan = json.loads((out / "plan.json").read_text())
    order = plan["seeds"]["0"]["order"]
    Xs, ys, groups = [], [], []
    for name in order:
        rows = json.loads((out / "blocks" / f"{name}.json").read_text())
        for lo in range(0, len(rows), 500):
            with np.load(out / "features" / name / f"{lo:06d}.npz") as z:
                X, y, g, idx, qi = z["X"], z["y"], z["groups"], z["idx"], z["query_index"]
            extra, off = [], 0
            for n, q in zip(g, qi):
                cols = (np.full((n, 2), np.nan, np.float32) if name == "llm"
                        else kc_cols(kc, rows[q]["vessel_name"], idx[off:off + n]))
                extra.append(cols)
                off += n
            Xs.append(np.hstack([X, np.concatenate(extra)]))
            ys.append(y)
            groups.extend(g.tolist())
    X, y = np.concatenate(Xs), np.concatenate(ys)
    print("train", X.shape, f"{time.time() - t0:.0f}s", flush=True)
    params = dict(plan["lgb_params"], num_threads=int(os.environ.get("THREADS", "3")))
    model = lgb.train(params, lgb.Dataset(X, y, group=groups, feature_name=B.FEATURES + KC_FEATURES), 600)
    dst = Path(__file__).parent / "kocat_train"
    dst.mkdir(exist_ok=True)
    model.save_model(str(dst / f"fold{fold}_ranker.txt"))
    print("trained", f"{time.time() - t0:.0f}s", flush=True)
    del X, Xs

    base = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    ex = Extra(ix, cat)
    shots = {en for en, _ in json.load(open(Path(table).parent.parent / "fewshot_pairs.json"))}
    rows = va[va.vessel_name.map(B.is_hangul)]
    rng = np.random.default_rng(900 + fold)  # same draws as fuzzy_probe.py
    res = {f"{c}+{t}|{v}": [] for c, t in CONDS for v in ("110", "kc")}
    for r in rows.to_dict("records"):
        keep = cat.vessel_name[ix.id2i[r["target_vessel_id"]]] not in shots
        b0 = {kk: r.get(kk) for kk in I.MODEL_FIELDS}
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
            for v in ("110", "kc"):
                cand = {kk: list(e) for kk, e in c110.items()}
                if v == "kc":
                    for rk, (i, _) in enumerate(kc.search(q["vessel_name"], k)):
                        e = cand.setdefault(int(i), [999, 999, 999])
                        e[0] = min(e[0], rk)
                idx, F = B.query_features(ix, q, cand)
                sc = base.predict(F, num_threads=1) if v == "110" else \
                    model.predict(np.hstack([F, kc_cols(kc, q["vessel_name"], idx)]), num_threads=1)
                order_ = sorted(range(len(idx)), key=lambda j: (-float(sc[j]), idx[j]))
                allowed = [j for j in order_ if idx[j] not in fit_targets] or order_
                gap = float(sc[allowed[0]] - sc[allowed[1]]) if len(allowed) > 1 else 99.0
                res[f"{nk}+{tk}|{v}"].append([int(t in cand), int(idx[allowed[0]] == t), round(gap, 3)])
    (dst / f"fold{fold}.json").write_text(json.dumps(res))
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], *(int(a) for a in sys.argv[3:]))
