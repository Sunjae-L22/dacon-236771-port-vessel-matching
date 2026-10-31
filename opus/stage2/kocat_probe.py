# author: opus
"""Does the LLM Hangul rendering of the catalog (shared/src/kocat.py) help?

real : every Korean train vessel not used as a few-shot example, its real record name searched against all
       78,595 catalog renderings by jamo similarity (name only): recall@1/10/100 of the true vessel.
FOLD : held-out Korean rows of fold FOLD as in fuzzy_probe.py; 110's candidates (F500+rev+ksh) vs 110 plus
       the top-K Korean-rendering matches, seed-0 79-recipe ranker, no retraining.

usage: python3 kocat_probe.py real TABLE.csv
       python3 kocat_probe.py FOLD TABLE.csv   -> kocat/fold{K}.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/src"))
from kocat import KoCat  # noqa: E402

DRIVE = Path(os.environ.get("CATALOG_KO_DIR") or next(
    (Path.home() / "Library/CloudStorage").glob("GoogleDrive-*/내 드라이브/Projects/항만문제해결을위한AI기술공모전/catalog_ko")))


def real(table):
    cat = pd.read_csv(ROOT / "data/open/vessel_catalog.csv", encoding="utf-8-sig")
    tr = pd.read_csv(ROOT / "data/open/train.csv", encoding="utf-8-sig")
    shots = {en for en, _ in json.load(open(DRIVE / "fewshot_pairs.json"))}
    id2i = {v: i for i, v in enumerate(cat.vessel_id)}
    kc = KoCat(cat, table)
    ko = tr[tr.vessel_name.map(lambda s: isinstance(s, str) and any("가" <= c <= "힣" for c in s))]
    ko = ko[~ko.target_vessel_id.map(lambda v: cat.vessel_name[id2i[v]] in shots)].drop_duplicates("vessel_name")
    ranks = []
    for r in ko.itertuples():
        hits = [i for i, _ in kc.search(r.vessel_name, 100)]
        t = id2i[r.target_vessel_id]
        ranks.append(hits.index(t) if t in hits else 999)
    ranks = np.array(ranks)
    print(f"real names {len(ranks)}: R@1 {np.mean(ranks < 1):.3f}  R@10 {np.mean(ranks < 10):.3f}  R@100 {np.mean(ranks < 100):.3f}")


def fold_eval(fold, table):
    sys.path.insert(0, str(ROOT / "astra"))
    sys.path.insert(0, str(Path(__file__).parent))
    import incumbent97 as I
    from fuzzy_probe import CONDS, Extra
    from jamo_fingerprint import name as jname
    import lightgbm as lgb
    B = I.B
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    model = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    ex, kc = Extra(ix, cat), KoCat(cat, table)
    rows = va[va.vessel_name.map(B.is_hangul)]
    rng = np.random.default_rng(900 + fold)  # same names as fuzzy_probe.py
    variants = ["110", "110+kc50", "110+kc200"]
    W = [1, 2, 4, 8]
    res_w = {f"{c}+{t}|kc200+w{w}": [] for c, t in CONDS for w in W}
    res = {f"{c}+{t}|{v}": [] for c, t in CONDS for v in variants}
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
            c110 = ex.add(B.query_candidates(ix, q), q, "F500+rev+ksh")
            hits = kc.search(q["vessel_name"], 200)
            for v in variants:
                cand = {k: list(e) for k, e in c110.items()}
                k = 0 if v == "110" else int(v.split("kc")[1])
                for rk, (i, _) in enumerate(hits[:k]):
                    e = cand.setdefault(int(i), [999, 999, 999])
                    e[0] = min(e[0], rk)
                idx, F = B.query_features(ix, q, cand)
                sc = model.predict(F, num_threads=1)
                order = sorted(range(len(idx)), key=lambda j: (-float(sc[j]), idx[j]))
                allowed = [j for j in order if idx[j] not in fit_targets] or order
                res[f"{nk}+{tk}|{v}"].append([int(t in cand), int(idx[allowed[0]] == t), len(idx)])
                if v == "110+kc200":
                    sim = kc.sim(q["vessel_name"], idx)
                    for w in W:
                        s2 = sc + w * sim
                        o2 = sorted(range(len(idx)), key=lambda j: (-float(s2[j]), idx[j]))
                        a2 = [j for j in o2 if idx[j] not in fit_targets] or o2
                        res_w[f"{nk}+{tk}|kc200+w{w}"].append([int(t in cand), int(idx[a2[0]] == t), len(idx)])
    dst = Path(__file__).parent / "kocat"
    dst.mkdir(exist_ok=True)
    (dst / f"fold{fold}.json").write_text(json.dumps({**res, **res_w}))


if __name__ == "__main__":
    if sys.argv[1] == "real":
        real(sys.argv[2])
    else:
        fold_eval(int(sys.argv[1]), sys.argv[2])
