"""DACON inference entry point (112_submit: 110 + Korean rankers retrained with LLM-Hangul catalog features, KoCat top-50, gate 4.0; base 93 + user v18 fuzzy Korean candidates; base decomposition of 92 onto 79). KO={"ranker": 1.0} COMPOSE=False EN={"en65": 1.5, "s62": 1.0}"""
from __future__ import annotations
from pathlib import Path
import os
import pickle
import sys
import unicodedata

if os.environ.get("PYTHONHASHSEED") is None:
    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED="0"))

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
import opus_lib as L
import opus_lib_65 as L65

KO = {"ranker": 1.0}
COMPOSE = False
EN = {"en65": 1.5, "s62": 1.0}
NEED_SUPER = COMPOSE or "sup98" in KO or "cln18" in KO
if NEED_SUPER:
    from super_features import CLEAN18_KEEP_IDX, SuperFeatureExtractor, compose_compat_jamo


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    with open(HERE / "model" / "stats.pkl", "rb") as f:
        stats = pickle.load(f)

    ix = L.Index(catalog, None)
    ix.cs_map = {k: ix.id2i[v] for k, v in stats["cs_map"].items() if v in ix.id2i}
    ix.flag_p, ix.type_p = stats["flag_p"], stats["type_p"]
    ix.train_targets = set(ix.cs_map.values())
    ix.ko = L.KA.KoMatcher(catalog, stats["ko_pairs"])
    ix.fit_alias(stats["alias_rows"])
    from fuzzy_ext import FuzzyExt
    fx = FuzzyExt(ix, catalog)
    from kocat import KoCat, kc_features
    kc = KoCat(catalog, HERE / "model" / "catalog_ko.csv")
    kc_models = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("kc113_*.txt"))]
    assert len(kc_models) == 6
    GATE = 4.0

    sfe = None
    if "sup98" in KO or "cln18" in KO:
        sfe = SuperFeatureExtractor(ix, catalog, None, L.FEATURES, fit_vids=set(stats["fit_vids"]),
                                    type_unit_counts=stats["type_unit_counts"], stateless=True)
        keep_idx = np.asarray(CLEAN18_KEEP_IDX, dtype=np.int32)

    models = {fam: [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob(fam + "_*.txt"))]
              for fam in list(KO) + list(EN)}
    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}
    preds = []
    for row in test.to_dict("records"):
        cs = ix.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix.ids[cs])
            continue
        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        if COMPOSE:
            cq = compose_compat_jamo(q)
            if cq != q:
                row = dict(row)
                row["vessel_name"] = q = cq
        if L.is_hangul(q):
            idx, F60 = L.query_features(ix, row, fx.add(L.query_candidates(ix, row), row))
            feats = {"ranker": F60}
            if sfe is not None:
                E24 = sfe.compute_query(row, idx, F60)
                feats["sup98"] = np.column_stack((F60, E24))
                feats["cln18"] = np.column_stack((F60, E24[:, keep_idx]))
            score = sum(w * sum(m.predict(feats[fam]) for m in models[fam]) for fam, w in KO.items())
            use_kc = True
            if GATE is not None:
                o110 = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
                a110 = [j for j in o110 if idx[j] not in train_targets] or o110
                g110 = float(score[a110[0]] - score[a110[1]]) / len(models["ranker"]) if len(a110) > 1 else 99.0
                use_kc = g110 < GATE
            if use_kc:
                cand = fx.add(L.query_candidates(ix, row), row)
                for rk, (ci, _) in enumerate(kc.search(q, 50)):
                    e = cand.setdefault(int(ci), [999, 999, 999])
                    e[0] = min(e[0], rk)
                idx, F62 = L.query_features(ix, row, cand)
                F62 = np.hstack([F62, kc_features(kc, q, idx)])
                score = sum(m.predict(F62) for m in kc_models)
        else:
            idx, F65 = L65.query_features(ix, row, L65.query_candidates(ix, row))
            feats = {"en65": F65, "s62": F65[:, :60]}
            score = sum(w * sum(m.predict(feats[fam]) for m in models[fam]) for fam, w in EN.items())
        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
        best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])
        preds.append(ix.ids[best])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(out, index=False)
    print(f"Saved {len(preds)} predictions to {out}")


if __name__ == "__main__":
    run()
