"""Build submissions/80_submit.zip:
Scales our 0.9418190937 architecture (72 Korean + 65 English, which only used 6 models each: SEED_OFFSET 0 + 5000)
to a full 12-model (4-seed-bundle: 0 + 5000 + 10000 + 15000) ensemble on BOTH Korean and English by blending with
62_submit.zip (37 + 38, SEED_OFFSET 10000 + 15000)!
"""
from __future__ import annotations
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(unicodedata.normalize("NFD", "/Users/sunjae/Desktop/항만"))

SCRIPT_80 = '''"""DACON inference entry point (80_submit: 12-model Korean [72 + 62] + 12-model English [65 + 62])."""
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

    models_72 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ranker_*.txt"))]
    models_62 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("s62_*.txt"))]
    models_65 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("en65_*.txt"))]
    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}
    preds = []
    for row in test.to_dict("records"):
        cs = ix.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix.ids[cs])
            continue
        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        if L.is_hangul(q):
            idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
            score = 1.5 * sum(m.predict(F) for m in models_72) + 1.0 * sum(m.predict(F) for m in models_62)
        else:
            idx, F65 = L65.query_features(ix, row, L65.query_candidates(ix, row))
            F60 = F65[:, :60]
            score = 1.5 * sum(m.predict(F65) for m in models_65) + 1.0 * sum(m.predict(F60) for m in models_62)
        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
        best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])
        preds.append(ix.ids[best])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(out, index=False)
    print(f"Saved {len(preds)} predictions to {out}")


if __name__ == "__main__":
    run()
'''


def main():
    with zipfile.ZipFile(ROOT / "submissions/72_submit.zip") as z72:
        items = {n: z72.read(n) for n in z72.namelist() if not n.endswith("/")}
    with zipfile.ZipFile(ROOT / "submissions/62_submit.zip") as z62:
        for i in range(6):
            items[f"model/s62_{i}.txt"] = z62.read(f"model/ranker_{i}.txt")
    with zipfile.ZipFile(ROOT / "submissions/65_submit.zip") as z65:
        items["model/opus_lib_65.py"] = z65.read("model/opus_lib.py")
        for i in range(6):
            items[f"model/en65_{i}.txt"] = z65.read(f"model/ranker_{i}.txt")
    items["script.py"] = SCRIPT_80.encode("utf-8")
    dst = ROOT / "submissions/80_submit.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in sorted(items.keys()):
            zout.writestr(n, items[n])
    print(f"Built {dst} ({dst.stat().st_size / 1e6:.2f} MB) with 6 (72) + 6 (62) + 6 (65) = 18 models.")


if __name__ == "__main__":
    main()
