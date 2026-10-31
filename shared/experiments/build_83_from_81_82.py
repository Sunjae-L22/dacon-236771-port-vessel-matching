"""Combine 81+82 (6 dedicated Korean-only models) + 72_submit (6 joint models) on Korean, and 65_submit on English -> 83_submit.zip."""
from __future__ import annotations
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(unicodedata.normalize("NFD", "/Users/sunjae/Desktop/항만"))

SCRIPT_83 = '''"""DACON inference entry point (83_submit: Dedicated Korean Ranker [81+82] + Joint [72] on Korean, [65] on English)."""
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
import opus_lib as L72
import opus_lib_81 as L81
import opus_lib_65 as L65


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    with open(HERE / "model" / "stats.pkl", "rb") as f:
        stats = pickle.load(f)

    ix72 = L72.Index(catalog, None)
    ix72.cs_map = {k: ix72.id2i[v] for k, v in stats["cs_map"].items() if v in ix72.id2i}
    ix72.flag_p, ix72.type_p = stats["flag_p"], stats["type_p"]
    ix72.train_targets = set(ix72.cs_map.values())
    ix72.ko = L72.KA.KoMatcher(catalog, stats["ko_pairs"])
    ix72.fit_alias(stats["alias_rows"])

    ix81 = L81.Index(catalog, None)
    ix81.cs_map = ix72.cs_map
    ix81.flag_p, ix81.type_p = stats["flag_p"], stats["type_p"]
    ix81.train_targets = ix72.train_targets
    ix81.ko = L81.KA.KoMatcher(catalog, stats["ko_pairs"])
    ix81.fit_alias(stats["alias_rows"])

    models_72 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ranker_*.txt"))]
    models_81 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ko81_*.txt"))]
    models_65 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("en65_*.txt"))]
    train_targets = {ix72.id2i[v] for v in stats["cs_map"].values() if v in ix72.id2i}
    preds = []
    for row in test.to_dict("records"):
        cs = ix72.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix72.ids[cs])
            continue
        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        if L72.is_hangul(q):
            cand = L72.query_candidates(ix72, row)
            idx, F72 = L72.query_features(ix72, row, cand)
            _, F81 = L81.query_features(ix81, row, cand)
            score = 1.0 * sum(m.predict(F72) for m in models_72) + 1.0 * sum(m.predict(F81) for m in models_81)
        else:
            idx, F65 = L65.query_features(ix72, row, L65.query_candidates(ix72, row))
            score = sum(m.predict(F65) for m in models_65)
        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
        best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])
        preds.append(ix72.ids[best])

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
    with zipfile.ZipFile(ROOT / "submissions/81_submit.zip") as z81:
        items["model/opus_lib_81.py"] = z81.read("model/opus_lib.py")
        items["model/ko_translit_argon_81.py"] = z81.read("model/ko_translit_argon.py")
        for i in range(3):
            items[f"model/ko81_{i}.txt"] = z81.read(f"model/ranker_{i}.txt")
    with zipfile.ZipFile(ROOT / "submissions/82_submit.zip") as z82:
        for i in range(3):
            items[f"model/ko81_{i+3}.txt"] = z82.read(f"model/ranker_{i}.txt")
    # Make opus_lib_81 import ko_translit_argon_81
    items["model/opus_lib_81.py"] = (
        items["model/opus_lib_81.py"]
        .decode("utf-8")
        .replace("import ko_translit_argon as KA", "import ko_translit_argon_81 as KA")
        .encode("utf-8")
    )
    with zipfile.ZipFile(ROOT / "submissions/65_submit.zip") as z65:
        items["model/opus_lib_65.py"] = z65.read("model/opus_lib.py")
        for i in range(6):
            items[f"model/en65_{i}.txt"] = z65.read(f"model/ranker_{i}.txt")
    items["script.py"] = SCRIPT_83.encode("utf-8")
    dst = ROOT / "submissions/83_submit.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in sorted(items.keys()):
            zout.writestr(n, items[n])
    print(f"Built {dst} ({dst.stat().st_size / 1e6:.2f} MB) with 6 (72) + 6 (81/82 Dedicated Ko) + 6 (65 En) = 18 models.")


if __name__ == "__main__":
    main()
