# author: opus
"""99: 93 + Korean stage-2 top-20 reranker (opus/stage2).

Korean route: stage-1 = mean of 72's six rankers; keep the top-20 allowed candidates (train targets
excluded) ordered by (-score, catalog index), exactly as the OOF data was built; stage-2 = mean of
three LightGBM rerankers on stage2_lib.relative features; pick the stage-2 argmax (ties -> stage-1
order). English route unchanged from 93.
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUM = 99

SCRIPT = '''"""DACON inference entry point (99_submit: 93 + Korean stage-2 top-20 reranker)."""
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
from stage2_lib import relative

TOPK = 20


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

    load = lambda fam: [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob(fam + "_*.txt"))]
    ko1, ko2, en65, s62 = load("ranker"), load("stage2"), load("en65"), load("s62")
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
            s1 = sum(m.predict(F) for m in ko1) / len(ko1)
            order = sorted(range(len(idx)), key=lambda j: (-float(s1[j]), idx[j]))
            top = ([j for j in order if idx[j] not in train_targets] or order)[:TOPK]
            Z = relative(F[top].astype(np.float32), s1[top].astype(np.float32), np.zeros(len(top), np.int64))
            s2 = sum(m.predict(Z) for m in ko2)
            best = idx[top[min(range(len(top)), key=lambda k: (-float(s2[k]), k))]]
        else:
            idx, F65 = L65.query_features(ix, row, L65.query_candidates(ix, row))
            score = 1.5 * sum(m.predict(F65) for m in en65) + 1.0 * sum(m.predict(F65[:, :60]) for m in s62)
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
    out = ROOT / "submissions" / f"{NUM}_submit.zip"
    assert not out.exists(), out
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        with zipfile.ZipFile(ROOT / "submissions" / "93_submit.zip") as z:
            z.extractall(d)
        (d / "script.py").unlink()
        for p in sorted((ROOT / "opus/stage2/final").glob("stage2_*.txt")):
            shutil.copy(p, d / "model" / p.name)
        shutil.copy(ROOT / "opus/stage2/stage2_lib.py", d / "model" / "stage2_lib.py")
        (d / "script.py").write_text(SCRIPT)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    z.write(p, str(p.relative_to(d)))
    print(NUM, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
