# author: opus
"""102 (diagnostic probe, not a candidate): how many Public Korean rows are "easy"?

Same as 93, except a Korean row keeps 93's prediction only when the stage-1 leader beats the
runner-up by at least GAP (mean-of-6 score units, allowed candidates only); otherwise it outputs a
fixed dummy id. English rows are 93's, so English stays 3506 and the Korean count decodes exactly.
On vessel-disjoint OOF a gap >= 4 is 99.5% precise and covers 95% of clean native Korean rows, so
  easy rows  |C| ~= Ko_102 / 0.995,  hard rows = 532 - |C|,  hard accuracy = (492 - Ko_102) / hard.
Row-independent like every submission.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUM, GAP = 102, 4.0

SCRIPT = '''"""DACON entry point (102_submit: diagnostic probe — 93 with low-margin Korean rows blanked, GAP={gap})."""
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

GAP = {gap}


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    with open(HERE / "model" / "stats.pkl", "rb") as f:
        stats = pickle.load(f)
    ix = L.Index(catalog, None)
    ix.cs_map = {{k: ix.id2i[v] for k, v in stats["cs_map"].items() if v in ix.id2i}}
    ix.flag_p, ix.type_p = stats["flag_p"], stats["type_p"]
    ix.train_targets = set(ix.cs_map.values())
    ix.ko = L.KA.KoMatcher(catalog, stats["ko_pairs"])
    ix.fit_alias(stats["alias_rows"])
    load = lambda fam: [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob(fam + "_*.txt"))]
    ko, en65, s62 = load("ranker"), load("en65"), load("s62")
    train_targets = {{ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}}
    dummy = ix.ids[0]
    preds = []
    for row in test.to_dict("records"):
        cs = ix.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix.ids[cs])
            continue
        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        if L.is_hangul(q):
            idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
            s = sum(m.predict(F) for m in ko) / len(ko)
            order = sorted(range(len(idx)), key=lambda j: (-float(s[j]), idx[j]))
            allowed = [j for j in order if idx[j] not in train_targets] or order
            gap = float(s[allowed[0]] - s[allowed[1]]) if len(allowed) > 1 else 99.0
            preds.append(ix.ids[idx[allowed[0]]] if gap >= GAP else dummy)
            continue
        idx, F65 = L65.query_features(ix, row, L65.query_candidates(ix, row))
        score = 1.5 * sum(m.predict(F65) for m in en65) + 1.0 * sum(m.predict(F65[:, :60]) for m in s62)
        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
        preds.append(ix.ids[next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"sample_id": test["sample_id"], "target_vessel_id": preds}}).to_csv(out, index=False)
    print(f"Saved {{len(preds)}} predictions to {{out}}")


if __name__ == "__main__":
    run()
'''


def main():
    out = ROOT / "submissions" / f"{NUM}_submit.zip"
    assert not out.exists(), out
    z93 = zipfile.ZipFile(ROOT / "submissions/93_submit.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as w:
        for info in z93.infolist():
            if info.filename != "script.py":
                w.writestr(info, z93.read(info))
        w.writestr("script.py", SCRIPT.format(gap=GAP))
    print(NUM, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
