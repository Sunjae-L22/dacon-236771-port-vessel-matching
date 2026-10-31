# author: opus
"""103: 93 + pooling across test rows that share a call_sign.

Per row, 93's models give allowed-candidate scores (train targets excluded, as 93). Scores are
turned into a softmax over that row's candidates, using per-model mean scale (Korean: mean of the
six 72 rankers; English: (1.5*sum65 + sum62) / 15). Rows with the same call_sign (any language)
add their probabilities, summed in sample_id order so the result does not depend on row order, and
each row takes the pooled argmax (ties -> lower catalog index). Rows without a call_sign, and rows
whose call_sign is a train call_sign, keep 93's logic.

NOTE: the written rule 평가 행 독립 forbids group voting over test rows; the user reports a written
DACON answer allowing it (10/31). Keep 93 as one of the final two selections.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUM = 103

SCRIPT = '''"""DACON entry point (103_submit: 93 + call_sign pooling across test rows; see build_103_group.py)."""
from __future__ import annotations
from collections import defaultdict
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
    load = lambda fam: [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob(fam + "_*.txt"))]
    ko, en65, s62 = load("ranker"), load("en65"), load("s62")
    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}

    preds, probs, groups = {}, {}, defaultdict(list)
    for row in test.to_dict("records"):
        sid = row["sample_id"]
        cs = ix.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds[sid] = cs
            continue
        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        if L.is_hangul(q):
            idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
            s = sum(m.predict(F) for m in ko) / len(ko)
        else:
            idx, F65 = L65.query_features(ix, row, L65.query_candidates(ix, row))
            s = (1.5 * sum(m.predict(F65) for m in en65) + sum(m.predict(F65[:, :60]) for m in s62)) / 15.0
        order = sorted(range(len(idx)), key=lambda j: (-float(s[j]), idx[j]))
        keep = [j for j in order if idx[j] not in train_targets] or order
        preds[sid] = idx[keep[0]]  # 93's choice
        sk = np.array([float(s[j]) for j in keep])
        p = np.exp(sk - sk.max())
        probs[sid] = dict(zip((idx[j] for j in keep), (p / p.sum()).tolist()))
        key = row.get("call_sign")
        if isinstance(key, str) and key:
            groups[key].append(sid)

    for key, sids in groups.items():
        if len(sids) < 2:
            continue
        pool = defaultdict(float)
        for sid in sorted(sids):
            for c, pr in probs[sid].items():
                pool[c] += pr
        best = min(pool.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        for sid in sids:
            preds[sid] = best

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"],
                  "target_vessel_id": [ix.ids[preds[s]] for s in test["sample_id"]]}).to_csv(out, index=False)
    print(f"Saved {len(test)} predictions to {out}")


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
        w.writestr("script.py", SCRIPT)
    print(NUM, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
