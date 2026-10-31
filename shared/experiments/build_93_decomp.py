# author: opus
"""10/30 decomposition submissions: put one 92 component at a time onto 79.

79 = Korean 72 rankers + English 65. 92 = Korean (72+Super98+Clean18) + Composer + English (65+62).
92's model/ is a byte-identical superset of 79's (stats.pkl shares every 79 key), so each variant
below differs from 79 (or 93) in exactly one thing.

usage: python3 shared/experiments/build_93_decomp.py   -> submissions/93..96_submit.zip
"""
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submissions" / "92_submit.zip"

# name: (Korean weights per model family, composer on?, English weights per family)
VARIANTS = {
    93: ({"ranker": 1.0}, False, {"en65": 1.5, "s62": 1.0}),       # 79 + 92's English only
    94: ({"ranker": 1.0}, True, {"en65": 1.5, "s62": 1.0}),        # 93 + Composer
    95: ({"ranker": 0.5, "sup98": 0.5}, False, {"en65": 1.5, "s62": 1.0}),  # 93 + Super98
    96: ({"ranker": 0.5, "cln18": 0.5}, False, {"en65": 1.5, "s62": 1.0}),  # 93 + Clean18
}

SCRIPT = '''"""DACON inference entry point ({num}_submit: decomposition of 92 onto 79). KO={ko} COMPOSE={compose} EN={en}"""
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

KO = {ko}
COMPOSE = {compose}
EN = {en}
NEED_SUPER = COMPOSE or "sup98" in KO or "cln18" in KO
if NEED_SUPER:
    from super_features import CLEAN18_KEEP_IDX, SuperFeatureExtractor, compose_compat_jamo


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

    sfe = None
    if "sup98" in KO or "cln18" in KO:
        sfe = SuperFeatureExtractor(ix, catalog, None, L.FEATURES, fit_vids=set(stats["fit_vids"]),
                                    type_unit_counts=stats["type_unit_counts"], stateless=True)
        keep_idx = np.asarray(CLEAN18_KEEP_IDX, dtype=np.int32)

    models = {{fam: [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob(fam + "_*.txt"))]
              for fam in list(KO) + list(EN)}}
    train_targets = {{ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}}
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
            idx, F60 = L.query_features(ix, row, L.query_candidates(ix, row))
            feats = {{"ranker": F60}}
            if sfe is not None:
                E24 = sfe.compute_query(row, idx, F60)
                feats["sup98"] = np.column_stack((F60, E24))
                feats["cln18"] = np.column_stack((F60, E24[:, keep_idx]))
            score = sum(w * sum(m.predict(feats.get(fam, F60)) for m in models[fam]) for fam, w in KO.items())
        else:
            idx, F65 = L65.query_features(ix, row, L65.query_candidates(ix, row))
            feats = {{"en65": F65, "s62": F65[:, :60]}}
            score = sum(w * sum(m.predict(feats[fam]) for m in models[fam]) for fam, w in EN.items())
        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
        best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])
        preds.append(ix.ids[best])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"sample_id": test["sample_id"], "target_vessel_id": preds}}).to_csv(out, index=False)
    print(f"Saved {{len(preds)}} predictions to {{out}}")


if __name__ == "__main__":
    run()
'''

COMMON = ["model/opus_lib.py", "model/opus_lib_65.py", "model/ko_translit_argon.py", "model/stats.pkl",
          "requirements.txt"]


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        with zipfile.ZipFile(SRC) as z:
            z.extractall(d)
        for num, (ko, compose, en) in VARIANTS.items():
            out = ROOT / "submissions" / f"{num}_submit.zip"
            assert not out.exists(), out
            files = list(COMMON)
            for fam in list(ko) + list(en):
                files += sorted(str(p.relative_to(d)) for p in (d / "model").glob(fam + "_*.txt"))
            if compose or "sup98" in ko or "cln18" in ko:
                files.append("model/super_features.py")
            script = SCRIPT.format(num=num, ko=json.dumps(ko), compose=compose, en=json.dumps(en))
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    z.write(d / f, f)
                z.writestr("script.py", script)
            print(num, out.stat().st_size / 1e6, "MB", len(files) + 1, "files")


if __name__ == "__main__":
    main()
