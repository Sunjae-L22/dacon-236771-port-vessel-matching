# author: opus
"""112: 110 + Korean-side retrieval on Korean rows (shared/src/kocat.py): catalog names rendered into Hangul
offline by Qwen2.5-72B (catalog_ko/4_catalog_ko.ipynb), matched to the record name at the jamo level.
Adds the top-K matches as candidates and optionally W * similarity to the six-ranker mean score.
Models and the English path are 93's.

usage: python3 build_112_kocat.py NUM TABLE.csv K W
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(num, table, k, w):
    edits = [
        (b'"""DACON inference entry point (110_submit: ',
         b'"""DACON inference entry point (%d_submit: 110 + LLM-Hangul catalog retrieval K=%d W=%s; base ' % (num, k, w.encode())),
        (b"    fx = FuzzyExt(ix, catalog)\n",
         b"    fx = FuzzyExt(ix, catalog)\n    from kocat import KoCat\n    kc = KoCat(catalog, HERE / \"model\" / \"catalog_ko.csv\")\n"),
        (b"            idx, F60 = L.query_features(ix, row, fx.add(L.query_candidates(ix, row), row))\n",
         b"            cand = fx.add(L.query_candidates(ix, row), row)\n"
         b"            for rk, (ci, _) in enumerate(kc.search(q, %d)):\n"
         b"                e = cand.setdefault(int(ci), [999, 999, 999])\n"
         b"                e[0] = min(e[0], rk)\n"
         b"            idx, F60 = L.query_features(ix, row, cand)\n" % k),
        (b"            score = sum(w * sum(m.predict(feats[fam]) for m in models[fam]) for fam, w in KO.items())\n",
         b"            score = sum(w * sum(m.predict(feats[fam]) for m in models[fam]) for fam, w in KO.items())\n"
         b"            score = score + %s * len(models[\"ranker\"]) * kc.sim(q, idx)\n" % w.encode()),
    ]
    out = ROOT / "submissions" / f"{num}_submit.zip"
    assert not out.exists(), out
    z = zipfile.ZipFile(ROOT / "submissions/110_submit.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as wz:
        for info in z.infolist():
            data = z.read(info)
            if info.filename == "script.py":
                for a, b in edits:
                    assert data.count(a) == 1, a
                    data = data.replace(a, b)
            wz.writestr(info, data)
        wz.write(ROOT / "shared/src/kocat.py", "model/kocat.py")
        wz.write(table, "model/catalog_ko.csv")
    print(num, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4])
