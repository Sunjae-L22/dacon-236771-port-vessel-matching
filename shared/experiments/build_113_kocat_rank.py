# author: opus
"""113: 110 with its six Korean rankers retrained with the LLM-Hangul catalog features (pinned/build113 =
build70 recipe + kc_sim / kc_margin from shared/src/kocat.py) and the top-K KoCat matches added as Korean
candidates. English path, dictionary, stats are 93's.

usage: python3 build_113_kocat_rank.py NUM TABLE.csv OUT_A OUT_B [K=50]
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(num, table, out_a, out_b, k=50):
    rankers = [Path(out_a) / "model" / f"ranker_{i}.txt" for i in range(3)] + \
              [Path(out_b) / "model" / f"ranker_{i}.txt" for i in range(3)]
    assert all(p.exists() for p in rankers)
    edits = [
        (b'"""DACON inference entry point (110_submit: ',
         b'"""DACON inference entry point (%d_submit: 110 + Korean rankers retrained with LLM-Hangul catalog features, KoCat top-%d; base ' % (num, k)),
        (b"    fx = FuzzyExt(ix, catalog)\n",
         b"    fx = FuzzyExt(ix, catalog)\n    from kocat import KoCat, kc_features\n"
         b"    kc = KoCat(catalog, HERE / \"model\" / \"catalog_ko.csv\")\n"),
        (b"            idx, F60 = L.query_features(ix, row, fx.add(L.query_candidates(ix, row), row))\n",
         b"            cand = fx.add(L.query_candidates(ix, row), row)\n"
         b"            for rk, (ci, _) in enumerate(kc.search(q, %d)):\n"
         b"                e = cand.setdefault(int(ci), [999, 999, 999])\n"
         b"                e[0] = min(e[0], rk)\n"
         b"            idx, F60 = L.query_features(ix, row, cand)\n"
         b"            F60 = np.hstack([F60, kc_features(kc, q, idx)])\n" % k),
    ]
    out = ROOT / "submissions" / f"{num}_submit.zip"
    assert not out.exists(), out
    z = zipfile.ZipFile(ROOT / "submissions/110_submit.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as wz:
        for info in z.infolist():
            if info.filename.startswith("model/ranker_"):
                continue
            data = z.read(info)
            if info.filename == "script.py":
                for a, b in edits:
                    assert data.count(a) == 1, a
                    data = data.replace(a, b)
            wz.writestr(info, data)
        for i, p in enumerate(rankers):
            wz.write(p, f"model/ranker_{i}.txt")
        wz.write(ROOT / "shared/src/kocat.py", "model/kocat.py")
        wz.write(table, "model/catalog_ko.csv")
    print(num, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], *(int(a) for a in sys.argv[5:]))
