# author: opus
"""110: 93 + the user's wide fuzzy candidates on Korean rows (pinned/build110/fuzzy_ext.py). Models, English
path and everything else are 93's byte for byte; only Korean candidate generation grows.

usage: python3 build_110_fuzzy.py [NUM]
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIN = ROOT / "shared/experiments/pinned/build110"
EDITS = [
    (b'"""DACON inference entry point (93_submit: ',
     b'"""DACON inference entry point (%d_submit: 93 + user v18 fuzzy Korean candidates; base '),
    (b"    ix.fit_alias(stats[\"alias_rows\"])\n",
     b"    ix.fit_alias(stats[\"alias_rows\"])\n    from fuzzy_ext import FuzzyExt\n    fx = FuzzyExt(ix, catalog)\n"),
    (b"idx, F60 = L.query_features(ix, row, L.query_candidates(ix, row))",
     b"idx, F60 = L.query_features(ix, row, fx.add(L.query_candidates(ix, row), row))"),
]


def main(num=110):
    out = ROOT / "submissions" / f"{num}_submit.zip"
    assert not out.exists(), out
    z93 = zipfile.ZipFile(ROOT / "submissions/93_submit.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as w:
        for info in z93.infolist():
            data = z93.read(info)
            if info.filename == "script.py":
                for a, b in EDITS:
                    assert data.count(a) == 1, a
                    data = data.replace(a, b % num if b"%d" in b else b)
            w.writestr(info, data)
        for f in ("fuzzy_ext.py", "ksh_hangul.py"):
            w.write(PIN / f, f"model/{f}")
    print(num, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
