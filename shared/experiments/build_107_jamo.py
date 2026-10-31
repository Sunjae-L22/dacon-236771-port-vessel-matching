# author: opus
"""107: 93 with its six Korean rankers replaced by build107 rankers (same 72 recipe + three Korean copies
with jamo-level typos and tonnage x U(0.8, 1.2); see pinned/build107/make_submit.py:augment_typo).

Why: opus/stage2/jamo_fingerprint.py — "two jamo typos + tonnage x U(.85,1.15)" reproduces the Public Korean
profile (acc .925, hard .30, hard acc .76, 105 -5.1%); on 5-fold held-out Korean rows the extra copies give
+38/1191 there and 0 on clean names (opus/stage2/train_jamo.py).
Everything else (English models, dictionary, stats, script) is 93's byte for byte.

usage: python3 build_107_jamo.py NUM OUT_A OUT_B
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(num, out_a, out_b):
    rankers = [Path(out_a) / "model" / f"ranker_{i}.txt" for i in range(3)] + \
              [Path(out_b) / "model" / f"ranker_{i}.txt" for i in range(3)]
    assert all(p.exists() for p in rankers)
    z93 = zipfile.ZipFile(ROOT / "submissions/93_submit.zip")
    lib = (Path(out_a) / "model" / "opus_lib.py").read_bytes()
    assert lib == z93.read("model/opus_lib.py"), "build107 inference library differs from 93"
    out = ROOT / "submissions" / f"{num}_submit.zip"
    assert not out.exists(), out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as w:
        for info in z93.infolist():
            if info.filename.startswith("model/ranker_"):
                continue
            data = z93.read(info)
            if info.filename == "script.py":
                data = data.replace(b'(93_submit: ', b'(%d_submit: 93 with jamo/tonnage-noise Korean rankers; base ' % num, 1)
            w.writestr(info, data)
        for i, p in enumerate(rankers):
            w.write(p, f"model/ranker_{i}.txt")
    print(num, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
