# author: opus
"""108: a base submission (107 by default) with its twelve English models replaced by build108 models
(same en65 / s62 recipes + one English copy with Latin typos and tonnage x U(0.8, 1.2); see
pinned/build108_63/make_submit.py and pinned/build108_31/make_submit.py:augment_en).

Why: opus/stage2/train_en.py — on folds 0/2/4 held-out English rows, "two typos + tonnage x U(.85,1.15)"
gains +23/2100 and clean names -2/2100.
The libraries differ from 93's only in the training-time KO_DICT default (and build31's training
augmentation), so the base ZIP's inference code is kept.

usage: python3 build_108_en.py NUM BASE_NUM OUT63_A OUT63_B OUT31_A OUT31_B
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(num, base, a63, b63, a31, b31):
    new = {f"model/en65_{i}.txt": Path(d) / "model" / f"ranker_{j}.txt"
           for i, (d, j) in enumerate((d, j) for d in (a63, b63) for j in range(3))}
    new.update({f"model/s62_{i}.txt": Path(d) / "model" / f"ranker_{j}.txt"
                for i, (d, j) in enumerate((d, j) for d in (a31, b31) for j in range(3))})
    assert all(p.exists() for p in new.values())
    zb = zipfile.ZipFile(ROOT / "submissions" / f"{base}_submit.zip")
    out = ROOT / "submissions" / f"{num}_submit.zip"
    assert not out.exists(), out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as w:
        for info in zb.infolist():
            if info.filename in new:
                continue
            data = zb.read(info)
            if info.filename == "script.py":
                data = data.replace(b'_submit: ', b'_submit: English models retrained with typo/tonnage noise; base %d: ' % base, 1)
                data = data.replace(b'(%d_submit' % base, b'(%d_submit' % num, 1)
            w.writestr(info, data)
        for name, p in new.items():
            w.write(p, name)
    print(num, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), *sys.argv[3:])
