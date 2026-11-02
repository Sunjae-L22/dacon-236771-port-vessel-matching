# author: opus
"""97/98: 93 with 36's twelve Korean rankers added to the Korean route (no retraining).

36 and 72 rankers share inference code and stats (opus_lib differs only in the training-time
augment_corrupt tonnage noise), so the 36 boosters score the same 60 features.
  97: family means weighted 0.5 (72) / 0.5 (36)
  98: all 18 boosters weighted equally (72 gets 1/3)
"""
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from build_93_decomp import COMMON, SCRIPT  # noqa: E402

SCRIPT = SCRIPT.replace("m.predict(feats[fam]) for m in models[fam]) for fam, w in KO.items()",
                        "m.predict(feats.get(fam, F60)) for m in models[fam]) for fam, w in KO.items()")  # the ko36 family reads F60

EN = {"en65": 1.5, "s62": 1.0}
VARIANTS = {
    97: {"ranker": 0.5 / 6, "r36": 0.5 / 12},
    98: {"ranker": 1.0, "r36": 1.0},
}


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        with zipfile.ZipFile(ROOT / "submissions" / "93_submit.zip") as z:
            z.extractall(d)
        with zipfile.ZipFile(ROOT / "submissions" / "36_submit.zip") as z:
            for i in range(12):
                (d / "model" / f"r36_{i}.txt").write_bytes(z.read(f"model/ranker_{i}.txt"))
        for num, ko in VARIANTS.items():
            out = ROOT / "submissions" / f"{num}_submit.zip"
            assert not out.exists(), out
            files = list(COMMON)
            for fam in list(ko) + list(EN):
                files += sorted(str(p.relative_to(d)) for p in (d / "model").glob(fam + "_*.txt"))
            script = SCRIPT.format(num=num, ko=json.dumps(ko), compose=False, en=json.dumps(EN))
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    z.write(d / f, f)
                z.writestr("script.py", script)
            print(num, round(out.stat().st_size / 1e6, 2), "MB", len(files) + 1, "files")


if __name__ == "__main__":
    main()
