# author: opus
"""Run a submission ZIP the way the server would, on a corrupted 3,000-row test built from train.

Checks: offline run without KO_DICT set, wall time, output shape, and that each row's prediction
is unchanged when the rows are shuffled and subset (competition rule).

usage: python3 shared/src/local_check.py <submission.zip>
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cv import DATA, score  # noqa: E402
import opus_lib as L  # noqa: E402


def corrupted_test(seed=7, n=3000):
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    t = tr.sample(n, random_state=seed).copy()
    rng = np.random.default_rng(seed)
    t["call_sign"] = np.where(rng.random(n) < 0.5, None, "CS_NEW_" + t.index.astype(str))
    c = rng.random(n) < 0.4
    t.loc[c, "vessel_name"] = [L._corrupt_name(s, rng) for s in t.loc[c, "vessel_name"]]
    g = rng.random(n) < 0.2
    t.loc[g, ["gross_tonnage", "international_gross_tonnage"]] = np.nan
    d = rng.random(n) < 0.1
    t.loc[d, "vessel_name"] = t.loc[d, "vessel_name"].map(lambda s: unicodedata.normalize("NFD", s))
    return t


def run(workdir, env):
    t0 = time.time()
    p = subprocess.run([sys.executable, "script.py"], cwd=workdir, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(p.stderr[-3000:])
    return time.time() - t0


def main(zip_path):
    env = {k: v for k, v in os.environ.items() if k != "KO_DICT"}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        shutil.unpack_archive(zip_path, d)
        (d / "data").mkdir()
        shutil.copy(DATA / "vessel_catalog.csv", d / "data")
        t = corrupted_test()
        t.sample(frac=1, random_state=1).drop(columns="target_vessel_id").to_csv(
            d / "data" / "test.csv", index=False, encoding="utf-8-sig")
        secs = run(d, env)
        full = pd.read_csv(d / "output" / "submission.csv")
        assert list(full.columns) == ["sample_id", "target_vessel_id"] and len(full) == len(t)
        truth = t.set_index("sample_id").loc[full.sample_id]
        print(f"rows {len(full)}  time {secs:.1f}s  score on corrupted train rows "
              f"{score(truth.vessel_name, truth.target_vessel_id, full.target_vessel_id)}")
        sub = pd.read_csv(d / "data" / "test.csv", encoding="utf-8-sig").sample(frac=0.2, random_state=5)
        sub.to_csv(d / "data" / "test.csv", index=False, encoding="utf-8-sig")
        shutil.rmtree(d / "output")
        run(d, env)
        part = pd.read_csv(d / "output" / "submission.csv").set_index("sample_id")
        mism = (full.set_index("sample_id").loc[part.index].target_vessel_id != part.target_vessel_id).sum()
        print(f"row independence: subset {len(part)} rows, mismatches {mism}")
    print(f"zip size {Path(zip_path).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
