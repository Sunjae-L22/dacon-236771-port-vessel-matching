# author: opus
"""Mechanics check for 103 (call_sign pooling).

1. local_check's corrupted 3,000-row test has no repeated call_signs, so 103 must equal 93 there.
2. A grouped test (train rows, one new call_sign per vessel, half the names corrupted): every group
   must get one prediction, the full run must be order-invariant, and runtime is reported.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/src"))
from cv import DATA  # noqa: E402
from local_check import corrupted_test  # noqa: E402
import opus_lib as L  # noqa: E402


def run(zip_path, test):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        shutil.unpack_archive(zip_path, d)
        (d / "data").mkdir()
        shutil.copy(DATA / "vessel_catalog.csv", d / "data")
        test.to_csv(d / "data" / "test.csv", index=False, encoding="utf-8-sig")
        env = {k: v for k, v in os.environ.items() if k != "KO_DICT"}
        t = time.time()
        p = subprocess.run([sys.executable, "script.py"], cwd=d, env=env, capture_output=True, text=True)
        if p.returncode:
            raise SystemExit(p.stderr[-2000:])
        return pd.read_csv(d / "output/submission.csv").set_index("sample_id").target_vessel_id, time.time() - t


def main():
    z93, z103 = ROOT / "submissions/93_submit.zip", ROOT / "submissions/103_submit.zip"
    t = corrupted_test().drop(columns="target_vessel_id")
    a, _ = run(z93, t)
    b, secs = run(z103, t)
    print(f"[1] no repeated call_signs: 103 vs 93 differing rows {(a != b.reindex(a.index)).sum()} / {len(a)}  ({secs:.0f}s)")

    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    rng = np.random.default_rng(0)
    vids = rng.choice(tr.target_vessel_id.unique(), 400, replace=False)
    g = tr[tr.target_vessel_id.isin(vids)].copy()
    g["call_sign"] = "CS_G_" + g.target_vessel_id.str[-5:]
    c = rng.random(len(g)) < 0.5
    g.loc[c, "vessel_name"] = [L._corrupt_name(str(s), rng) for s in g.loc[c, "vessel_name"]]
    g = g.drop(columns="target_vessel_id")
    p1, secs = run(z103, g)
    p2, _ = run(z103, g.sample(frac=1, random_state=3))
    per_group = g.assign(pred=p1.reindex(g.sample_id).values).groupby("call_sign").pred.nunique()
    print(f"[2] grouped test rows {len(g)} groups {g.call_sign.nunique()}  ({secs:.0f}s)")
    print(f"    groups with >1 distinct prediction: {(per_group > 1).sum()}")
    print(f"    order change -> differing rows: {(p1 != p2.reindex(p1.index)).sum()}")


if __name__ == "__main__":
    main()
