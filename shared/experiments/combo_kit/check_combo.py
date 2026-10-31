# author: opus
"""Run a combo submission ZIP like the server would, on the local 3,000-row test in this kit.

Checks: it runs, wall time, output shape, English rows equal 93's saved predictions on the same file,
and row independence (a 600-row subset must give the same predictions).

usage: python check_combo.py COMBO.zip [WORKDIR]
"""
import os
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

KIT = Path(__file__).resolve().parent


def is_hangul(s):
    return isinstance(s, str) and any("가" <= c <= "힣" for c in unicodedata.normalize("NFC", s))


def run(work, df):
    shutil.rmtree(work / "data", ignore_errors=True)
    shutil.rmtree(work / "output", ignore_errors=True)
    (work / "data").mkdir()
    df.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8-sig")
    shutil.copy(KIT / "vessel_catalog.csv", work / "data" / "vessel_catalog.csv")
    t = time.time()
    p = subprocess.run([sys.executable, "script.py"], cwd=work)
    if p.returncode:
        raise SystemExit(f"script.py failed ({p.returncode})")
    return pd.read_csv(work / "output" / "submission.csv", dtype=str).set_index("sample_id")["target_vessel_id"], time.time() - t


def main(zip_path, workdir=None):
    work = Path(workdir or Path(zip_path).with_suffix("")).resolve()
    shutil.rmtree(work, ignore_errors=True)
    print("unzipping to", work, flush=True)
    zipfile.ZipFile(zip_path).extractall(work)
    test = pd.read_csv(KIT / "test_local3000.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    full, secs = run(work, test)
    assert len(full) == len(test) and full.index.tolist() == test["sample_id"].tolist(), "output rows/order"
    assert full.notna().all(), "empty predictions"
    ko = test.set_index("sample_id")["vessel_name"].map(is_hangul)
    exp = pd.read_csv(KIT / "expected_93_local.csv", dtype=str).set_index("sample_id")["target_vessel_id"]
    en_diff = int((full[~ko] != exp.reindex(full.index)[~ko]).sum())
    print(f"[1] full {len(full)} rows in {secs:.0f}s  (Korean rows {int(ko.sum())})")
    print(f"[2] English rows differing from 93: {en_diff} / {int((~ko).sum())}   (must be 0)")
    sub = test.sample(frac=0.2, random_state=5)
    part, _ = run(work, sub)
    mism = int((full.loc[part.index] != part).sum())
    print(f"[3] row independence: subset {len(part)} rows, mismatches {mism}   (must be 0)")
    print(f"zip size {os.path.getsize(zip_path) / 1e9:.2f} GB (limit 10GB)")
    shutil.rmtree(work, ignore_errors=True)
    print("OK" if en_diff == 0 and mism == 0 else "PROBLEM - do not submit")


if __name__ == "__main__":
    main(*sys.argv[1:])
