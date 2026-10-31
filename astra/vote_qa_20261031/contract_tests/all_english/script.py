"""DACON entry point (3-way Korean vote v18 / HanSunwook / ours, ours kept when 110 gap >= 4.0, tie -> ko; see build_vote.py)."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata

import pandas as pd

HERE = Path(__file__).resolve().parent
GAP = 4.0
TIE = "ko"
RUN = ("import importlib.util, inspect, os, sys; p = sys.argv[1]; sys.path.insert(0, os.path.dirname(p)); "
       "s = importlib.util.spec_from_file_location('entry', p); m = importlib.util.module_from_spec(s); "
       "s.loader.exec_module(m); ps = inspect.signature(m.run).parameters; "
       "k = 'output_path' if 'output_path' in ps else 'out'; "
       "m.run(test_path=sys.argv[2], catalog_path=sys.argv[3], **{k: sys.argv[4]})")


def is_hangul(s):
    return any("\uac00" <= c <= "\ud7a3" for c in unicodedata.normalize("NFC", s))


def start(pkg, df, tmp, catalog):
    """Launch one package on its rows in its own process (the three run concurrently: GPU v20, CPU 110/112, CPU hsw)."""
    if df.empty:
        return None
    src, dst = os.path.join(tmp, pkg + "_test.csv"), os.path.join(tmp, pkg + "_pred.csv")
    df.to_csv(src, index=False, encoding="utf-8-sig")
    p = subprocess.Popen([sys.executable, "-c", RUN, str(HERE / "model" / pkg / "script.py"), src, catalog, dst],
                         cwd=str(HERE / "model" / pkg), env=dict(os.environ, PYTHONHASHSEED="0"))
    return pkg, p, dst, len(df), time.time()


def finish(job):
    if job is None:
        return pd.DataFrame(columns=["target_vessel_id"])
    pkg, p, dst, n, t = job
    if p.wait():
        raise SystemExit(f"{pkg} failed with code {p.returncode}")
    print(f"{pkg}: {n} rows in {time.time() - t:.0f}s", flush=True)
    return pd.read_csv(dst, dtype={"sample_id": str, "target_vessel_id": str}).set_index("sample_id")


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    catalog = str(Path(catalog_path).resolve())
    ko = test["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    tmp = tempfile.mkdtemp()
    try:
        jobs = [start("en", test, tmp, catalog), start("ko", test[ko], tmp, catalog), start("hs", test[ko], tmp, catalog)]
        en, v18, hs = (finish(j) for j in jobs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sid = test["sample_id"]
    ours = en["target_vessel_id"].reindex(sid).to_numpy()
    gap = pd.to_numeric(en["ko_gap"], errors="coerce").reindex(sid).to_numpy()
    a = v18["target_vessel_id"].reindex(sid).to_numpy() if len(v18) else ours
    b = hs["target_vessel_id"].reindex(sid).to_numpy() if len(hs) else ours
    pred, n = [], {"sure": 0, "agree": 0, "tie": 0}
    for i in range(len(test)):
        if not ko[i]:
            pred.append(ours[i])
        elif gap[i] >= GAP:
            pred.append(ours[i]); n["sure"] += 1
        elif b[i] == ours[i] or a[i] == b[i] or a[i] == ours[i]:
            pred.append(b[i] if b[i] in (ours[i], a[i]) else a[i]); n["agree"] += 1
        else:
            pred.append(ours[i] if TIE == "ours" else a[i]); n["tie"] += 1
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": sid, "target_vessel_id": pred}).to_csv(out, index=False)
    print(f"Saved {len(test)} predictions to {out} (Korean {int(ko.sum())}: {n})")


if __name__ == "__main__":
    run()
