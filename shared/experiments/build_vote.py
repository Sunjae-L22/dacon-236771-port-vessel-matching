# author: opus
"""Three-pipeline Korean vote: user's v18 (Korean 500), HanSunwook 1031_08 (495), ours 110 (493).

The three fail differently (v18: answer missing from candidates in 67% of its errors; HanSunwook: 20%, mostly
sister ships with typos; ours: easy-row precision 371/373 on Public), so a row-wise vote between models can
fix rows where one model is wrong and the other two agree. Voting is across models on the same row, never
across test rows.
  non-Korean rows                     -> 110 (English 3506)
  Korean, 110 gap >= GAP (sure)       -> 110
  other Korean rows                   -> answer shared by >= 2 of {v18, hsw, ours}; all differ -> TIE (ours / v18)
BASE may be 110 or 112 (112-style scripts report 110's gap, build_fuse.KC_EDITS).

Layout: model/ko = v18, model/hs = HanSunwook, model/en = 110 writing ko_gap (build_fuse.gap_script).
v18 and hsw run on Korean rows only; each package in its own process.

usage: python build_vote.py OUT.zip V18_ZIP HSW_ZIP BASE_ZIP [GAP=4] [TIE=ours]
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fuse import copy_into, gap_script  # noqa: E402

ROUTER = '''"""DACON entry point (3-way Korean vote v18 / HanSunwook / ours, ours kept when 110 gap >= {gap}, tie -> {tie}; see build_vote.py)."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GAP = {gap}
TIE = "{tie}"
RUN = ("import importlib.util, inspect, os, sys; p = sys.argv[1]; sys.path.insert(0, os.path.dirname(p)); "
       "s = importlib.util.spec_from_file_location('entry', p); m = importlib.util.module_from_spec(s); "
       "s.loader.exec_module(m); ps = inspect.signature(m.run).parameters; "
       "k = 'output_path' if 'output_path' in ps else 'out'; "
       "m.run(test_path=sys.argv[2], catalog_path=sys.argv[3], **{{k: sys.argv[4]}})")


def is_hangul(s):
    return any("\\uac00" <= c <= "\\ud7a3" for c in unicodedata.normalize("NFC", s))


def start(pkg, df, tmp, catalog):
    """Launch one package on its rows in its own process (the three run concurrently: GPU v20, CPU 110/112, CPU hsw)."""
    if df.empty:
        return None
    src, dst = os.path.join(tmp, pkg + "_test.csv"), os.path.join(tmp, pkg + "_pred.csv")
    df.to_csv(src, index=False, encoding="utf-8-sig")
    p = subprocess.Popen([sys.executable, "-c", RUN, str(HERE / "model" / pkg / "script.py"), src, catalog, dst],
                         cwd=str(HERE / "model" / pkg), env=dict(os.environ, PYTHONHASHSEED="0"))
    return pkg, p, dst, len(df), time.time()


def finish(job, sids, required, jobs):
    """Wait for one package; check it answered exactly its rows. A failed or incomplete voter (ko, hs) loses its
    votes (fallback towards ours); a failed 'en' (ours, every row) stops the run after killing the others."""
    if job is None:
        return pd.DataFrame(columns=["target_vessel_id"])
    pkg, p, dst, n, t = job
    try:
        if p.wait():
            raise RuntimeError(f"exit code {{p.returncode}}")
        d = pd.read_csv(dst, dtype={{"sample_id": str, "target_vessel_id": str}})
        if d["sample_id"].duplicated().any() or set(d["sample_id"]) != set(sids) or d["target_vessel_id"].isna().any():
            raise RuntimeError("rows missing, duplicated or empty")
        print(f"{{pkg}}: {{n}} rows in {{time.time() - t:.0f}}s", flush=True)
        return d.set_index("sample_id")
    except Exception as e:  # noqa: BLE001
        if required:
            for j in jobs:
                if j is not None and j[1].poll() is None:
                    j[1].kill()
                    j[1].wait()
            raise SystemExit(f"{{pkg}} failed: {{e}}")
        print(f"WARNING {{pkg}} failed ({{e}}); its votes are dropped", flush=True)
        return pd.DataFrame(columns=["target_vessel_id"])


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    catalog = str(Path(catalog_path).resolve())
    ko = test["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    tmp = tempfile.mkdtemp()
    try:
        jobs = []
        try:
            for pkg, df in (("en", test), ("ko", test[ko]), ("hs", test[ko])):
                jobs.append(start(pkg, df, tmp, catalog))
        except Exception:
            for j in jobs:
                if j is not None:
                    j[1].kill()
                    j[1].wait()
            raise
        en = finish(jobs[0], test["sample_id"], True, jobs)
        v18 = finish(jobs[1], test["sample_id"][ko], False, jobs)
        hs = finish(jobs[2], test["sample_id"][ko], False, jobs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sid = test["sample_id"]
    ours = en["target_vessel_id"].reindex(sid).to_numpy()
    gap = pd.to_numeric(en["ko_gap"], errors="coerce").reindex(sid).to_numpy()
    if not np.isfinite(gap[ko]).all():
        raise SystemExit("ours: missing Korean gap")
    a = v18["target_vessel_id"].reindex(sid).to_numpy() if len(v18) else ours
    b = hs["target_vessel_id"].reindex(sid).to_numpy() if len(hs) else ours
    pred, n = [], {{"sure": 0, "agree": 0, "tie": 0}}
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
    pd.DataFrame({{"sample_id": sid, "target_vessel_id": pred}}).to_csv(out, index=False)
    print(f"Saved {{len(test)}} predictions to {{out}} (Korean {{int(ko.sum())}}: {{n}})")


if __name__ == "__main__":
    run()
'''


def main(out, v18_zip, hsw_zip, base_zip, gap="4", tie="ours"):
    out = Path(out)
    assert not out.exists(), out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        req = copy_into(w, zipfile.ZipFile(v18_zip), "model/ko")
        req += copy_into(w, zipfile.ZipFile(hsw_zip), "model/hs")
        req += copy_into(w, zipfile.ZipFile(base_zip), "model/en", gap_script)
        w.writestr("script.py", ROUTER.format(gap=float(gap), tie=tie))
        w.writestr("requirements.txt", "\n".join(dict.fromkeys(req)) + "\n" if req else "# preinstalled packages only\n")
    print(out, round(out.stat().st_size / 1e9, 3), "GB", "requirements:", req)


if __name__ == "__main__":
    main(*sys.argv[1:])
