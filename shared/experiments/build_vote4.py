# author: opus
"""F1 plus a strict override: hard Korean rows take v20's answer unless 112, HanSunwook and Erde all agree on another.

Public: F1 (hard Korean rows -> v20) 508 beat C_vote (2-of-{v20, hsw, 112}) 507, so when hsw and 112 agree against
v20, v20 is still right slightly more often. Requiring a third, differently built model (Erde v42b, EM transducer +
PMI + LightGBM) to agree as well keeps only the strongest of those overrides. Voting is across models on the same
row, never across test rows.
  non-Korean rows                          -> 112 (English 3506)
  Korean, 110 gap >= GAP (sure)            -> 112
  other Korean rows                        -> v20, except: hsw == 112 == Erde != v20 -> 112
v20, hsw and 112 run concurrently on their rows; Erde then runs only on the rows where hsw == 112 != v20 (its
per-row output does not depend on which rows it is given, checked by Astra 017). A failed v20 -> 112 on every row;
a failed hsw or Erde -> no override (F1 behaviour); a failed 112 stops the run.

Layout: model/ko = v20, model/hs = HanSunwook, model/er = Erde, model/en = 112 writing ko_gap (build_fuse.gap_script).

usage: python build_vote4.py OUT.zip V20_ZIP HSW_ZIP ERDE_ZIP BASE_ZIP [GAP=4]
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fuse import copy_into, gap_script  # noqa: E402

ROUTER = '''"""DACON entry point (hard Korean rows -> v20 unless 112 == HanSunwook == Erde; 112 elsewhere; see build_vote4.py)."""
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
RUN = ("import importlib.util, inspect, os, sys; p = sys.argv[1]; sys.path.insert(0, os.path.dirname(p)); "
       "s = importlib.util.spec_from_file_location('entry', p); m = importlib.util.module_from_spec(s); "
       "s.loader.exec_module(m); ps = inspect.signature(m.run).parameters; "
       "k = 'output_path' if 'output_path' in ps else 'out'; "
       "m.run(test_path=sys.argv[2], catalog_path=sys.argv[3], **{{k: sys.argv[4]}})")


def is_hangul(s):
    return any("\\uac00" <= c <= "\\ud7a3" for c in unicodedata.normalize("NFC", s))


def start(pkg, df, tmp, catalog):
    """Launch one package on its rows in its own process."""
    if df.empty:
        return None
    src, dst = os.path.join(tmp, pkg + "_test.csv"), os.path.join(tmp, pkg + "_pred.csv")
    df.to_csv(src, index=False, encoding="utf-8-sig")
    p = subprocess.Popen([sys.executable, "-c", RUN, str(HERE / "model" / pkg / "script.py"), src, catalog, dst],
                         cwd=str(HERE / "model" / pkg), env=dict(os.environ, PYTHONHASHSEED="0"))
    return pkg, p, dst, len(df), time.time()


def finish(job, sids, required, jobs):
    """Wait for one package; check it answered exactly its rows. A failed or incomplete optional package loses its
    say; a failed 'en' (112, every row) stops the run after killing the others."""
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
        print(f"WARNING {{pkg}} failed ({{e}}); its answers are dropped", flush=True)
        return pd.DataFrame(columns=["target_vessel_id"])


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    catalog = str(Path(catalog_path).resolve())
    ko = test["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    sid = test["sample_id"]
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
        en = finish(jobs[0], sid, True, jobs)
        v20 = finish(jobs[1], sid[ko], False, jobs)
        hs = finish(jobs[2], sid[ko], False, jobs)
        ours = en["target_vessel_id"].reindex(sid).to_numpy()
        gap = pd.to_numeric(en["ko_gap"], errors="coerce").reindex(sid).to_numpy()
        if not np.isfinite(gap[ko]).all():
            raise SystemExit("ours: missing Korean gap")
        a = v20["target_vessel_id"].reindex(sid).to_numpy() if len(v20) else ours
        b = hs["target_vessel_id"].reindex(sid).to_numpy() if len(hs) else a
        ask = ko & (gap < GAP) & (b == ours) & (a != ours)
        er = finish(start("er", test[ask], tmp, catalog), sid[ask], False, [])
        c = er["target_vessel_id"].reindex(sid).to_numpy() if len(er) else a
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pred, n = [], {{"sure": 0, "v20": 0, "asked": int(ask.sum()), "override": 0}}
    for i in range(len(test)):
        if not ko[i]:
            pred.append(ours[i])
        elif gap[i] >= GAP:
            pred.append(ours[i]); n["sure"] += 1
        elif ask[i] and c[i] == ours[i]:
            pred.append(ours[i]); n["override"] += 1
        else:
            pred.append(a[i]); n["v20"] += 1
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"sample_id": sid, "target_vessel_id": pred}}).to_csv(out, index=False)
    print(f"Saved {{len(test)}} predictions to {{out}} (Korean {{int(ko.sum())}}: {{n}})")


if __name__ == "__main__":
    run()
'''


def main(out, v20_zip, hsw_zip, erde_zip, base_zip, gap="4"):
    out = Path(out)
    assert not out.exists(), out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        req = copy_into(w, zipfile.ZipFile(v20_zip), "model/ko")
        req += copy_into(w, zipfile.ZipFile(hsw_zip), "model/hs")
        req += copy_into(w, zipfile.ZipFile(erde_zip), "model/er")
        req += copy_into(w, zipfile.ZipFile(base_zip), "model/en", gap_script)
        w.writestr("script.py", ROUTER.format(gap=float(gap)))
        w.writestr("requirements.txt", "\n".join(dict.fromkeys(req)) + "\n" if req else "# preinstalled packages only\n")
    print(out, round(out.stat().st_size / 1e9, 3), "GB", "requirements:", req)


if __name__ == "__main__":
    main(*sys.argv[1:])
