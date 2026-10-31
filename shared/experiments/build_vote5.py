# author: opus
"""F1 plus selectable overrides on hard Korean rows (v20 by default), decided by 112 (o), HanSunwook (h) and Erde (e).

Rules (comma list RULES; a = v20):
  S1: h == o == e != a  -> o        (V4)
  S2: h == o != a, e != o -> o      (C_vote's other overrides; Public says S1+S2 together lose 1 vs F1)
  S3: o == e != a, h != o -> o
  S4: h == e != a, o != h -> h
Each rule is measured on Public against F1 (508), one ZIP per rule set, cut from one build with --rules.
v20, hsw and 112 run concurrently; Erde then runs only on the hard rows where v20 disagrees with 112 or HanSunwook
(its per-row output does not depend on which rows it is given, checked by Astra 017). A failed v20 -> 112 on every
row; a failed hsw or Erde -> no override (F1 behaviour); a failed 112 stops the run. Voting is across models on the
same row, never across test rows.

Layout: model/ko = v20, model/hs = HanSunwook, model/er = Erde, model/en = 112 writing ko_gap (build_fuse.gap_script).

usage: python build_vote5.py OUT.zip V20_ZIP HSW_ZIP ERDE_ZIP BASE_ZIP GAP RULES      (e.g. 4 S1,S3)
       python build_vote5.py --rules IN.zip OUT.zip RULES   (copy IN with only the router's RULES changed)
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fuse import copy_into, gap_script  # noqa: E402

ROUTER = '''"""DACON entry point (hard Korean rows -> v20 with overrides RULES; 112 elsewhere; see build_vote5.py)."""
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
RULES = "{rules}"  # set by build_vote5.py
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
        ask = ko & (gap < GAP) & ((a != ours) | (a != b)) & (len(v20) > 0) & (len(hs) > 0)
        er = finish(start("er", test[ask], tmp, catalog), sid[ask], False, [])
        c = er["target_vessel_id"].reindex(sid).to_numpy() if len(er) else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rules = set(RULES.split(","))
    pred, n = [], {{"sure": 0, "v20": 0, "asked": int(ask.sum()), "S1": 0, "S2": 0, "S3": 0, "S4": 0}}
    for i in range(len(test)):
        if not ko[i]:
            pred.append(ours[i]); continue
        if gap[i] >= GAP:
            pred.append(ours[i]); n["sure"] += 1; continue
        o, h, e, x = ours[i], b[i], (c[i] if c is not None and ask[i] else None), a[i]
        pick, rule = x, None
        if e is not None:
            if h == o == e != x:
                rule = "S1"
            elif h == o != x and e != o:
                rule = "S2"
            elif o == e != x and h != o:
                rule = "S3"
            elif h == e != x and o != h:
                rule = "S4"
        if rule in rules:
            pick = h if rule == "S4" else o
            n[rule] += 1
        else:
            n["v20"] += 1
        pred.append(pick)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"sample_id": sid, "target_vessel_id": pred}}).to_csv(out, index=False)
    print(f"Saved {{len(test)}} predictions to {{out}} (Korean {{int(ko.sum())}}: {{n}})")


if __name__ == "__main__":
    run()
'''


def set_rules(src, out, rules):
    """Copy a built ZIP, changing only the RULES line of its router (no rebuild of the 7 GB package)."""
    out = Path(out)
    assert not out.exists(), out
    with zipfile.ZipFile(src) as z, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        for info in z.infolist():
            if info.filename == "script.py":
                lines = z.read(info).decode().splitlines(True)
                k = [i for i, l in enumerate(lines) if l.startswith("RULES = ")]
                assert len(k) == 1, k
                lines[k[0]] = f'RULES = "{rules}"  # set by build_vote5.py\n'
                w.writestr("script.py", "".join(lines))
                continue
            with z.open(info) as fsrc, w.open(zipfile.ZipInfo(info.filename, info.date_time), "w",
                                              force_zip64=info.file_size > 50_000_000) as fdst:
                while chunk := fsrc.read(1 << 24):
                    fdst.write(chunk)
    print(out, round(out.stat().st_size / 1e9, 3), "GB", "RULES =", rules)


def main(out, v20_zip, hsw_zip, erde_zip, base_zip, gap="4", rules="S1"):
    out = Path(out)
    assert not out.exists(), out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        req = copy_into(w, zipfile.ZipFile(v20_zip), "model/ko")
        req += copy_into(w, zipfile.ZipFile(hsw_zip), "model/hs")
        req += copy_into(w, zipfile.ZipFile(erde_zip), "model/er")
        req += copy_into(w, zipfile.ZipFile(base_zip), "model/en", gap_script)
        assert set(rules.split(",")) <= {"S1", "S2", "S3", "S4"}, rules
        w.writestr("script.py", ROUTER.format(gap=float(gap), rules=rules))
        w.writestr("requirements.txt", "\n".join(dict.fromkeys(req)) + "\n" if req else "# preinstalled packages only\n")
    print(out, round(out.stat().st_size / 1e9, 3), "GB", "requirements:", req)


if __name__ == "__main__":
    if sys.argv[1] == "--rules":
        set_rules(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
