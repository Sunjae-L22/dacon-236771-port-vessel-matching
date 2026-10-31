# author: opus (build_vote5.py + v20 confidence)
"""V5_S1S3 plus v20's own confidence: on hard Korean rows that stay with v20 after rules S1/S3, take 112's answer
when v20's top-1 minus top-2 probability is below TAU (and, with GUARD, only if Erde does not back v20).

Public: F1 508, V4 (S1) 508, V5_S1S3 512, so S3 (112 == Erde != v20, hsw elsewhere) nets +4 on 42 rows. The
remaining v20 rows that 112 disputes are judged here by v20's margin, information no rule saw. v20's script is
patched (build_fuse_conf.V20_EDITS) to write a v20_margin column; a missing column -> no confidence overrides
(V5_S1S3 behaviour). TAU = 0 is V5_S1S3. Rules, fallbacks and layout as build_vote5.py.

usage: python build_vote7.py OUT.zip V20_ZIP HSW_ZIP ERDE_ZIP BASE_ZIP GAP TAU GUARD   (e.g. 4 0.1 1)
       python build_vote7.py --conf IN.zip OUT.zip TAU GUARD   (copy IN with only the router's TAU/GUARD line changed)
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fuse import copy_into, gap_script  # noqa: E402
from build_fuse_conf import v20_script  # noqa: E402

ROUTER = '''"""DACON entry point (hard Korean rows -> v20 with overrides S1,S3 and margin < TAU; 112 elsewhere; see build_vote7.py)."""
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
RULES = "S1,S3"
TAU, GUARD = {tau}, {guard}  # set by build_vote7.py
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
        margin = np.full(len(test), np.inf)
        if len(v20):
            if "v20_margin" in v20.columns:
                m = pd.to_numeric(v20["v20_margin"], errors="coerce").reindex(sid)
                margin = m.fillna(0.0).to_numpy(dtype=float)  # a row v20 answered without candidates: unsure
            else:
                print("WARNING ko wrote no v20_margin; no confidence overrides", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rules = set(RULES.split(","))
    pred, n = [], {{"sure": 0, "v20": 0, "asked": int(ask.sum()), "S1": 0, "S2": 0, "S3": 0, "S4": 0, "conf": 0, "low_any": 0, "low_guard": 0}}
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
        elif margin[i] < TAU and o != x:
            backed = e is not None and e == x
            n["low_any"] += 1
            n["low_guard"] += int(not backed)
            if not (GUARD and backed):
                pick = o
                n["conf"] += 1
            else:
                n["v20"] += 1
        else:
            n["v20"] += 1
        pred.append(pick)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"sample_id": sid, "target_vessel_id": pred}}).to_csv(out, index=False)
    print(f"Saved {{len(test)}} predictions to {{out}} (Korean {{int(ko.sum())}}: {{n}}, TAU {{TAU}} GUARD {{GUARD}})")


if __name__ == "__main__":
    run()
'''


def set_conf(src, out, tau, guard):
    """Copy a built ZIP, changing only the TAU/GUARD line of its router (no rebuild of the 7 GB package)."""
    out = Path(out)
    assert not out.exists(), out
    with zipfile.ZipFile(src) as z, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        for info in z.infolist():
            if info.filename == "script.py":
                lines = z.read(info).decode().splitlines(True)
                k = [i for i, l in enumerate(lines) if l.startswith("TAU, GUARD = ")]
                assert len(k) == 1, k
                lines[k[0]] = f"TAU, GUARD = {float(tau)}, {bool(int(guard))}  # set by build_vote7.py\n"
                w.writestr("script.py", "".join(lines))
                continue
            with z.open(info) as fsrc, w.open(zipfile.ZipInfo(info.filename, info.date_time), "w",
                                              force_zip64=info.file_size > 50_000_000) as fdst:
                while chunk := fsrc.read(1 << 24):
                    fdst.write(chunk)
    print(out, round(out.stat().st_size / 1e9, 3), "GB", "TAU =", float(tau), "GUARD =", bool(int(guard)))


def main(out, v20_zip, hsw_zip, erde_zip, base_zip, gap="4", tau="0.1", guard="1"):
    out = Path(out)
    assert not out.exists(), out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        req = copy_into(w, zipfile.ZipFile(v20_zip), "model/ko", v20_script)
        req += copy_into(w, zipfile.ZipFile(hsw_zip), "model/hs")
        req += copy_into(w, zipfile.ZipFile(erde_zip), "model/er")
        req += copy_into(w, zipfile.ZipFile(base_zip), "model/en", gap_script)
        w.writestr("script.py", ROUTER.format(gap=float(gap), tau=float(tau), guard=bool(int(guard))))
        w.writestr("requirements.txt", "\n".join(dict.fromkeys(req)) + "\n" if req else "# preinstalled packages only\n")
    print(out, round(out.stat().st_size / 1e9, 3), "GB", "requirements:", req)


if __name__ == "__main__":
    if sys.argv[1] == "--conf":
        set_conf(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
