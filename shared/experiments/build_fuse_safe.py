# author: opus (final-night variant of build_fuse.py)
"""Guarded fusion. Non-Korean rows and Korean rows where BASE's 110 gap >= GAP -> BASE (112-style ZIP);
other Korean rows -> KO (v20). Both packages start at once.

BASE is required. KO is optional: same CPU priority as BASE (so C_vote's server time bounds this router's),
oom_score_adj 1000, and a timer thread kills KO's process group BUDGET_MIN minutes after the run's start
anchor = the oldest inode-change time (ctime) of the unpacked files (unzip sets it at extraction), credited up
to 10 min; set BUDGET_MIN = min(24, 28.5 - (C_vote server minutes - 11.8)). A KO that crashed, was killed or
wrote an unusable file gives BASE's answers on its rows; missing/empty rows fall back one by one.

usage: python build_fuse_safe.py OUT.zip KO_ZIP BASE_ZIP [GAP=4] [BUDGET_MIN=24]
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fuse import copy_into, gap_script  # noqa: E402

ROUTER = '''"""DACON entry point (guarded fusion: Korean rows -> model/ko unless model/en gap >= {gap}; model/ko optional, OOM-first, killed {budget:.0f} s after the unpack anchor; see build_fuse_safe.py)."""
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def start_anchor(here, credit_max=600.0):
    """Oldest ctime among the unpacked files (= start of extraction), if it lies within the last credit_max
    seconds; otherwise now (files unpacked long before this run earn no credit)."""
    now = time.time()
    try:
        c = min(os.stat(os.path.join(d, f)).st_ctime for d, _, fs in os.walk(here) for f in fs)
    except (OSError, ValueError):
        return now
    return c if now - credit_max <= c <= now else now


T0 = start_anchor(HERE)
GAP = float("{gap}")
BUDGET = {budget}
RUNNER = """
import importlib.util, inspect, os, sys
p, test, cat, out = sys.argv[1:5]
sys.path.insert(0, os.path.dirname(p))
s = importlib.util.spec_from_file_location("entry", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
ps = list(inspect.signature(m.run).parameters)
if "test_path" in ps and "catalog_path" in ps and len(ps) >= 3:
    kw = dict(test_path=test, catalog_path=cat)
    kw["output_path" if "output_path" in ps else ("out" if "out" in ps else ps[2])] = out
    m.run(**kw)
else:
    m.run(test, cat, out)
"""


def is_hangul(s):
    return any("\\uac00" <= c <= "\\ud7a3" for c in unicodedata.normalize("NFC", s))


def oom_first():
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def start(pkg, df, tmp, optional):
    src, dst = os.path.join(tmp, pkg + "_test.csv"), os.path.join(tmp, pkg + "_pred.csv")
    df.to_csv(src, index=False, encoding="utf-8-sig")
    p = subprocess.Popen([sys.executable, "-c", RUNNER, str(HERE / "model" / pkg / "script.py"), src, CATALOG, dst],
                         cwd=str(HERE / "model" / pkg), env=dict(os.environ, PYTHONHASHSEED="0"),
                         start_new_session=True, preexec_fn=oom_first if optional else None)
    return pkg, p, dst, len(df), time.time()


def stop(job):
    p = job[1]
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            p.kill()
        p.wait()


def watchdog(job):
    if job[1].poll() is None:
        print(f"WARNING {{job[0]}} still running {{time.time() - T0:.0f}}s after start; killed", flush=True)
        stop(job)


def collect(job, timeout):
    pkg, p, dst, n, t = job
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop(job)
        raise RuntimeError("still running at the time budget, killed")
    if rc:
        raise RuntimeError(f"exit code {{rc}}")
    d = pd.read_csv(dst, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    print(f"{{pkg}}: {{n}} rows in {{time.time() - t:.0f}}s", flush=True)
    return d


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    global CATALOG
    print(f"router start offset {{time.time() - T0:.0f}}s (budget {{BUDGET:.0f}}s from the anchor)", flush=True)
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    CATALOG = str(Path(catalog_path).resolve())
    sid = test["sample_id"].astype(str)
    ko = test["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    tmp = tempfile.mkdtemp()
    jobs, timer = [], None
    try:
        jobs.append(start("en", test, tmp, False))
        if ko.any():
            try:
                jobs.append(start("ko", test[ko], tmp, True))
                timer = threading.Timer(max(1.0, T0 + BUDGET - time.time()), watchdog, args=(jobs[1],))
                timer.daemon = True
                timer.start()
            except Exception as e:  # noqa: BLE001
                print(f"WARNING ko did not start ({{e}}); Korean rows fall back to en", flush=True)
        try:
            en = collect(jobs[0], None)
            if en["sample_id"].duplicated().any() or set(en["sample_id"]) != set(sid) \\
                    or (en["target_vessel_id"].str.strip() == "").any():
                raise RuntimeError("rows missing, duplicated or empty")
        except Exception as e:  # noqa: BLE001
            for j in jobs:
                stop(j)
            raise SystemExit(f"en failed: {{e}}")
        en = en.set_index("sample_id")
        ours = en["target_vessel_id"].reindex(sid).to_numpy()
        gap = pd.to_numeric(en["ko_gap"], errors="coerce").reindex(sid).to_numpy()
        if not np.isfinite(gap[ko]).all():
            for j in jobs:
                stop(j)
            raise SystemExit("en failed: missing Korean gap")
        theirs = np.array([""] * len(test), dtype=object)
        if len(jobs) > 1:
            try:
                d = collect(jobs[1], max(1.0, T0 + BUDGET + 30 - time.time()))  # backstop; the watchdog acts first
                d = d[d["target_vessel_id"].str.strip() != ""].drop_duplicates("sample_id", keep=False)
                theirs = d.set_index("sample_id")["target_vessel_id"].reindex(sid).fillna("").to_numpy(dtype=object)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING ko failed ({{e}}); Korean rows fall back to en", flush=True)
    finally:
        if timer is not None:
            timer.cancel()
        for j in jobs:
            stop(j)
        shutil.rmtree(tmp, ignore_errors=True)
    want = ko & ~(gap >= GAP)
    use = want & (theirs != "")
    pred = np.where(use, theirs, ours)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"sample_id": sid, "target_vessel_id": pred}}).to_csv(out, index=False)
    print(f"Saved {{len(test)}} predictions to {{out}} (Korean {{int(ko.sum())}}, routed to ko {{int(want.sum())}}, "
          f"answered by ko {{int(use.sum())}}, fell back {{int((want & ~use).sum())}}, {{time.time() - T0:.0f}}s)")


if __name__ == "__main__":
    run()
'''


def main(out, ko_zip, base_zip, gap="4", budget_min="24"):
    out = Path(out)
    assert not out.exists(), out
    assert 18 <= float(budget_min) <= 24, "BUDGET_MIN must be 18-24 (see Step 0.2)"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        req = copy_into(w, zipfile.ZipFile(ko_zip), "model/ko")
        req += copy_into(w, zipfile.ZipFile(base_zip), "model/en", gap_script)
        w.writestr("script.py", ROUTER.format(gap=float(gap), budget=float(budget_min) * 60))
        w.writestr("requirements.txt", "\n".join(dict.fromkeys(req)) + "\n" if req else "# preinstalled packages only\n")
    print(out, round(out.stat().st_size / 1e9, 3), "GB", "requirements:", req)


if __name__ == "__main__":
    main(*sys.argv[1:])
