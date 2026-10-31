# author: opus
"""Combo submission: Korean rows by one finished submission, all other rows by another, row by row.

Decoded Public: user's v18_fuzzy500 (ksh pipeline) = Korean 500 / English 3448; our 93 = Korean 492 /
English 3506. Korean v18 + English 93 -> 500 / 3506 = 0.95268 if each pipeline keeps its own rows' answers.

Layout: model/ko/ = the Korean submission unpacked (its script.py + model/), model/en/ = the other one.
The router splits data/test.csv by the competition's Korean rule (a Hangul syllable in the NFC name),
writes each part to a temp dir, runs each package's own run() in a separate process (separate memory and
GPU), and merges in the original row order. A row's route and prediction depend only on that row.

usage: python3 build_combo.py NUM KO_ZIP EN_ZIP
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ROUTER = '''"""DACON entry point ({num}_submit: Korean rows -> model/ko ({ko}), other rows -> model/en ({en}); see build_combo.py)."""
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
RUN = ("import importlib.util, sys; p = sys.argv[1]; sys.path.insert(0, p.rsplit('/', 1)[0]); "
       "s = importlib.util.spec_from_file_location('entry', p); m = importlib.util.module_from_spec(s); "
       "s.loader.exec_module(m); m.run(sys.argv[2], sys.argv[3], sys.argv[4])")


def is_hangul(s):
    return any("\\uac00" <= c <= "\\ud7a3" for c in unicodedata.normalize("NFC", s))


def part(pkg, df, tmp, catalog):
    if df.empty:
        return pd.Series(dtype=object)
    src, dst = os.path.join(tmp, pkg + "_test.csv"), os.path.join(tmp, pkg + "_pred.csv")
    df.to_csv(src, index=False, encoding="utf-8-sig")
    t = time.time()
    env = dict(os.environ, PYTHONHASHSEED="0")
    p = subprocess.run([sys.executable, "-c", RUN, str(HERE / "model" / pkg / "script.py"), src, catalog, dst],
                       cwd=str(HERE / "model" / pkg), env=env)
    if p.returncode:
        raise SystemExit(f"{{pkg}} failed with code {{p.returncode}}")
    print(f"{{pkg}}: {{len(df)}} rows in {{time.time() - t:.0f}}s", flush=True)
    return pd.read_csv(dst, dtype=str).set_index("sample_id")["target_vessel_id"]


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    catalog = str(Path(catalog_path).resolve())
    ko = test["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    tmp = tempfile.mkdtemp()
    try:
        pred = pd.concat([part("ko", test[ko], tmp, catalog), part("en", test[~ko], tmp, catalog)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"sample_id": test["sample_id"], "target_vessel_id": pred.reindex(test["sample_id"]).values}}).to_csv(out, index=False)
    print(f"Saved {{len(test)}} predictions to {{out}} (Korean {{int(ko.sum())}})")


if __name__ == "__main__":
    run()
'''


def main(num, ko_zip, en_zip):
    out = ROOT / "submissions" / f"{num}_submit.zip"
    assert not out.exists(), out
    req = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        for pkg, src in (("ko", ko_zip), ("en", en_zip)):
            z = zipfile.ZipFile(src)
            names = z.namelist()
            assert "script.py" in names, (src, "script.py must be at the top level")
            for info in z.infolist():
                if info.is_dir():
                    continue
                if info.filename == "requirements.txt":
                    req += [l for l in z.read(info).decode().splitlines() if l.strip() and not l.startswith("#")]
                    continue
                big = info.file_size > 50_000_000
                with z.open(info) as fsrc, w.open(zipfile.ZipInfo(f"model/{pkg}/{info.filename}", info.date_time),
                                                  "w", force_zip64=big) as fdst:
                    while chunk := fsrc.read(1 << 24):
                        fdst.write(chunk)
        w.writestr("script.py", ROUTER.format(num=num, ko=Path(ko_zip).name, en=Path(en_zip).name))
        w.writestr("requirements.txt", "\n".join(dict.fromkeys(req)) + "\n" if req else
                   "# preinstalled packages only (both parts)\n")
    print(num, round(out.stat().st_size / 1e9, 3), "GB", "requirements:", req)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
