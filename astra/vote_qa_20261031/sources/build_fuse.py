# author: opus
"""Fusion submission: user's v18 on Korean rows, except rows our 110 ranker is sure about.

102 measured on Public that Korean rows whose 93 leader / runner-up gap (mean of the six Korean rankers,
train targets excluded) is >= 4 are right 371 / 373 times. v18 (Korean 500 / 532) is the better model on the
rest. So: non-Korean rows -> 110 (= 93 English); Korean rows with gap >= GAP -> 110; other Korean rows -> v18.
Worst case vs pure v18 Korean: -2 (110's two easy-row errors); every v18 error on our easy rows is fixed.

model/en = 110 with its script writing an extra ko_gap column (internal, never the final output);
model/ko = v18 unpacked. Each runs in its own process; routing uses only the row itself.

usage: python build_fuse.py OUT.zip V18_ZIP BASE110_ZIP [GAP=4]
"""
import sys
import zipfile
from pathlib import Path

EN_EDITS = [
    (b"    preds = []\n", b"    preds, gaps = [], []\n"),
    (b"            preds.append(ix.ids[cs])\n            continue\n",
     b"            preds.append(ix.ids[cs])\n            gaps.append(99.0)\n            continue\n"),
    (b"        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))\n"
     b"        best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])\n",
     b"        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))\n"
     b"        allowed = [j for j in order if idx[j] not in train_targets] or order\n"
     b"        best = idx[allowed[0]]\n"
     b"        if L.is_hangul(q):\n"
     b"            n_ko = len(models[\"ranker\"])\n"
     b"            gaps.append(float(score[allowed[0]] - score[allowed[1]]) / n_ko if len(allowed) > 1 else 99.0)\n"
     b"        else:\n"
     b"            gaps.append(float(\"nan\"))\n"),
    (b'pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds})',
     b'pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds, "ko_gap": gaps})'),
]

ROUTER = '''"""DACON entry point (fusion: Korean rows -> v18 unless 110's gap >= {gap}; other rows -> 110; see build_fuse.py)."""
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
GAP = {gap}
RUN = ("import importlib.util, os, sys; p = sys.argv[1]; sys.path.insert(0, os.path.dirname(p)); "
       "s = importlib.util.spec_from_file_location('entry', p); m = importlib.util.module_from_spec(s); "
       "s.loader.exec_module(m); m.run(sys.argv[2], sys.argv[3], sys.argv[4])")


def is_hangul(s):
    return any("\\uac00" <= c <= "\\ud7a3" for c in unicodedata.normalize("NFC", s))


def part(pkg, df, tmp, catalog):
    if df.empty:
        return pd.DataFrame(columns=["target_vessel_id"])
    src, dst = os.path.join(tmp, pkg + "_test.csv"), os.path.join(tmp, pkg + "_pred.csv")
    df.to_csv(src, index=False, encoding="utf-8-sig")
    t = time.time()
    p = subprocess.run([sys.executable, "-c", RUN, str(HERE / "model" / pkg / "script.py"), src, catalog, dst],
                       cwd=str(HERE / "model" / pkg), env=dict(os.environ, PYTHONHASHSEED="0"))
    if p.returncode:
        raise SystemExit(f"{{pkg}} failed with code {{p.returncode}}")
    print(f"{{pkg}}: {{len(df)}} rows in {{time.time() - t:.0f}}s", flush=True)
    return pd.read_csv(dst, dtype={{"sample_id": str, "target_vessel_id": str}}).set_index("sample_id")


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    catalog = str(Path(catalog_path).resolve())
    ko = test["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    tmp = tempfile.mkdtemp()
    try:
        en = part("en", test, tmp, catalog)
        v18 = part("ko", test[ko], tmp, catalog)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sid = test["sample_id"]
    ours = en["target_vessel_id"].reindex(sid).to_numpy()
    gap = pd.to_numeric(en["ko_gap"], errors="coerce").reindex(sid).to_numpy()
    theirs = v18["target_vessel_id"].reindex(sid).to_numpy() if len(v18) else ours
    use_v18 = ko & ~(gap >= GAP)
    pred = [t if u else o for t, o, u in zip(theirs, ours, use_v18)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"sample_id": sid, "target_vessel_id": pred}}).to_csv(out, index=False)
    print(f"Saved {{len(test)}} predictions to {{out}} (Korean {{int(ko.sum())}}, from v18 {{int(use_v18.sum())}})")


if __name__ == "__main__":
    run()
'''


def copy_into(w, z, prefix, edit_script=None):
    req = []
    for info in z.infolist():
        if info.is_dir():
            continue
        if info.filename == "requirements.txt":
            req += [l for l in z.read(info).decode().splitlines() if l.strip() and not l.startswith("#")]
            continue
        if info.filename == "script.py" and edit_script:
            w.writestr(f"{prefix}/script.py", edit_script(z.read(info)))
            continue
        with z.open(info) as fsrc, w.open(zipfile.ZipInfo(f"{prefix}/{info.filename}", info.date_time), "w",
                                          force_zip64=info.file_size > 50_000_000) as fdst:
            while chunk := fsrc.read(1 << 24):
                fdst.write(chunk)
    return req


KC_EDITS = [  # 112-style scripts: report 110's gap (the gate), not the kc rankers' gap
    (b"            use_kc = True\n", b"            use_kc = True\n            row_gap = None\n"),
    (b"                use_kc = g110 < GATE\n", b"                use_kc = g110 < GATE\n                row_gap = g110\n"),
    (b"            gaps.append(float(score[allowed[0]] - score[allowed[1]]) / n_ko if len(allowed) > 1 else 99.0)\n",
     b"            gaps.append(row_gap if row_gap is not None else\n"
     b"                        (float(score[allowed[0]] - score[allowed[1]]) / n_ko if len(allowed) > 1 else 99.0))\n"),
]


def gap_script(data):
    for a, b in EN_EDITS:
        assert data.count(a) == 1, a
        data = data.replace(a, b)
    if b"use_kc = g110 < GATE" in data:
        for a, b in KC_EDITS:
            assert data.count(a) == 1, a
            data = data.replace(a, b)
    return data


def main(out, v18_zip, base_zip, gap="4"):
    out = Path(out)
    assert not out.exists(), out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        req = copy_into(w, zipfile.ZipFile(v18_zip), "model/ko")
        req += copy_into(w, zipfile.ZipFile(base_zip), "model/en", gap_script)
        w.writestr("script.py", ROUTER.format(gap=float(gap)))
        w.writestr("requirements.txt", "\n".join(dict.fromkeys(req)) + "\n" if req else "# preinstalled packages only\n")
    print(out, round(out.stat().st_size / 1e9, 3), "GB", "requirements:", req)


if __name__ == "__main__":
    main(*sys.argv[1:])
