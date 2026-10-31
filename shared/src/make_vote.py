# author: opus
"""Pack several finished submissions into one that runs them side by side and votes per row.

usage: python3 shared/src/make_vote.py <out.zip> <tiebreak.zip> <other.zip> [...]
Each member keeps its own code under model/mNN/ and runs in its own process (their
opus_lib versions differ). A row takes the ID most members chose; when every member
disagrees, the first zip on the command line decides.
"""
import sys
import zipfile
from pathlib import Path

SCRIPT = '''"""DACON entry point: run member submissions in parallel, majority vote per row."""
from collections import Counter
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import pandas as pd

HERE = Path(__file__).resolve().parent
MEMBERS = {members!r}  # first member breaks ties


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test_path, catalog_path = str(Path(test_path).resolve()), str(Path(catalog_path).resolve())
    tmp = Path(tempfile.mkdtemp())
    procs = {{}}
    for m in MEMBERS:
        code = ("import runpy; g = runpy.run_path({{s!r}}, run_name='member'); "
                "g['run']({{t!r}}, {{c!r}}, {{o!r}})").format(
            s=str(HERE / "model" / m / "script.py"), t=test_path, c=catalog_path, o=str(tmp / f"{{m}}.csv"))
        procs[m] = subprocess.Popen([sys.executable, "-c", code], cwd=str(HERE / "model" / m),
                                    env=dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1",
                                             OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                             NUMEXPR_NUM_THREADS="1"))
    for m, p in procs.items():
        if p.wait() != 0:
            raise RuntimeError(f"member {{m}} failed")
    preds = {{m: pd.read_csv(tmp / f"{{m}}.csv", dtype=str).set_index("sample_id")["target_vessel_id"] for m in MEMBERS}}
    test = pd.read_csv(test_path, encoding="utf-8-sig", dtype={{"sample_id": str}})
    final = []
    for sid in test["sample_id"]:
        votes = [preds[m].loc[sid] for m in MEMBERS]
        top, n = Counter(votes).most_common(1)[0]
        final.append(top if n > 1 else votes[0])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"sample_id": test["sample_id"], "target_vessel_id": final}}).to_csv(out, index=False)
    print(f"Saved {{len(final)}} voted predictions to {{out}}")


if __name__ == "__main__":
    run()
'''


def main(out, *zips):
    members = [f"m{Path(z).name[:2]}" for z in zips]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
        for z, m in zip(zips, members):
            with zipfile.ZipFile(z) as zi:
                for n in zi.namelist():
                    if n.endswith("/") or n == "requirements.txt":
                        continue
                    contents = zi.read(n)
                    if n == "script.py":
                        # These members predict tiny per-row matrices. Extra OpenMP
                        # workers compete across members on the 6-vCPU server.
                        contents = contents.decode("utf-8").replace(
                            "m.predict(F)", "m.predict(F, num_threads=1)").encode("utf-8")
                    zo.writestr(f"model/{m}/{n}", contents)
        zo.writestr("script.py", SCRIPT.format(members=members))
        zo.writestr("requirements.txt", "# uses preinstalled lightgbm, rapidfuzz, numpy, pandas\n")
    print("wrote", out, "members", members)


if __name__ == "__main__":
    main(*sys.argv[1:])
