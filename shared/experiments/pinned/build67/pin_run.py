# author: opus
"""Run an experiment script from a frozen copy of shared/src, so later edits can't leak into it.

usage: python3 shared/src/pin_run.py <run_name> <script.py> [args...]
The copy lives in shared/experiments/pinned/<run_name>/; caches stay in shared/experiments/cache.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
EXP = SRC.parent / "experiments"


def main(name, script, *args):
    dst = EXP / "pinned" / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for p in SRC.glob("*.py"):
        shutil.copy(p, dst / p.name)
    env = dict(os.environ, CACHE_DIR=str(EXP / "cache"))
    sys.exit(subprocess.call([sys.executable, str(dst / script), *args], env=env, cwd=dst))


if __name__ == "__main__":
    main(*sys.argv[1:])
