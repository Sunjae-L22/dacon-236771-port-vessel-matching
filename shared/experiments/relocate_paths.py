# author: opus
"""Rewrite the absolute project root hardcoded in old build scripts to this clone's location.

Many pinned recipes and run_*.sh files name /Users/sunjae/Desktop/항만 (NFC or NFD) and one names a pyenv python.
Run once in a fresh clone that lives anywhere else; tracked .py, .sh and .ipynb files are edited in place.

usage: python3 shared/experiments/relocate_paths.py [--dry-run]
"""
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = "/Users/sunjae/Desktop/항만"
PY = "/Users/sunjae/.pyenv/versions/3.11.9/bin/python3"


def main(dry=False):
    files = subprocess.run(["git", "ls-files", "*.py", "*.sh", "*.ipynb"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.splitlines()
    new = str(ROOT)
    changed = 0
    for name in files:
        p = ROOT / name
        if p.resolve() == Path(__file__).resolve():
            continue
        try:
            s = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        t = s.replace(PY, "python3")
        for form in ("NFC", "NFD"):
            t = t.replace(unicodedata.normalize(form, OLD), new)
        if t != s:
            changed += 1
            if not dry:
                p.write_text(t, encoding="utf-8")
    print(f"{'would change' if dry else 'changed'} {changed} files; root = {new}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv[1:])
