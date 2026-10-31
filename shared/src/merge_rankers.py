# author: opus
"""Merge the rankers of submissions built with the same recipe (same code and stats) into one.

usage: python3 shared/src/merge_rankers.py <out.zip> <first.zip> <second.zip> [...]
The first zip supplies the code, stats and script; every zip must carry identical model code and
stats, which is checked. Rankers are renumbered ranker_0..N and their scores are summed at inference.
"""
import sys
import zipfile


def main(out, *zips):
    base = zipfile.ZipFile(zips[0])
    files = {n: base.read(n) for n in base.namelist() if not n.endswith("/") and "ranker" not in n}
    rankers = []
    for z in map(zipfile.ZipFile, zips):
        for n in ("model/opus_lib.py", "model/ko_translit_argon.py", "model/stats.pkl"):
            assert z.read(n) == files[n], f"{z.filename}: {n} differs from {zips[0]}"
        rankers += [z.read(n) for n in sorted(x for x in z.namelist() if "ranker" in x)]
    for k, b in enumerate(rankers):
        files[f"model/ranker_{k}.txt"] = b
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
        for n, b in files.items():
            zo.writestr(n, b)
    print("wrote", out, "with", len(rankers), "rankers")


if __name__ == "__main__":
    main(*sys.argv[1:])
