# author: opus
"""Make predictions independent of Python's per-process string/NaN hash order.

Applied to source files and to the copies inside submission zips:
  * KoMatcher.candidates: iterate n-gram sets in sorted order (vote ties broke by hash order)
  * opus_lib: iterate (gt, igt) in a fixed order ({gt, igt} with NaN reorders per process)
  * script.py: ties between equal scores go to the lower catalog index; re-exec with PYTHONHASHSEED=0
"""
import re
import sys
import zipfile


def patch_argon(s: str) -> str:
    s = s.replace("        for bg in _ngrams(q_trans, 3 if len(q_trans) >= 4 else 2):",
                  "        for bg in sorted(_ngrams(q_trans, 3 if len(q_trans) >= 4 else 2)):")
    s = s.replace("        for bg in _ngrams(q_cons_t, 2) | _ngrams(q_cons_r, 2):",
                  "        for bg in sorted(_ngrams(q_cons_t, 2) | _ngrams(q_cons_r, 2)):")
    return s


def patch_lib(s: str) -> str:
    s = s.replace("for v in {gt, igt}:", "for v in dict.fromkeys((gt, igt)):")
    return s


def patch_script(s: str) -> str:
    if "PYTHONHASHSEED" not in s:
        s = s.replace("import sys\n", "import os\nimport sys\n\n"
                      "if os.environ.get(\"PYTHONHASHSEED\") is None:  # same predictions in every process\n"
                      "    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED=\"0\"))\n", 1)
    s = s.replace("        preds.append(ix.ids[idx[int(np.argmax(sum(m.predict(F) for m in models)))]])",
                  "        score = sum(m.predict(F) for m in models)\n"
                  "        preds.append(ix.ids[idx[min(range(len(idx)), key=lambda j: (-score[j], idx[j]))]])")
    s = s.replace("        order = np.argsort(-score)\n",
                  "        order = sorted(range(len(idx)), key=lambda j: (-score[j], idx[j]))\n")
    return s


def patch_zip(src, out):
    with zipfile.ZipFile(src) as z:
        files = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    for n in list(files):
        t = files[n].decode("utf-8") if n.endswith(".py") else None
        if t is None:
            continue
        if n.endswith("ko_translit_argon.py"):
            t = patch_argon(t)
        elif n.endswith("opus_lib.py"):
            t = patch_lib(t)
        elif n.endswith("script.py"):
            t = patch_script(t)
        files[n] = t.encode("utf-8")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    print("patched", src, "->", out)


if __name__ == "__main__":
    patch_zip(*sys.argv[1:3])
