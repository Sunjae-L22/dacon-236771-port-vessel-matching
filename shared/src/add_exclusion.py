# author: opus
"""Post-hoc 'train ships are never the answer' rule for an already-built submission, plus the
determinism patch. Same transformation that turned 08 into 18.

usage: python3 shared/src/add_exclusion.py <in.zip> <out.zip>
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import determinism_patch as D  # noqa: E402

ARGMAX = "        preds.append(ix.ids[idx[int(np.argmax(sum(m.predict(F) for m in models)))]])"
# template after the determinism fix (ties go to the lower catalog index)
ARGMAX_DET = ("        score = sum(m.predict(F) for m in models)\n"
              "        preds.append(ix.ids[idx[min(range(len(idx)), key=lambda j: (-score[j], idx[j]))]])")
EXCLUDE = '''        score = sum(m.predict(F) for m in models)
        # Evaluation ships do not appear in train.csv, so skip candidates that are train targets
        order = np.argsort(-score)
        best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])
        preds.append(ix.ids[best])'''


def main(src, out):
    with zipfile.ZipFile(src) as z:
        files = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    s = files["script.py"].decode("utf-8")
    assert ARGMAX in s or ARGMAX_DET in s, "unexpected script layout"
    s = s.replace(ARGMAX_DET, EXCLUDE).replace(ARGMAX, EXCLUDE)
    s = s.replace("    preds = []", '    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}\n'
                  "    preds = []", 1)
    files["script.py"] = s.encode("utf-8")
    tmp = Path(out).with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    D.patch_zip(str(tmp), out)
    tmp.unlink()


if __name__ == "__main__":
    main(*sys.argv[1:3])
