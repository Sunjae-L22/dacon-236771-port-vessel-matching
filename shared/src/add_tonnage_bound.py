# author: opus
"""Add the tonnage bound to an already-built submission (no retraining).

In train.csv the true vessel's catalog gross tonnage is always within +-10% of the record's GT or IGT
(0 of 11,429 rows outside), so a candidate outside [0.9, 1.1] on both cannot be the answer. The rule only
skips such candidates when the record has a tonnage and the candidate has a catalog tonnage; if every
candidate is out of bounds the old choice stands.

usage: python3 shared/src/add_tonnage_bound.py <in.zip> <out.zip>
"""
import sys
import zipfile

OLD = "        best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])"
NEW = '''        qt = [float(v) for v in (row.get("gross_tonnage"), row.get("international_gross_tonnage")) if v == v and v and float(v) > 0]

        def in_bounds(i):  # true vessels are always within +-10% of the record's GT or IGT
            c = ix.gt[i]
            return not qt or not np.isfinite(c) or any(0.9 <= q / c <= 1.1 for q in qt)

        allowed = [j for j in order if idx[j] not in train_targets]
        best = next((idx[j] for j in allowed if in_bounds(idx[j])), idx[allowed[0]] if allowed else idx[order[0]])'''


def main(src, out):
    with zipfile.ZipFile(src) as z:
        files = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    s = files["script.py"].decode("utf-8")
    assert s.count(OLD) == 1, "unexpected script layout"
    files["script.py"] = s.replace(OLD, NEW).encode("utf-8")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
