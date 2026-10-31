# author: terra (Sol)
"""Freeze public-train pilot queries using existing stress functions, without ML imports."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "queries_v1"
SPLIT = Path(__file__).resolve().parent / "split_v1/rows.csv"
VARIANTS = ("orig", "edit1", "edit2", "jamo2", "trunc60%", "drop_last_word")
SEED_LABEL = "byt5-pilot-queries-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def helpers(path, names):
    # Extract only the named existing definitions; do not import modules whose
    # top level imports ML libraries and creates cache directories.
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            selected.append(node)
    ns = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    assert names <= ns.keys()
    return ns


SOURCES = {n: ROOT / f"shared/src/opus_stress{n}.py" for n in (2, 3, 4)}
H2 = helpers(SOURCES[2], {"drop_word", "trunc"})
H3 = helpers(SOURCES[3], {"gen_edit"})
H4 = helpers(SOURCES[4], {"jamo_typo", "_VOWEL_SWAP", "_CHO_SWAP"})


def generate(records):
    rows = []
    for r in records:
        for variant in VARIANTS:
            seed = int.from_bytes(hashlib.sha256(
                f"{SEED_LABEL}|{r['sample_id']}|{variant}".encode()
            ).digest()[:8], "big")
            s = r["vessel_name"]
            if variant.startswith("edit"):
                s = H3["gen_edit"](s, int(variant[-1]), random.Random(seed))
            elif variant == "jamo2":
                H4["R"] = random.Random(seed)
                s = H4["jamo_typo"](s, "", 2)
            elif variant == "trunc60%":
                s = H2["trunc"](s)
            elif variant == "drop_last_word":
                s = H2["drop_word"](s)
            q = dict(r, vessel_name=s, orig_sample_id=r["sample_id"],
                     query_sample_id=r["sample_id"] if variant == "orig" else f"{r['sample_id']}__{variant}",
                     variant=variant, generator_seed=str(seed),
                     original_vessel_name=r["vessel_name"], name_changed=str(s != r["vessel_name"]))
            assert s.strip()
            assert all(q[k] == v for k, v in r.items() if k != "vessel_name")
            rows.append(q)
    return sorted(rows, key=lambda q: (q["orig_sample_id"], VARIANTS.index(q["variant"])))


def encode(rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def main():
    source = ROOT / "data/open/train.csv"
    roles = {r["sample_id"]: r for r in read(SPLIT)}
    records = [dict(r, role=roles[r["sample_id"]]["role"]) for r in read(source)
               if roles[r["sample_id"]]["role"] in {"tune", "eval"}
               and roles[r["sample_id"]]["is_hangul"] == "True"]
    assert len(records) == 357
    rows = generate(records)
    shuffled = list(records)
    random.Random(20261027).shuffle(shuffled)
    assert encode(rows) == encode(generate(shuffled))
    assert len({r["query_sample_id"] for r in rows}) == len(rows)
    assert {r["orig_sample_id"] for r in rows if r["variant"] == "orig"} == {r["sample_id"] for r in records}
    OUT.mkdir(exist_ok=True)
    payload = encode(rows)
    dest = OUT / "queries.csv"
    if dest.exists():
        assert dest.read_bytes() == payload, "preserve existing manifest; use a new version"
    else:
        dest.write_bytes(payload)
    report = {
        "status": "QUERIES_ONLY_NO_MODEL_NO_GPU_NO_ACCURACY",
        "seed_policy": f"first 8 SHA256 bytes, big endian, of {SEED_LABEL}|sample_id|variant",
        "semantics": "Existing stress function bodies; per-query RNG reset. Not identical draws to historical stress tables.",
        "query_id": "orig keeps original sample_id; other variants append __variant; use query_sample_id as scoring key",
        "counts": {role: {variant: {
            "rows": len(d := [r for r in rows if r["role"] == role and r["variant"] == variant]),
            "vessels": len({r["target_vessel_id"] for r in d}),
            "changed_names": sum(r["name_changed"] == "True" for r in d)
        } for variant in VARIANTS} for role in ("tune", "eval")},
        "qa": {"order_invariant": True, "nonempty_names": True, "unique_query_ids": True,
               "metadata_unchanged": True, "queries": len(rows)},
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                   (source, SPLIT, *SOURCES.values(), Path(__file__), dest)},
        "limits": ["public train tune/eval only; no test access", "3-way holdout, not OOF",
                   "unchanged outputs retained and flagged; report separately when useful",
                   "no candidate scoring; no training or GPU execution"]
    }
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"qa": report["qa"], "counts": report["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
