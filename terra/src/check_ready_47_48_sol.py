# author: Sol
"""Read-only release checks for 47/48; writes only optional JSON receipts.

Reuse the pipeline's existing runtime/row-independence checks. This script does
not train, run inference, unpack to data/, modify a ZIP, or select a submission.
Exit 0=both ready, 2=waiting, 1=failed validation.
"""
import argparse
import hashlib
import json
import math
import pickle
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "shared/experiments"
RECIPES = {
    47: {"inputs": (43, 44), "train_rows": 21239, "dict_rows": 21239, "dict_csv": "llm_korean_scored_v12.csv"},
    48: {"inputs": (45, 46), "train_rows": 21239, "dict_rows": 7967, "dict_csv": "llm_korean_scored.csv"},
}
RANKER = re.compile(r"model/ranker_(\d+)\.txt$")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same_value(a, b):
    """Compare known stats structures independently of dict insertion order."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def check_log(text):
    errors = re.search(r"Traceback|MemoryError|AssertionError|Killed|Exception", text)
    if errors:
        raise ValueError("local check reported " + errors.group())
    mismatches = re.findall(r"row independence: subset (\d+) rows, mismatches (\d+)", text)
    if mismatches:
        require(all(int(m) == 0 for _, m in mismatches), "row-independence mismatches are nonzero")
    runtime = re.findall(r"rows (\d+)\s+time ([\d.]+)s", text)
    sizes = re.findall(r"zip size ([\d.]+) MB", text)
    if not (mismatches and runtime and sizes):
        return None
    rows, secs = int(runtime[-1][0]), float(runtime[-1][1])
    require(rows == 3000, "local check must cover 3000 rows")
    require(int(mismatches[-1][0]) == 600, "subset check must cover 600 rows")
    require(0 < secs < 1800, "local runtime is outside the 30-minute budget")
    return {"rows": rows, "local_seconds": secs, "subset_rows": 600, "mismatches": 0,
            "note": "Observed on local Mac; does not establish L4 server throughput."}


def members(z):
    files = [n for n in z.namelist() if not n.endswith("/")]
    require(len(files) == len(set(files)), "duplicate ZIP member names")
    require(all(not n.startswith("/") and ".." not in Path(n).parts for n in files), "unsafe ZIP path")
    required = {"script.py", "requirements.txt", "model/opus_lib.py", "model/ko_translit_argon.py", "model/stats.pkl"}
    require(required.issubset(files), "missing required submission files")
    require(z.testzip() is None, "ZIP CRC mismatch")
    rankers = sorted((n for n in files if RANKER.fullmatch(n)), key=lambda n: int(RANKER.fullmatch(n).group(1)))
    return files, rankers


def expected_pairs(csv_name):
    # Same filter/order as frozen build43 llm_korean_rows, without importing a build.
    import pandas as pd
    tr = pd.read_csv(ROOT / "data/open/train.csv", encoding="utf-8-sig")
    real = tr[tr.vessel_name.map(lambda s: isinstance(s, str) and any("가" <= c <= "힣" for c in s))]
    base = tr.drop_duplicates("target_vessel_id").set_index("target_vessel_id")
    gen = pd.read_csv(E / "pinned/build43" / csv_name, encoding="utf-8-sig")
    gen = gen[(gen.score >= .70) & gen.vessel_id.isin(base.index)]
    llm = base.loc[gen.vessel_id].reset_index()
    llm["vessel_name"] = gen.ko.to_numpy()
    llm = llm[llm.gross_tonnage.notna()]
    columns = ["vessel_name", "target_vessel_id"]
    return real[columns].to_dict("records") + llm[columns].to_dict("records"), len(llm)


def inspect(number):
    spec = RECIPES[number]
    path = ROOT / f"submissions/{number:02d}_submit.zip"
    log = E / f"check_{number}.log"
    result = {"number": number, "status": "WAITING", "path": str(path), "inputs": list(spec["inputs"])}
    recovery_file = ROOT / "terra/experiments/submission_watch_47_48/recovery_receipt.json"
    recovery = json.loads(recovery_file.read_text()) if recovery_file.exists() else None
    if recovery and recovery.get("status") not in ("PUBLISHED", "COORDINATED_RUNTIME_CHECKS"):
        lock = recovery_file.parent / "recovery_running.lock"
        if lock.exists():
            result["reason"] = "model training finished; postprocessing recovery and runtime checks are running"
            return result
        raise ValueError("postprocessing recovery is incomplete; inspect recovery_run.log")
    for source in spec["inputs"]:
        make = E / f"make{source}.log"
        if make.exists() and re.search(r"Traceback|Killed|MemoryError", make.read_text()):
            require(recovery is not None and "shared/src/add_exclusion.py" in make.read_text()
                    and "PermissionError" in make.read_text(), f"source build {source} failed; inspect {make}")
    if not path.exists() or not log.exists():
        result["reason"] = "ZIP or completed local-check log is not available yet"
        return result
    result["runtime_check"] = check_log(log.read_text())
    if result["runtime_check"] is None or log.stat().st_mtime_ns < path.stat().st_mtime_ns:
        result["reason"] = "local check is incomplete or older than the ZIP"
        return result
    start = (path.stat().st_size, path.stat().st_mtime_ns)
    zip_hash_start = digest(path.read_bytes())
    log_hash_start = digest(log.read_bytes())
    if recovery:
        item = recovery["merged"][str(number)]
        require(item["zip_sha256"] == zip_hash_start, "recovered merged ZIP hash differs")
        require(digest(Path(item["check_log"]).read_bytes()) == log_hash_start, "recovered runtime log differs")
    with zipfile.ZipFile(path) as z, zipfile.ZipFile(ROOT / "submissions/33_submit.zip") as baseline:
        files, rankers = members(z)
        require(rankers == [f"model/ranker_{i}.txt" for i in range(6)], "expected exactly ranker_0..5")
        require(set(files) == set(rankers) | {"script.py", "requirements.txt", "model/opus_lib.py",
                                            "model/ko_translit_argon.py", "model/stats.pkl"},
                "unexpected files in submission")
        hashes = {n: digest(z.read(n)) for n in files}
        for n in ["script.py", "requirements.txt", "model/opus_lib.py", "model/ko_translit_argon.py"]:
            require(z.read(n) == baseline.read(n), f"unexpected change from 33: {n}")
        lib = z.read("model/opus_lib.py").decode()
        require(re.search(r'os\.environ\.get\("KO_DICT",\s*"1"\)', lib), "shipped KO_DICT default is not 1")
        stats = pickle.loads(z.read("model/stats.pkl"))
        old_stats = pickle.loads(baseline.read("model/stats.pkl"))
        require(set(stats) == set(old_stats), "stats keys changed unexpectedly")
        for key in set(stats) - {"ko_pairs"}:
            require(same_value(stats[key], old_stats[key]), f"unexpected stats change: {key}")
        expected, llm_rows = expected_pairs(spec["dict_csv"])
        require(llm_rows == spec["dict_rows"], "frozen dictionary CSV row count differs from recipe")
        require(stats["ko_pairs"] == expected, "shipped dictionary differs from intended frozen CSV/train pairs")
        baseline_model = baseline.read("model/ranker_0.txt").decode()
        expected_names = re.search(r"^feature_names=(.*)$", baseline_model, re.M).group(1)
        for n in rankers:
            model = z.read(n).decode()
            require(len(re.findall(r"^Tree=", model, re.M)) == 600, f"{n}: expected 600 trees")
            require(re.search(r"^feature_names=(.*)$", model, re.M).group(1) == expected_names,
                    f"{n}: unexpected feature names/order")
        require(len({hashes[n] for n in rankers}) == 6, "duplicate models; investigate source seed diversity")
        offset = 0
        source_hashes = {}
        for source in spec["inputs"]:
            original_path = ROOT / f"submissions/{source:02d}_submit.zip"
            sp = original_path
            if recovery:
                recovered = recovery["sources"][str(source)]
                snapshot = Path(recovered["snapshot_path"])
                sp = Path(recovered["prepared_path"])
                require(digest(snapshot.read_bytes()) == recovered["snapshot_sha256"], "raw source snapshot changed")
                require(digest(sp.read_bytes()) == recovered["prepared_sha256"], "postprocessed source changed")
                with zipfile.ZipFile(snapshot) as raw, zipfile.ZipFile(sp) as prepared, zipfile.ZipFile(original_path) as current:
                    require(set(raw.namelist()) == set(prepared.namelist()) == set(current.namelist()), "source member set changed")
                    for name in raw.namelist():
                        if name != "script.py":
                            require(raw.read(name) == prepared.read(name) == current.read(name),
                                    f"source {source}: recovery changed model/stats/code {name}")
                    require(current.read("script.py") in (raw.read("script.py"), prepared.read("script.py")),
                            "source script changed beyond the documented postprocessing")
            if "train_rows" in spec:
                make_text = (E / f"make{source}.log").read_text()
                require(f"LLM Korean rows: ranker {spec['train_rows']}, dictionary {spec['dict_rows']}" in make_text,
                        f"source {source}: logged data configuration differs from recipe")
                require(all(f"trained seed {i}" in make_text for i in range(3)),
                        f"source {source}: incomplete three-model training log")
                wrote = re.findall(r"^wrote (.+_submit\.zip)$", make_text, re.M)
                require(wrote and original_path.samefile(wrote[-1]), f"source {source}: output path in training log differs")
                if not recovery:
                    source_check = E / f"check_{source}.log"
                    require(source_check.exists() and check_log(source_check.read_text()) is not None,
                            f"source {source}: local check incomplete")
                    require(source_check.stat().st_mtime_ns >= sp.stat().st_mtime_ns,
                            f"source {source}: local check is older than ZIP")
            with zipfile.ZipFile(sp) as src:
                _, srankers = members(src)
                require(len(srankers) == 3, f"source {source}: expected 3 models")
                for n in ["script.py", "requirements.txt", "model/opus_lib.py", "model/ko_translit_argon.py", "model/stats.pkl"]:
                    require(z.read(n) == src.read(n), f"source {source}: merge member differs: {n}")
                for n in srankers:
                    require(z.read(f"model/ranker_{offset}.txt") == src.read(n), f"source {source}: merged model differs")
                    offset += 1
            source_hashes[str(source)] = digest(sp.read_bytes())
        result.update(status="READY", members_sha256=hashes, rankers=6, ko_dict=1,
                      dictionary_pairs=len(expected), llm_dictionary_pairs=llm_rows,
                      source_zip_sha256=source_hashes,
                      dictionary_csv_sha256=digest((E / "pinned/build43" / spec["dict_csv"]).read_bytes()))
    result["baseline33_zip_sha256"] = digest((ROOT / "submissions/33_submit.zip").read_bytes())
    result["validator_sha256"] = digest(Path(__file__).read_bytes())
    result["zip_sha256"] = digest(path.read_bytes())
    require(result["zip_sha256"] == zip_hash_start, "ZIP content changed during validation")
    require(start == (path.stat().st_size, path.stat().st_mtime_ns), "ZIP changed during validation")
    result["local_check_sha256"] = digest(log.read_bytes())
    require(result["local_check_sha256"] == log_hash_start, "local check log changed during validation")
    result["provenance_limits"] = ["Training-row counts and augmentation offsets are checked against the existing build record/logs, not recovered from model text."]
    if recovery:
        result["recovery_receipt_sha256"] = digest(recovery_file.read_bytes())
        result["runtime_binding"] = recovery.get("runtime_binding", "Recovery wrapper verified the merged ZIP SHA256 before and after the full runtime check.")
        result["source_runtime_scope"] = "Source archives/models were checked statically; full inference and independence were checked on this six-model merged artifact."
    else:
        result["provenance_limits"].append("The upstream runtime logger does not print a ZIP hash; its unique pipeline target and fresh completed log are used.")
    result["seed_offsets"] = {"documented_source_recipe": [0, 5000],
                               "note": "Augmentation offsets come from the frozen build record, not ranker filenames."}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path)
    args = parser.parse_args()
    results = []
    for number in RECIPES:
        try:
            result = inspect(number)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
            result = {"number": number, "status": "FAILED", "reason": str(error)}
        results.append(result)
        if args.receipt_dir and result["status"] == "READY":
            args.receipt_dir.mkdir(parents=True, exist_ok=True)
            (args.receipt_dir / f"{number}_{result['zip_sha256']}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if any(r["status"] == "FAILED" for r in results) else 0 if all(r["status"] == "READY" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
