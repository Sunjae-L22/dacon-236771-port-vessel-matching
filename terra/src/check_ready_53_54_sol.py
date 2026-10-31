# author: Sol
"""Read-only checks for the two different 53/54 recipes (not the 47/48 checker)."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ready_47_48_sol import check_log, digest, members, require, same_value

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "shared/experiments"
WATCH = ROOT / "terra/experiments/submission_watch_53_54"
RECIPES = {
    53: {"inputs": (49, 50), "build": "build49", "dict_csv": "llm_korean_scored.csv", "dict_rows": 7967, "sample_ids": True},
    54: {"inputs": (51, 52), "build": "build43", "dict_csv": "llm_korean_scored_v12.csv", "dict_rows": 21239, "sample_ids": False},
}
RANKER = re.compile(r"model/ranker_(\d+)\.txt$")


def recovered_source(entry):
    """Independently bind a copied source to original learned models and a checked input."""
    path = Path(entry["prepared_path"])
    snapshot = Path(entry["snapshot_path"])
    runtime_input = Path(entry["runtime_input_path"])
    log = Path(entry["check_log"])
    require(digest(path.read_bytes()) == entry["prepared_sha256"] == entry["validated_zip_sha256"], "prepared source changed")
    require(digest(snapshot.read_bytes()) == entry["snapshot_sha256"], "original source snapshot changed")
    require(digest(runtime_input.read_bytes()) == entry["runtime_input_sha256"], "runtime input changed")
    require(digest(log.read_bytes()) == entry["check_log_sha256"], "recovery runtime log changed")
    require(check_log(log.read_text()) is not None, "recovery runtime check incomplete")
    # Reused logs may precede the private ZIP: the checked archive's member bytes
    # must be identical, and its successful log must follow that checked archive.
    prior = Path(entry.get("runtime_reused_from", str(log)))
    require(prior.stat().st_mtime_ns >= runtime_input.stat().st_mtime_ns, "recovered runtime check stale")
    require(prior.read_bytes() == log.read_bytes(), "reused runtime log differs")
    with zipfile.ZipFile(path) as prepared, zipfile.ZipFile(snapshot) as raw, zipfile.ZipFile(runtime_input) as checked, zipfile.ZipFile(entry["original_path"]) as original:
        prepared_files, ranks = members(prepared)
        raw_files, raw_ranks = members(raw)
        checked_files, _ = members(checked)
        original_files, original_ranks = members(original)
        require(set(prepared_files) == set(raw_files) == set(checked_files) == set(original_files), "recovery source members changed")
        require(ranks == raw_ranks and len(ranks) == 3, "recovery source model count differs")
        require(original_ranks == ranks, "original input model slots changed")
        for name in prepared_files:
            require(prepared.read(name) == checked.read(name), "reused runtime input differs from prepared source")
            if name not in {"script.py", "model/opus_lib.py", "model/ko_translit_argon.py"}:
                require(prepared.read(name) == raw.read(name), "recovery changed learned models or stats")
                require(prepared.read(name) == original.read(name), "actual original input models/stats differ from snapshot")
    return path, log


def contract():
    result = {}
    for number, spec in RECIPES.items():
        folder = E / "pinned" / spec["build"]
        result[str(number)] = {name: digest((folder / name).read_bytes()) for name in
                              ("opus_lib.py", "ko_translit_argon.py", "make_submit.py", "opus_cv.py", "cv.py", spec["dict_csv"])}
    return result


def expected_pairs(spec):
    import pandas as pd
    train = pd.read_csv(ROOT / "data/open/train.csv", encoding="utf-8-sig")
    real = train[train.vessel_name.map(lambda s: isinstance(s, str) and any("가" <= c <= "힣" for c in s))]
    base = train.drop_duplicates("target_vessel_id").set_index("target_vessel_id")
    gen = pd.read_csv(E / "pinned" / spec["build"] / spec["dict_csv"], encoding="utf-8-sig")
    gen = gen[(gen.score >= .70) & gen.vessel_id.isin(base.index)]
    llm = base.loc[gen.vessel_id].reset_index()
    llm["vessel_name"] = gen.ko.to_numpy()
    llm = llm[llm.gross_tonnage.notna()].reset_index(drop=True)
    llm["sample_id"] = [f"LLM_{i:05d}" for i in range(len(llm))]
    cols = (["sample_id"] if spec["sample_ids"] else []) + ["vessel_name", "target_vessel_id"]
    require(len(real) == 1191 and len(llm) == spec["dict_rows"], "unexpected dictionary source counts")
    return real[cols].to_dict("records") + llm[cols].to_dict("records")


def inspect(number):
    spec = RECIPES[number]
    path = ROOT / f"submissions/{number}_submit.zip"
    log = E / f"check_{number}.log"
    out = {"number": number, "status": "WAITING", "path": str(path), "inputs": list(spec["inputs"])}
    saved_contract = WATCH / "source_contract.json"
    if saved_contract.exists():
        require(contract()[str(number)] == json.loads(saved_contract.read_text())[str(number)], "active pinned source changed")
    recovery_path = WATCH / "recovery_receipt.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    if recovery and recovery["status"] != "PUBLISHED":
        require(recovery["status"] != "FAILED", "copy recovery failed; inspect recovery receipt")
        out["reason"] = "copy recovery in progress: " + recovery["status"]
        return out
    for source in spec["inputs"]:
        make = E / f"make{source}.log"
        if make.exists() and re.search(r"Traceback|Killed|MemoryError", make.read_text()):
            require(recovery is not None and str(source) in recovery["sources"],
                    f"source {source} failed; preserve models and inspect its log before recovery")
    if not path.exists() or not log.exists():
        out["reason"] = "ZIP or completed check log not available"
        return out
    check = check_log(log.read_text())
    if check is None or log.stat().st_mtime_ns < path.stat().st_mtime_ns:
        out["reason"] = "runtime/independence check incomplete or stale"
        return out
    initial_sha, log_sha = digest(path.read_bytes()), digest(log.read_bytes())
    before_stat = (path.stat().st_size, path.stat().st_mtime_ns)
    sys.path.insert(0, str(ROOT / "shared/src"))
    import determinism_patch as D
    folder = E / "pinned" / spec["build"]
    expected_lib = D.patch_lib((folder / "opus_lib.py").read_text().replace(
        'os.environ.get("KO_DICT", "0")', 'os.environ.get("KO_DICT", "1")')).encode()
    expected_matcher = D.patch_argon((folder / "ko_translit_argon.py").read_text()).encode()
    with zipfile.ZipFile(path) as z, zipfile.ZipFile(ROOT / "submissions/33_submit.zip") as baseline:
        files, rankers = members(z)
        require(rankers == [f"model/ranker_{i}.txt" for i in range(6)], "expected exactly six model slots")
        require(set(files) == set(rankers) | {"script.py", "requirements.txt", "model/opus_lib.py",
                                            "model/ko_translit_argon.py", "model/stats.pkl"}, "unexpected archive members")
        require(z.read("script.py") == baseline.read("script.py"), "post-ranking exclusion/script differs from 33")
        require(z.read("requirements.txt") == baseline.read("requirements.txt"), "requirements changed")
        require(z.read("model/opus_lib.py") == expected_lib, "shipped library differs from its frozen recipe")
        require(z.read("model/ko_translit_argon.py") == expected_matcher, "wrong matcher implementation")
        require('os.environ.get("KO_DICT", "1")' in expected_lib.decode(), "KO_DICT default is not 1")
        stats = pickle.loads(z.read("model/stats.pkl"))
        old = pickle.loads(baseline.read("model/stats.pkl"))
        require(set(stats) == set(old), "unexpected stats fields")
        for key in set(old) - {"ko_pairs"}:
            require(same_value(stats[key], old[key]), f"unexpected non-dictionary stats change: {key}")
        pairs = expected_pairs(spec)
        require(stats["ko_pairs"] == pairs, "shipped dictionary/source IDs differ from the intended CSV")
        if spec["sample_ids"]:
            require(sum(str(p["sample_id"]).startswith("LLM_") for p in pairs) == 7967, "LLM provenance tags lost")
            require(all(str(p["sample_id"]).startswith("TRAIN_") for p in pairs[:1191]), "real provenance tags lost")
        hashes = {n: digest(z.read(n)) for n in files}
        require(len({hashes[n] for n in rankers}) == 6, "duplicate models in paired ensemble")
        model0 = baseline.read("model/ranker_0.txt").decode()
        names = re.search(r"^feature_names=(.*)$", model0, re.M).group(1)
        for name in rankers:
            text = z.read(name).decode()
            require(len(re.findall(r"^Tree=", text, re.M)) == 600, "unexpected tree count")
            require(re.search(r"^feature_names=(.*)$", text, re.M).group(1) == names, "feature names/order changed")
        source_hashes, offset = {}, 0
        for source in spec["inputs"]:
            source_path = ROOT / f"submissions/{source}_submit.zip"
            make_text = (E / f"make{source}.log").read_text()
            require(f"LLM Korean rows: ranker 7967, dictionary {spec['dict_rows']}" in make_text, "training data recipe differs")
            require(all(f"trained seed {i}" in make_text for i in range(3)), "source training incomplete")
            if recovery:
                source_path, source_log = recovered_source(recovery["sources"][str(source)])
            else:
                source_log = E / f"check_{source}.log"
                require(source_log.exists() and check_log(source_log.read_text()) is not None, "source check incomplete")
                require(source_log.stat().st_mtime_ns >= source_path.stat().st_mtime_ns, "source check stale")
            with zipfile.ZipFile(source_path) as src:
                _, srankers = members(src)
                require(len(srankers) == 3, "source must contain three models")
                for member in files:
                    if member not in rankers:
                        require(src.read(member) == z.read(member), "merged code/stats differ from source")
                for member in srankers:
                    require(src.read(member) == z.read(f"model/ranker_{offset}.txt"), "merged model slot differs")
                    offset += 1
            source_hashes[str(source)] = digest(source_path.read_bytes())
    require(initial_sha == digest(path.read_bytes()), "ZIP changed during verification")
    require(before_stat == (path.stat().st_size, path.stat().st_mtime_ns), "ZIP metadata changed during verification")
    require(log_sha == digest(log.read_bytes()), "runtime log changed during verification")
    out.update(status="READY", zip_sha256=initial_sha, members_sha256=hashes, runtime_check=check,
               source_zip_sha256=source_hashes, ko_dict=1, dictionary_pairs=len(pairs),
               llm_dictionary_pairs=spec["dict_rows"], preserves_sample_ids=spec["sample_ids"],
               validator_sha256=digest(Path(__file__).read_bytes()), local_check_sha256=log_sha,
               documented_augmentation_offsets=[0, 5000],
               copy_recovery_receipt_sha256=digest(recovery_path.read_bytes()) if recovery else None,
               limits=["Augmentation offsets/data recipe are documented build provenance, not inferred from model-slot numbers.",
                       "Local Mac runtime does not establish L4 throughput; upstream local-check logs do not include a ZIP hash."])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-contract", action="store_true")
    args = parser.parse_args()
    WATCH.mkdir(parents=True, exist_ok=True)
    frozen = WATCH / "source_contract.json"
    if args.freeze_contract:
        now = contract()
        if frozen.exists():
            require(json.loads(frozen.read_text()) == now, "never replace a changed source contract")
        else:
            frozen.write_text(json.dumps(now, ensure_ascii=False, indent=2) + "\n")
    results = []
    for number in RECIPES:
        try:
            result = inspect(number)
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
            result = {"number": number, "status": "FAILED", "reason": str(error)}
        results.append(result)
        if result["status"] == "READY":
            (WATCH / f"{number}_{result['zip_sha256']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if any(x["status"] == "FAILED" for x in results) else 0 if all(x["status"] == "READY" for x in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
