# author: Sol
"""Read-only release check for 56+57: baseline 33 plus exactly two features."""
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
PIN = E / "pinned/build56"
WORK = ROOT / "terra/experiments/gt_window56"
WATCH = ROOT / "terra/experiments/submission_watch_58"


def inspect():
    state = json.loads((WORK / "queue_state.json").read_text())
    require(state["status"] != "FAILED", "56/57 driver failed; inspect queue_state traceback")
    receipt_path = WORK / "build_receipt.json"
    if not receipt_path.exists() or not (ROOT / "submissions/58_submit.zip").exists():
        return {"number": 58, "status": "WAITING", "driver_state": state["status"]}
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "READY_FOR_LEAD_REVIEW", "driver not ready for review")
    manifest_path = PIN / "BUILD_MANIFEST_SOL.json"
    manifest = json.loads(manifest_path.read_text())
    require(digest(manifest_path.read_bytes()) == receipt["manifest_sha256"] == state["manifest_sha256"], "pinned manifest changed")
    for name, sha in manifest["sha256"].items():
        require(digest((PIN / name).read_bytes()) == sha, "pinned code/CSV changed: " + name)
    for name, sha in manifest["data_sha256"].items():
        require(digest((ROOT / "data/open" / name).read_bytes()) == sha, "input data changed")
    require(digest((ROOT / "submissions/33_submit.zip").read_bytes()) == manifest["baseline33_zip_sha256"], "baseline33 changed")
    require(manifest["source_archives"] == {"56": 0, "57": 5000}, "augmentation offset pair differs")
    require({w["number"]: w["seed_offset"] for w in state["workers"]} == {56: 0, 57: 5000}, "actual launch offsets differ")
    require(json.loads((WORK / "feature_contract_check.json").read_text())["status"] == "PASS_IMPLEMENTATION_CONTRACT_ONLY", "feature contract missing")
    sys.path.insert(0, str(ROOT / "shared/src"))
    import determinism_patch as D
    expected_lib = D.patch_lib((PIN / "opus_lib.py").read_text().replace(
        'os.environ.get("KO_DICT", "0")', 'os.environ.get("KO_DICT", "1")')).encode()
    expected_matcher = D.patch_argon((PIN / "ko_translit_argon.py").read_text()).encode()
    require('os.environ.get("KO_DICT", "1")' in expected_lib.decode(), "KO_DICT1 not baked")
    results = {}
    with zipfile.ZipFile(ROOT / "submissions/33_submit.zip") as baseline:
        old_names = re.search(r"^feature_names=(.*)$", baseline.read("model/ranker_0.txt").decode(), re.M).group(1).split()
        require(manifest["expected_features"] == old_names + ["ko_gt_win_diff_max", "ko_gt_win_margin"], "not exactly two appended features")
        for number, count in ((56, 3), (57, 3), (58, 6)):
            path = ROOT / f"submissions/{number}_submit.zip"
            before = digest(path.read_bytes())
            item = receipt["artifacts"][str(number)]
            require(before == item["zip_sha256"] == item["runtime"]["zip_sha256"], "archive differs from checked receipt")
            log = E / f"check_{number}.log"
            require(digest(log.read_bytes()) == item["runtime"]["local_check_sha256"], "runtime log changed")
            require(log.stat().st_mtime_ns >= path.stat().st_mtime_ns, "stale runtime check")
            runtime = check_log(log.read_text())
            require(runtime is not None, "runtime/independence check incomplete")
            with zipfile.ZipFile(path) as z:
                files, ranks = members(z)
                require(ranks == [f"model/ranker_{i}.txt" for i in range(count)], "wrong model slots")
                require(set(files) == set(ranks) | {"script.py", "requirements.txt", "model/opus_lib.py", "model/ko_translit_argon.py", "model/stats.pkl"}, "unexpected archive structure")
                require(z.read("script.py") == baseline.read("script.py"), "baseline post-ranking exclusion changed")
                require(z.read("requirements.txt") == baseline.read("requirements.txt"), "requirements changed")
                require(z.read("model/opus_lib.py") == expected_lib, "wrong two-feature library")
                require(z.read("model/ko_translit_argon.py") == expected_matcher == baseline.read("model/ko_translit_argon.py"), "matcher changed")
                require(same_value(pickle.loads(z.read("model/stats.pkl")), pickle.loads(baseline.read("model/stats.pkl"))), "v1 dictionary or other stats differ from33")
                hashes = {k: digest(z.read(k)) for k in files}
                rank_hashes = [hashes[k] for k in ranks]
                require(len(set(rank_hashes)) == count, "duplicate trained models")
                require(rank_hashes == item["ranker_sha256"], "driver model hashes differ")
                for k in ranks:
                    model = z.read(k).decode()
                    require(len(re.findall(r"^Tree=", model, re.M)) == 600, "tree count differs")
                    require(re.search(r"^feature_names=(.*)$", model, re.M).group(1).split() == manifest["expected_features"], "wrong62 feature order")
                if number in (56, 57):
                    with zipfile.ZipFile(E / f"staging56_sol/{number}_submit.zip") as raw:
                        _, raw_ranks = members(raw)
                        require(raw_ranks == ranks, "raw source slots differ")
                        for k in files:
                            if k not in {"script.py", "model/opus_lib.py", "model/ko_translit_argon.py"}:
                                require(z.read(k) == raw.read(k), "postprocessing changed raw model/stats")
                    text = (E / f"make{number}.log").read_text()
                    require("LLM Korean rows: ranker 7967, dictionary 7967" in text, "not v1 dictionary/train")
                    require(all(f"trained seed {s}" in text for s in range(3)), "training incomplete")
            require(digest(path.read_bytes()) == before, "ZIP changed during verification")
            results[str(number)] = {"zip_sha256": before, "ranker_sha256": rank_hashes,
                                    "members_sha256": hashes, "runtime_check": runtime}
    require(results["58"]["ranker_sha256"] == results["56"]["ranker_sha256"] + results["57"]["ranker_sha256"], "merged 3+3 model order differs")
    return {"number": 58, "status": "READY", "path": str(ROOT / "submissions/58_submit.zip"),
            "artifacts": results, "ko_dict": 1, "features": 62,
            "manifest_sha256": digest(manifest_path.read_bytes()), "build_receipt_sha256": digest(receipt_path.read_bytes()),
            "validator_sha256": digest(Path(__file__).read_bytes()),
            "limits": ["Implementation checks are not accuracy/CV results.", "Mac runtime does not establish L4 throughput."]}


if __name__ == "__main__":
    try:
        result = inspect()
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        result = {"number": 58, "status": "FAILED", "reason": str(error)}
    if result["status"] == "READY":
        WATCH.mkdir(parents=True, exist_ok=True)
        (WATCH / f"58_{result['artifacts']['58']['zip_sha256']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit({"READY": 0, "WAITING": 2, "FAILED": 1}[result["status"]])
