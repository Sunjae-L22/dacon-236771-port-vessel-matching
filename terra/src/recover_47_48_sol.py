# author: Sol
"""Finish failed postprocessing using new copies; never overwrite source ZIPs."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WATCH = ROOT / "terra/experiments/submission_watch_47_48"
WORK = WATCH / "recovery_20261027_1735"
RECEIPT = WATCH / "recovery_receipt.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(data):
    temp = RECEIPT.with_suffix(".new.json")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(RECEIPT)


def new_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert sha(src) == sha(dst), f"refusing to replace {dst}"
        return
    with Path(src).open("rb") as source, dst.open("xb") as target:
        shutil.copyfileobj(source, target)
    shutil.copystat(src, dst)


def main():
    WATCH.mkdir(parents=True, exist_ok=True)
    lock = WATCH / "recovery_running.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        receipt = {"status": "PREPARING", "original_zips_modified_by_this_recovery": False,
                   "training_performed": False, "individual_source_inference_repeated": False,
                   "sources": {}, "merged": {}}
        with zipfile.ZipFile(ROOT / "submissions/33_submit.zip") as baseline:
            script = baseline.read("script.py")
        for n in (43, 44, 45, 46):
            original = ROOT / f"submissions/{n}_submit.zip"
            snapshot = WORK / "raw" / original.name
            prepared = WORK / "prepared" / original.name
            before = sha(original)
            new_copy(original, snapshot)
            assert before == sha(original) == sha(snapshot), "source changed during snapshot"
            prepared.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(snapshot) as src:
                already_done = src.read("script.py") == script
            if not prepared.exists():
                if already_done:
                    new_copy(snapshot, prepared)
                else:
                    subprocess.run([sys.executable, str(ROOT / "shared/src/add_exclusion.py"),
                                    str(snapshot), str(prepared)], cwd=ROOT, check=True)
            with zipfile.ZipFile(snapshot) as src, zipfile.ZipFile(prepared) as dst:
                assert src.testzip() is None and dst.testzip() is None
                assert set(src.namelist()) == set(dst.namelist())
                assert dst.read("script.py") == script
                for name in src.namelist():
                    if name != "script.py":
                        assert src.read(name) == dst.read(name), f"{n}: unexpected member change {name}"
            receipt["sources"][str(n)] = {"original_path": str(original), "snapshot_path": str(snapshot),
                "snapshot_sha256": sha(snapshot), "prepared_path": str(prepared), "prepared_sha256": sha(prepared),
                "postprocess_already_present_in_snapshot": already_done}
            print(f"prepared source {n}; source models unchanged", flush=True)
        receipt["status"] = "SOURCES_PREPARED"
        save(receipt)
        for n, inputs in ((47, (43, 44)), (48, (45, 46))):
            output = WORK / f"{n}_submit.zip"
            assert not output.exists(), f"private merged output already exists: {output}"
            subprocess.run([sys.executable, str(ROOT / "shared/src/merge_rankers.py"), str(output),
                            *[receipt["sources"][str(i)]["prepared_path"] for i in inputs]], cwd=ROOT, check=True)
            receipt["merged"][str(n)] = {"private_path": str(output), "zip_sha256": sha(output),
                "check_log": str(WORK / f"check_{n}.log")}
        save(receipt)

        def check(n):
            item = receipt["merged"][str(n)]
            with Path(item["check_log"]).open("x") as log:
                subprocess.run([sys.executable, str(ROOT / "shared/src/local_check.py"), item["private_path"]],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            sys.path.insert(0, str(Path(__file__).parent))
            from check_ready_47_48_sol import check_log
            assert check_log(Path(item["check_log"]).read_text()) is not None
            assert sha(item["private_path"]) == item["zip_sha256"]
            print(f"merged {n}: runtime and row-independence checks passed", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(check, (47, 48)))
        receipt["status"] = "RUNTIME_CHECKED"
        save(receipt)
        for n in (47, 48):
            item = receipt["merged"][str(n)]
            published = ROOT / f"submissions/{n}_submit.zip"
            new_copy(item["private_path"], published)
            item["published_path"] = str(published)
        receipt["status"] = "PUBLISHED"
        save(receipt)
        # Publish success logs last; the existing queue is already waiting for check_48.
        for n in (47, 48):
            new_copy(receipt["merged"][str(n)]["check_log"], ROOT / f"shared/experiments/check_{n}.log")
        print("recovered 47/48 published; original source ZIPs preserved", flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
