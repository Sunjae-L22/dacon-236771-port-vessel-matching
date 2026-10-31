# author: Sol
"""Recover completed 49-52 models on copies, then release new 53/54 archives.

Never retrains or changes source ZIPs/pinned code. Existing successful checks are
reused only for identical archives; legacy checks finish before new inference.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WATCH = ROOT / "terra/experiments/submission_watch_53_54"
WORK = WATCH / "recovery_20261027_2218"
RECEIPT = WATCH / "recovery_receipt.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ready_47_48_sol import check_log


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(r):
    r["updated_at_epoch"] = time.time()
    p = RECEIPT.with_suffix(".new.json")
    p.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n")
    p.replace(RECEIPT)


def copy_new(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert sha(src) == sha(dst), f"refuse replacement: {dst}"
        return
    with Path(src).open("rb") as a, dst.open("xb") as b:
        shutil.copyfileobj(a, b)


def command(*args, stdout=None):
    subprocess.run([sys.executable, "-u", *map(str, args)], cwd="/private/tmp",
                   stdin=subprocess.DEVNULL, stdout=stdout,
                   stderr=subprocess.STDOUT, check=True)


def main():
    lock = WATCH / "recovery_running.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    r = {"status": "PREPARING", "pid": os.getpid(), "training_performed": False,
         "original_zips_modified_by_this_recovery": False, "sources": {}, "merged": {}}
    try:
        snapshots = json.loads((WORK / "snapshots.json").read_text())["sources"]
        sys.path.insert(0, str(ROOT / "shared/src"))
        import determinism_patch as D
        baseline = zipfile.ZipFile(ROOT / "submissions/33_submit.zip")
        script = baseline.read("script.py")
        baseline.close()
        for n in (49, 50, 51, 52):
            entry = snapshots[str(n)]
            raw = Path(entry["snapshot_path"])
            assert sha(raw) == entry["snapshot_sha256"]
            prepared = WORK / "prepared" / raw.name
            prepared.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(raw) as z:
                done = z.read("script.py") == script
            if not prepared.exists():
                if done:
                    copy_new(raw, prepared)
                else:
                    command(ROOT / "shared/src/add_exclusion.py", raw, prepared)
            folder = ROOT / "shared/experiments/pinned" / ("build49" if n < 51 else "build43")
            expected_lib = D.patch_lib((folder / "opus_lib.py").read_text().replace(
                'os.environ.get("KO_DICT", "0")', 'os.environ.get("KO_DICT", "1")')).encode()
            expected_matcher = D.patch_argon((folder / "ko_translit_argon.py").read_text()).encode()
            with zipfile.ZipFile(raw) as a, zipfile.ZipFile(prepared) as b:
                assert a.testzip() is None and b.testzip() is None
                assert set(a.namelist()) == set(b.namelist())
                assert b.read("script.py") == script
                assert b.read("model/opus_lib.py") == expected_lib
                assert b.read("model/ko_translit_argon.py") == expected_matcher
                for name in a.namelist():
                    if name not in {"script.py", "model/opus_lib.py", "model/ko_translit_argon.py"}:
                        assert a.read(name) == b.read(name), f"source {n}: changed {name}"
                ranks = sorted(k for k in b.namelist() if re.fullmatch(r"model/ranker_\d+\.txt", k))
                assert len(ranks) == 3
                hashes = {k: hashlib.sha256(b.read(k)).hexdigest() for k in b.namelist()}
                assert {k: hashes[k] for k in ranks} == entry["models"]
            r["sources"][str(n)] = dict(entry, prepared_path=str(prepared), prepared_sha256=sha(prepared),
                members_sha256=hashes, postprocess_verified=True)
            save(r)
            print(f"Prepared {n}: original three models and stats preserved", flush=True)

        # Avoid competing with the legacy 49/51/52 inference checks already seen.
        r["status"] = "WAITING_EXISTING_CHECKS"
        save(r)
        legacy = {85546, 85785}
        # Other wrapper children may have begun check_51 after the initial snapshot.
        for line in subprocess.check_output(["ps", "-axo", "pid=,args="], text=True).splitlines():
            m = re.match(r"\s*(\d+)\s+(.+)", line)
            if m and "local_check.py" in m[2] and re.search(r"(?:49|50|51|52)_submit\.zip", m[2]):
                legacy.add(int(m[1]))
        deadline = time.monotonic() + 1200
        while legacy:
            live = set()
            for pid in legacy:
                out = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
                if out and not out.startswith("Z"):
                    live.add(pid)
            legacy = live
            if not legacy:
                break
            assert time.monotonic() < deadline, f"existing checks did not finish: {sorted(legacy)}"
            time.sleep(10)

        for n in (49, 50, 51, 52):
            entry = r["sources"][str(n)]
            prepared, original = Path(entry["prepared_path"]), Path(entry["original_path"])
            prior = ROOT / f"shared/experiments/check_{n}.log"
            with zipfile.ZipFile(original) as z:
                assert z.testzip() is None
                original_members = {k: hashlib.sha256(z.read(k)).hexdigest() for k in z.namelist()}
            reusable = (prior.exists() and prior.stat().st_mtime_ns >= original.stat().st_mtime_ns
                        and check_log(prior.read_text()) is not None
                        and original_members == entry["members_sha256"])
            log = WORK / f"check_{n}.log"
            if reusable:
                copy_new(prior, log)
                entry["runtime_reused_from"] = str(prior)
                entry["runtime_input_path"] = str(original)
                entry["runtime_input_sha256"] = sha(original)
                entry["runtime_input_members_sha256"] = original_members
            elif not log.exists():
                r["status"] = f"CHECKING_SOURCE_{n}"
                save(r)
                before = sha(prepared)
                with log.open("x") as f:
                    command(ROOT / "shared/src/local_check.py", prepared, stdout=f)
                assert sha(prepared) == before
                entry["runtime_input_path"] = str(prepared)
                entry["runtime_input_sha256"] = before
                entry["runtime_input_members_sha256"] = entry["members_sha256"]
            parsed = check_log(log.read_text())
            assert parsed is not None, f"missing complete nonempty runtime log {n}"
            entry.update(check_log=str(log), check_log_sha256=sha(log), runtime_check=parsed,
                         validated_zip_sha256=sha(prepared))
            save(r)
            print(f"Source {n}: runtime and row independence passed; reused={reusable}", flush=True)

        for number, inputs in ((53, (49, 50)), (54, (51, 52))):
            output = WORK / f"{number}_submit.zip"
            assert not output.exists(), f"private merge already exists: {output}"
            command(ROOT / "shared/src/merge_rankers.py", output,
                    *[r["sources"][str(n)]["prepared_path"] for n in inputs])
            before = sha(output)
            with zipfile.ZipFile(output) as z:
                assert z.testzip() is None
                ranks = sorted(k for k in z.namelist() if re.fullmatch(r"model/ranker_\d+\.txt", k))
                assert len(ranks) == 6
                assert len({hashlib.sha256(z.read(k)).hexdigest() for k in ranks}) == 6
                slot = 0
                for source in inputs:
                    with zipfile.ZipFile(r["sources"][str(source)]["prepared_path"]) as a:
                        for k in sorted(k for k in a.namelist() if re.fullmatch(r"model/ranker_\d+\.txt", k)):
                            assert a.read(k) == z.read(f"model/ranker_{slot}.txt")
                            slot += 1
            log = WORK / f"check_{number}.log"
            r["status"] = f"CHECKING_MERGED_{number}"
            r["merged"][str(number)] = {"private_path": str(output), "zip_sha256": before, "check_log": str(log)}
            save(r)
            with log.open("x") as f:
                command(ROOT / "shared/src/local_check.py", output, stdout=f)
            assert check_log(log.read_text()) is not None, "runtime/independence check incomplete"
            assert sha(output) == before
            r["merged"][str(number)].update(check_log_sha256=sha(log), runtime_check=check_log(log.read_text()))
            save(r)
            print(f"Merged {number}: runtime and row independence passed", flush=True)

        # Never replace any original source or already-published submission.
        for number in (53, 54):
            entry = r["merged"][str(number)]
            final = ROOT / f"submissions/{number}_submit.zip"
            copy_new(entry["private_path"], final)
            entry["published_path"] = str(final)
            copy_new(entry["check_log"], ROOT / f"shared/experiments/check_{number}.log")
        r["status"] = "PUBLISHED"
        save(r)
        print("New 53/54 released for independent lead verification; source ZIPs untouched", flush=True)
    except BaseException:
        r["status"] = "FAILED"
        r["traceback"] = traceback.format_exc()
        save(r)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
