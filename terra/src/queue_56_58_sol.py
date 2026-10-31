# author: Sol
"""Wait for two free training slots, then build the 33+two-feature pair.

The pinned source is immutable. Workers use nohup, KO_DICT=1 and offsets
0/5000. Raw ZIPs stay in staging; existing submission ZIPs are never replaced.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo
import zipfile

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "shared/experiments"
PIN = E / "pinned/build56"
WORK = ROOT / "terra/experiments/gt_window56"
STATE = WORK / "queue_state.json"
RAW = E / "staging56_sol"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def save(stage, **fields):
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    previous.update(status=stage, updated_at_kst=stamp(), **fields)
    temp = STATE.with_suffix(".new.json")
    temp.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n")
    temp.replace(STATE)
    print(stage, fields, flush=True)


def active_training():
    r = subprocess.run(["ps", "ax", "-o", "pid=,command="], capture_output=True, text=True, check=True, timeout=30)
    found = []
    for line in r.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and re.match(r"^\S*/python(?:3(?:\.\d+)?)?\s+make_submit\.py(?:\s|$)", fields[1]):
            found.append(int(fields[0]))
    return found


def inspect_zip(number, expected_count, expected_features):
    p = ROOT / f"submissions/{number}_submit.zip"
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        rankers = sorted(n for n in z.namelist() if re.fullmatch(r"model/ranker_\d+\.txt", n))
        assert len(rankers) == expected_count
        hashes = [hashlib.sha256(z.read(n)).hexdigest() for n in rankers]
        assert len(set(hashes)) == expected_count
        for name in rankers:
            model = z.read(name).decode()
            assert len(re.findall(r"^Tree=", model, re.M)) == 600
            assert re.search(r"^feature_names=(.*)$", model, re.M).group(1).split() == expected_features
        assert 'os.environ.get("KO_DICT", "1")' in z.read("model/opus_lib.py").decode()
        return {"zip_sha256": sha(p), "ranker_sha256": hashes, "rankers": len(rankers)}


def runtime_check(number):
    sys.path.insert(0, str(Path(__file__).parent))
    from check_ready_47_48_sol import check_log
    p = ROOT / f"submissions/{number}_submit.zip"
    before = sha(p)
    log = E / f"check_{number}.log"
    assert not log.exists(), "Refusing to replace an existing check log"
    with log.open("x") as out:
        subprocess.run([sys.executable, "-u", str(ROOT / "shared/src/local_check.py"), str(p)],
                       cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)
    result = check_log(log.read_text())
    assert result is not None, "No complete check output; never accept an empty success log"
    assert sha(p) == before
    return dict(result, zip_sha256=before, local_check_sha256=sha(log))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = WORK / "queue_running.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        save("WAITING_REVIEW_AND_CAPACITY", queue_pid=os.getpid())
        last_count = None
        while True:
            gate = WORK / "REVIEW_READY.json"
            reviewed = gate.exists() and json.loads(gate.read_text()).get("status") == "APPROVED_FOR_CONTROLLED_BUILD"
            if not reviewed:
                time.sleep(30)
                continue
            try:
                running = active_training()
            except (OSError, subprocess.SubprocessError) as error:
                save("WAITING_PROCESS_CHECK", reason=str(error))
                time.sleep(30)
                continue
            if len(running) != last_count:
                save("WAITING_CAPACITY", active_training_pids=running, maximum_training_jobs=4)
                last_count = len(running)
            if len(running) <= 2:
                break
            time.sleep(30)
        manifest = json.loads((PIN / "BUILD_MANIFEST_SOL.json").read_text())
        for name, expected in manifest["sha256"].items():
            assert sha(PIN / name) == expected, f"Pinned source changed: {name}"
        assert json.loads((WORK / "feature_contract_check.json").read_text())["status"] == "PASS_IMPLEMENTATION_CONTRACT_ONLY"
        for n in (56, 57, 58):
            assert not (ROOT / f"submissions/{n}_submit.zip").exists()
            assert not (E / f"check_{n}.log").exists()
        RAW.mkdir(parents=True, exist_ok=True)
        workers = []
        for n, offset in ((56, 0), (57, 5000)):
            while len(active_training()) >= 4:
                time.sleep(30)
            assert not (RAW / f"{n}_submit.zip").exists()
            assert not (E / f"work{n}_sol").exists(), "Refusing to replace an existing model work folder"
            env = dict(os.environ, KO_DICT="1", LLM_DICT_CSV="llm_korean_scored.csv",
                       LLM_TRAIN_CSV="llm_korean_scored.csv", SUBMIT_NUM=str(n), SEEDS="3",
                       SEED_OFFSET=str(offset), OUT_DIR=f"shared/experiments/work{n}_sol",
                       CACHE_DIR=str(E / "cache"), SUBMISSION_DIR=str(RAW), MONO="0", PYTHONUNBUFFERED="1")
            with (E / f"make{n}.log").open("x") as out:
                p = subprocess.Popen(["nohup", sys.executable, "-u", "make_submit.py"], cwd=PIN, env=env,
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            workers.append((n, offset, p))
        save("TRAINING", workers=[{"number": n, "seed_offset": off, "pid": p.pid} for n, off, p in workers],
             manifest_sha256=sha(PIN / "BUILD_MANIFEST_SOL.json"), launch_active_training_before=running)
        for n, _, p in workers:
            code = p.wait()
            assert code == 0 and (RAW / f"{n}_submit.zip").exists(), f"Build {n} failed, exit={code}"
        save("POSTPROCESSING_AND_VALIDATING")
        artifacts = {}
        for n in (56, 57):
            target = ROOT / f"submissions/{n}_submit.zip"
            assert not target.exists()
            subprocess.run([sys.executable, str(ROOT / "shared/src/add_exclusion.py"),
                            str(RAW / f"{n}_submit.zip"), str(target)], cwd=ROOT, check=True)
            artifacts[str(n)] = inspect_zip(n, 3, manifest["expected_features"])
            artifacts[str(n)]["runtime"] = runtime_check(n)
        target = ROOT / "submissions/58_submit.zip"
        assert not target.exists()
        subprocess.run([sys.executable, str(ROOT / "shared/src/merge_rankers.py"), str(target),
                        str(ROOT / "submissions/56_submit.zip"), str(ROOT / "submissions/57_submit.zip")],
                       cwd=ROOT, check=True)
        artifacts["58"] = inspect_zip(58, 6, manifest["expected_features"])
        assert artifacts["58"]["ranker_sha256"] == artifacts["56"]["ranker_sha256"] + artifacts["57"]["ranker_sha256"]
        artifacts["58"]["runtime"] = runtime_check(58)
        (WORK / "build_receipt.json").write_text(json.dumps({"status": "READY_FOR_LEAD_REVIEW", "artifacts": artifacts,
            "manifest_sha256": sha(PIN / "BUILD_MANIFEST_SOL.json"), "completed_at_kst": stamp()}, ensure_ascii=False, indent=2) + "\n")
        save("READY_FOR_LEAD_REVIEW", merged_number=58, artifacts=artifacts)
    except BaseException as error:
        save("FAILED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
