# author: Sol
"""Freeze the next controlled experiment: 33 recipe + two window features."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "shared/experiments/pinned/build43"
OUT = ROOT / "shared/experiments/pinned/build56"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new, 1)


def main():
    assert not OUT.exists(), "Never replace an existing pinned build"
    shutil.copytree(BASE, OUT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helper_path = ROOT / "terra/src/gt_window_features_sol.py"
    helper_text = helper_path.read_text()
    helper = next(n for n in ast.parse(helper_text).body if isinstance(n, ast.FunctionDef) and n.name == "window_features")
    function = ast.get_source_segment(helper_text, helper).replace("def window_features(", "def _sol_window_features(", 1)
    lib_path = OUT / "opus_lib.py"
    lib = lib_path.read_text()
    lib = replace_once(lib, "FEATURES = [", function + "\n\n\nFEATURES = [")
    lib = replace_once(lib, '\n\n\ndef initial_parts(q: str):',
                       '\nFEATURES += ["ko_gt_win_diff_max", "ko_gt_win_margin"]\n\n\ndef initial_parts(q: str):')
    lib = replace_once(lib, "    F = np.zeros((n, len(FEATURES)), np.float32)\n",
                       "    F = np.zeros((n, len(FEATURES)), np.float32)\n"
                       "    F[:, -2:] = np.nan  # Sol: only the two new columns; non-Korean stays missing.\n")
    lib = replace_once(lib, "    return idx, F\n",
                       "    if ko and n:\n"
                       "        F[:, -2:] = _sol_window_features(F[:, col[\"ko_combined_score\"]], F[:, col[\"best_gap\"]])\n"
                       "    return idx, F\n")
    lib_path.write_text("# modified by Sol: only two tonnage-window name features appended\n" + lib)
    make_path = OUT / "make_submit.py"
    make = make_path.read_text()
    # Raw archives stay in a new staging folder. Postprocessing creates a new final ZIP.
    make = replace_once(make, '    (ROOT / "submissions").mkdir(exist_ok=True)\n',
                        '    submission_dir = Path(os.environ.get("SUBMISSION_DIR", str(ROOT / "submissions")))\n'
                        '    submission_dir.mkdir(parents=True, exist_ok=True)\n')
    make = replace_once(make, '(ROOT / "submissions").glob("[0-9][0-9]_submit.zip")',
                        'submission_dir.glob("[0-9][0-9]_submit.zip")')
    make = replace_once(make, '    zp = ROOT / "submissions" / f"{num:02d}_submit.zip"\n',
                        '    zp = submission_dir / f"{num:02d}_submit.zip"\n'
                        '    assert not zp.exists(), "Refusing to overwrite an existing raw ZIP"\n')
    make_path.write_text("# modified by Sol: immutable raw staging output; training recipe unchanged\n" + make)
    assert sha(OUT / "ko_translit_argon.py") == sha(BASE / "ko_translit_argon.py")
    assert sha(OUT / "opus_cv.py") == sha(BASE / "opus_cv.py")
    for name in ("opus_lib.py", "make_submit.py"):
        ast.parse((OUT / name).read_text())
    files = ["opus_lib.py", "ko_translit_argon.py", "opus_cv.py", "make_submit.py", "cv.py",
             "llm_korean_scored.csv"]
    manifest = {"status": "PINNED_NOT_TRAINING", "recipe": "33 base recipe plus two appended window columns",
                "source_build": str(BASE), "pinned_build": str(OUT), "submissions": [56, 57, 58],
                "source_archives": {"56": 0, "57": 5000}, "merged": {"58": [56, 57]},
                "KO_DICT": "1", "dict_csv": "llm_korean_scored.csv", "train_csv": "llm_korean_scored.csv",
                "feature_contract": {"score": "existing forward ko_combined_score", "threshold": 0.12,
                    "outside_or_empty": ["NaN", "NaN"], "singleton": [0, "NaN"], "tied_max": [0, 0],
                    "candidate_pool": "unchanged, before post-ranking exclusion", "old_features": "unchanged"},
                "sha256": {n: sha(OUT / n) for n in files},
                "base_sha256": {n: sha(BASE / n) for n in files}, "helper_sha256": sha(helper_path)}
    (OUT / "BUILD_MANIFEST_SOL.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "build": str(OUT), "new_features": 2}, ensure_ascii=False))


if __name__ == "__main__":
    main()
