# author: terra (Sol)
"""Small, synthetic feature-contract checks; no model fit or dataset inference."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MESSAGE = ROOT / "argon/007_20261027-1522_reply-opus021-v4lite-oof-and-type-d-features.md"
OUT = ROOT / "terra/experiments/20261027_argon007_review"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    msg = MESSAGE.read_text()
    snippet = next(s.split("```", 1)[0] for s in msg.split("```python\n")[1:]
                   if "win_mask = best_gap <= 0.12" in s)
    snippet = textwrap.dedent(snippet)
    cases = {
        "outside_beats_window": ([.9, .6, .3], [.5, .01, .02]),
        "tied_window_leaders": ([.9, .6, .6], [.5, .01, .02]),
        "singleton_window": ([.9, .6], [.5, .01]),
        "no_compatible_candidate": ([.9, .6], [.5, .4]),
        "near_tie_window": ([.6000000005, .6, .3], [.01, .02, .03]),
    }
    rows = {}
    for name, (comb, gap) in cases.items():
        ns = {"np": np, "n": len(comb), "comb": np.array(comb),
              "best_gap": np.array(gap), "ko_cols": {
                  k: np.array(comb) for k in ("ko_tail_sim", "ko_partial_sim", "ko_token_cov")}}
        exec(compile(snippet, str(MESSAGE), "exec"), ns)
        original_mask = np.array(gap) <= .12
        rows[name] = {"comb": comb, "best_gap": gap,
                      "original_window_count": int(original_mask.sum()),
                      **{k: ns[k].tolist() for k in (
                          "ko_gt_win_best", "ko_gt_win_margin", "ko_gt_win_zscore")}}
        rows[name]["positive_margin_count_all"] = int((ns["ko_gt_win_margin"] > 0).sum())
        rows[name]["positive_margin_count_inside_original_window"] = int(
            ((ns["ko_gt_win_margin"] > 0) & original_mask).sum())
        old_rel = np.array(comb) - max(comb)
        new_rel = ns["ko_gt_win_best"]
        assert np.allclose(old_rel[:, None] - old_rel[None, :],
                           new_rel[:, None] - new_rel[None, :])
        assert all(np.isfinite(ns[k]).all() for k in (
            "ko_gt_win_best", "ko_gt_win_margin", "ko_gt_win_zscore"))

    model_paths = [ROOT / f"submissions/{n}_submit.zip" for n in (31, 36)]
    models = {}
    for p in model_paths:
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.startswith("model/ranker") and n.endswith(".txt")]
            models[p.name] = {"ranker_count": len(names),
                              "ranker_0_sha256": sha(z.read("model/ranker_0.txt"))}
    assert models[model_paths[0].name]["ranker_0_sha256"] == models[model_paths[1].name]["ranker_0_sha256"]

    paths = [MESSAGE, ROOT / "shared/src/ko_translit_argon_v4.py",
             ROOT / "shared/experiments/pinned/build49/ko_translit_argon.py",
             ROOT / "shared/experiments/pinned/build31/make_submit.py",
             ROOT / "shared/experiments/pinned/build31/opus_lib.py",
             ROOT / "shared/src/merge_rankers.py", ROOT / "shared/src/opus_cv.py"]
    # Parse source only. Do not import libraries that might create caches outside terra/.
    for p in paths[1:]:
        ast.parse(p.read_text())
    result = {"boundary": "Synthetic snippet checks and read-only source/archive provenance; no OOF reproduction, model fit, GPU, official test, or submission",
              "source_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths},
              "snippet_sha256": sha(snippet.encode()), "cases": rows, "archives": models,
              "ranker_0_31_equals_36": True}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "proof.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases_checked": len(rows), "archives": models,
                      "outside_case": rows["outside_beats_window"],
                      "proof": str(OUT / "proof.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
