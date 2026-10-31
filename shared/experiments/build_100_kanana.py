# author: opus
"""100/101: 93 with the Korean route retrained on Kanana-2 LLM spellings.

101 (--num 101): rankers trained on Kanana rows, KoMatcher dictionary kept Qwen v1, so stats.pkl must equal 93's.

Takes 93 (Korean 72 rankers + English 65/62) and swaps in the six Kanana-trained rankers
(kanana_builds/900 = aug seeds 0/1000/2000, 901 = 5000/6000/7000) and their stats.pkl, whose
ko_pairs carry the Kanana dictionary rows. Everything else, including script.py, stays 93's.
"""
import pickle
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "shared/experiments/kanana_builds"
import sys
NUM = int(sys.argv[sys.argv.index("--num") + 1]) if "--num" in sys.argv else 100
SRC = {100: ("900", "901"), 101: ("902", "903")}[NUM]


def main():
    out = ROOT / "submissions" / f"{NUM}_submit.zip"
    assert not out.exists(), out
    z900, z901 = (zipfile.ZipFile(B / f"{n}_submit.zip") for n in SRC)
    assert z900.read("model/stats.pkl") == z901.read("model/stats.pkl")
    z93 = zipfile.ZipFile(ROOT / "submissions/93_submit.zip")
    for name in ("model/opus_lib.py", "model/ko_translit_argon.py"):
        assert z900.read(name) == z93.read(name), name  # same inference code as 93
    new = pickle.loads(z900.read("model/stats.pkl"))
    old = pickle.loads(z93.read("model/stats.pkl"))
    for k in ("cs_map", "flag_p", "type_p", "alias_rows"):
        assert new[k] == old[k], k
    print("ko_pairs", len(old["ko_pairs"]), "->", len(new["ko_pairs"]))
    if NUM == 101:
        assert new["ko_pairs"] == old["ko_pairs"], "101 must keep the Qwen dictionary"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as w:
        for info in z93.infolist():
            if info.filename.startswith("model/ranker_") or info.filename == "model/stats.pkl":
                continue
            w.writestr(info, z93.read(info))
        w.writestr("model/stats.pkl", z900.read("model/stats.pkl"))
        for i, (z, j) in enumerate([(z900, 0), (z900, 1), (z900, 2), (z901, 0), (z901, 1), (z901, 2)]):
            w.writestr(f"model/ranker_{i}.txt", z.read(f"model/ranker_{j}.txt"))
    print(NUM, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
