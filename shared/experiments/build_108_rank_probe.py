# author: opus
"""108 / 109 (diagnostic): on Public hard Korean rows, how often is the answer 93's runner-up / third?

102 split Public Korean into easy rows (gap >= 4; 371 correct) and ~159 hard rows (121 correct by 93's
leader, 38 wrong). Every model change since (99, 100, 105, 107) only reshuffled hard rows with net <= 0,
so before building a heavier re-ranker (LLM judge on the L4) we measure whether the answers are inside
93's top candidates at all. Both are 93 except on hard Korean rows (same gap expression as 102/105):
  108: output the 2nd allowed candidate  -> Korean count = 371 + (hard rows whose answer is 93's #2)
  109: output the 3rd allowed candidate  -> Korean count = 371 + (hard rows whose answer is 93's #3)
English rows are 93's.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from build_105_hard_signal import SCRIPT  # noqa: E402

RULE = {
    108: '''            pick = min(1, len(allowed) - 1)''',
    109: '''            pick = min(2, len(allowed) - 1)''',
}


def main():
    z93 = zipfile.ZipFile(ROOT / "submissions/93_submit.zip")
    for num, what in ((108, "runner-up (diagnostic)"), (109, "third candidate (diagnostic)")):
        out = ROOT / "submissions" / f"{num}_submit.zip"
        assert not out.exists(), out
        script = SCRIPT.format(num=num, what=what, rule=RULE[num]).replace(
            "see build_105_hard_signal.py", "see build_108_rank_probe.py")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as w:
            for info in z93.infolist():
                if info.filename != "script.py":
                    w.writestr(info, z93.read(info))
            w.writestr("script.py", script)
        print(num, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
