# author: argon
"""Submission ZIP patcher (`patch_submit_argon.py`) — fixed for `model/` internal ZIP paths.

Modes:
- `v4-lite` (default): Replaces `model/ko_translit_argon.py` inside the ZIP with
  `ko_translit_argon_v4.py` (which has ONLY the `ch->kh` regex fix + 3-rule LLM dictionary
  hygiene on top of `build31`'s exact `KoMatcher v2`, with ZERO extra hand-added words and
  ZERO changes to `opus_lib.py` candidate retrieval or feature definitions).
  Also ensures `KO_DICT = "1"` in `model/opus_lib.py` if patching `40/41/42_submit.zip` where
  `KO_DICT` was accidentally left at `"0"`.

Usage:
    python3 shared/src/patch_submit_argon.py submissions/36_submit.zip submissions/36_submit_argon_v4.zip
    python3 shared/src/patch_submit_argon.py submissions/42_submit.zip submissions/42_submit_argon_v4.zip
"""
from __future__ import annotations

import filecmp
from pathlib import Path
import sys
import zipfile


def patch_zip(src_zip: str | Path, dst_zip: str | Path, ensure_ko_dict_1: bool = True) -> None:
    src_path = Path(src_zip)
    dst_path = Path(dst_zip)
    v4_path = Path(__file__).resolve().parent / "ko_translit_argon_v4.py"
    v4_code = v4_path.read_text(encoding="utf-8")

    replaced_ko = False
    updated_opus = False

    with zipfile.ZipFile(src_path, "r") as zin, zipfile.ZipFile(dst_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in ("ko_translit_argon.py", "model/ko_translit_argon.py"):
                zout.writestr(item, v4_code.encode("utf-8"))
                replaced_ko = True
            elif ensure_ko_dict_1 and item.filename in ("opus_lib.py", "model/opus_lib.py"):
                text = data.decode("utf-8")
                new_text = text.replace('os.environ.get("KO_DICT", "0") != "1"', 'os.environ.get("KO_DICT", "1") != "1"')
                if new_text != text:
                    updated_opus = True
                zout.writestr(item, new_text.encode("utf-8"))
            else:
                zout.writestr(item, data)

    if not replaced_ko:
        raise RuntimeError(f"Could not find model/ko_translit_argon.py in {src_path}")
    same = filecmp.cmp(src_path, dst_path, shallow=False)
    if same:
        raise RuntimeError(f"ERROR: {dst_path} is byte-identical to {src_path}!")
    print(
        f"Patched {src_path.name} -> {dst_path.name} | replaced_ko={replaced_ko}, "
        f"ko_dict_default_fixed={updated_opus}, byte_identical={same} "
        f"({dst_path.stat().st_size:,} bytes vs orig {src_path.stat().st_size:,} bytes)"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 shared/src/patch_submit_argon.py <input.zip> <output.zip>")
        sys.exit(1)
    patch_zip(sys.argv[1], sys.argv[2])
