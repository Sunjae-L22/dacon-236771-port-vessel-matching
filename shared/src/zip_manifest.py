# author: opus
"""Record what each submission zip actually ships: shipped KO_DICT default, ranker count, file hashes,
and the size and origin of the Korean dictionary pairs in stats.pkl.

usage: python3 shared/src/zip_manifest.py <out.md> <zip> [<zip> ...]
"""
import hashlib
import pickle
import re
import sys
import zipfile
from pathlib import Path


def row(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    if "model/opus_lib.py" not in names:
        prefixes = sorted(n[:-len("model/opus_lib.py")] for n in names if n.endswith("/model/opus_lib.py"))
        if not prefixes:
            raise ValueError(f"{path}: no member opus_lib.py")
        values = [member_row(z, prefix) for prefix in prefixes]
        combined = ["/".join(v[col] for v in values) for col in range(7)]
        combined[0] = str(sum(int(v[0]) for v in values))
        return [Path(path).name, *combined, hashlib.sha256(z.read("script.py")).hexdigest()[:12],
                hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]]
    return [Path(path).name, *member_row(z), hashlib.sha256(z.read("script.py")).hexdigest()[:12],
            hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]]


def member_row(z, prefix=""):
    names = [n[len(prefix):] for n in z.namelist() if n.startswith(prefix)]
    sha = lambda n: hashlib.sha256(z.read(prefix + n)).hexdigest()[:12] if n in names else "-"
    lib = z.read(prefix + "model/opus_lib.py").decode("utf-8")
    ko_dict = re.search(r'KO_DICT", "([01])"', lib)
    stats = pickle.loads(z.read(prefix + "model/stats.pkl"))
    pairs = stats["ko_pairs"]
    n_llm = sum(str(p.get("sample_id", "")).startswith("LLM") for p in pairs)
    with_sid = "sample_id" in pairs[0] if pairs else False
    return [str(sum("ranker" in n for n in names)), ko_dict.group(1) if ko_dict else "?",
            str(len(pairs)), str(n_llm) if with_sid else "(sample_id 없음)",
            sha("model/ko_translit_argon.py"), sha("model/opus_lib.py"), sha("model/stats.pkl")]


def main(out, *zips):
    head = ["zip", "rankers", "KO_DICT", "ko_pairs", "LLM pairs", "matcher", "opus_lib", "stats", "script", "zip sha"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(row(z)) + " |" for z in zips]
    Path(out).write_text("# 제출 zip manifest (sha256 앞 12자리)\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
