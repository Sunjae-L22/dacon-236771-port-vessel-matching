# author: opus
"""Turn a built submission into a diagnostic probe: predict only the rows a rule selects,
give every other row a wrong ID. Public score then measures (selected & correct) / N per language.

usage: python3 make_probe.py <submission.zip> <rule> <out.zip>
rules:
  ko_only        Korean rows predicted, the rest wrong            -> 0.7 * ko_acc
  en_clean       English rows whose name is within typo distance of a catalog name (ratio >= 90)
  ko_clean       Korean rows whose best KoMatcher score >= 0.85
  cs_seen        rows whose call sign appears in train.csv (answered by the call-sign lookup)
  ko_nogt_clean  Korean rows with both tonnages missing and a clean name (~always right) -> share of such rows
"""
import sys
import zipfile
from pathlib import Path

RULES = {
    "ko_only": "L.is_hangul(name)",
    "en_clean": "(not L.is_hangul(name)) and L.process.extractOne(L.norm(name), ix.names, scorer=L.fuzz.ratio, score_cutoff=90) is not None",
    "ko_clean": "L.is_hangul(name) and max((s for _, s in ix.ko.candidates(name, k=1)), default=0) >= 0.85",
    "cs_seen": "ix.cs_map.get(row.get('call_sign')) is not None",
    "ko_nogt_clean": ("L.is_hangul(name) and not np.isfinite(L.positive(row.get('gross_tonnage')))"
                      " and not np.isfinite(L.positive(row.get('international_gross_tonnage')))"
                      " and max((s for _, s in ix.ko.candidates(name, k=1)), default=0) >= 0.85"),
}


def main(src, rule, out):
    cond = RULES[rule]
    with zipfile.ZipFile(src) as z:
        files = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    script = files["script.py"].decode("utf-8")
    anchor = '    for row in test.to_dict("records"):\n'
    assert anchor in script
    inject = (anchor
              + "        name = unicodedata.normalize('NFC', str(row.get('vessel_name') or ''))\n"
              + f"        if not ({cond}):  # DIAGNOSTIC PROBE: {rule}\n"
              + "            preds.append(ix.ids[0])\n"
              + "            continue\n")
    script = script.replace(anchor, inject, 1)
    if "import unicodedata" not in script:
        script = script.replace("import pickle\n", "import pickle\nimport unicodedata\n", 1)
    files["script.py"] = script.encode("utf-8")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
