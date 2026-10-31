"""Rule-based Korean romanisation variants (RR / MR-ish / common ad-hoc spellings)."""
from __future__ import annotations

import re

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

# (RR, alt) per jamo
CHO_R = {"ㄱ": ("G", "K"), "ㄲ": ("KK", "K"), "ㄴ": ("N", "N"), "ㄷ": ("D", "T"), "ㄸ": ("TT", "T"),
         "ㄹ": ("R", "L"), "ㅁ": ("M", "M"), "ㅂ": ("B", "P"), "ㅃ": ("PP", "P"), "ㅅ": ("S", "S"),
         "ㅆ": ("SS", "S"), "ㅇ": ("", ""), "ㅈ": ("J", "CH"), "ㅉ": ("JJ", "J"), "ㅊ": ("CH", "CH"),
         "ㅋ": ("K", "K"), "ㅌ": ("T", "T"), "ㅍ": ("P", "P"), "ㅎ": ("H", "H")}
JUNG_R = {"ㅏ": ("A", "A"), "ㅐ": ("AE", "A"), "ㅑ": ("YA", "YA"), "ㅒ": ("YAE", "YA"), "ㅓ": ("EO", "U"),
          "ㅔ": ("E", "E"), "ㅕ": ("YEO", "YOU"), "ㅖ": ("YE", "YE"), "ㅗ": ("O", "O"), "ㅘ": ("WA", "WA"),
          "ㅙ": ("WAE", "WE"), "ㅚ": ("OE", "OI"), "ㅛ": ("YO", "YO"), "ㅜ": ("U", "OO"), "ㅝ": ("WO", "WO"),
          "ㅞ": ("WE", "WE"), "ㅟ": ("WI", "WI"), "ㅠ": ("YU", "YOO"), "ㅡ": ("EU", "U"), "ㅢ": ("UI", "EE"),
          "ㅣ": ("I", "EE")}
JONG_R = {"": ("", ""), "ㄱ": ("K", "G"), "ㄲ": ("K", "K"), "ㄳ": ("K", "K"), "ㄴ": ("N", "N"), "ㄵ": ("N", "N"),
          "ㄶ": ("N", "N"), "ㄷ": ("T", "T"), "ㄹ": ("L", "L"), "ㄺ": ("K", "K"), "ㄻ": ("M", "M"), "ㄼ": ("L", "L"),
          "ㄽ": ("L", "L"), "ㄾ": ("L", "L"), "ㄿ": ("P", "P"), "ㅀ": ("L", "L"), "ㅁ": ("M", "M"), "ㅂ": ("P", "B"),
          "ㅄ": ("P", "P"), "ㅅ": ("T", "S"), "ㅆ": ("T", "T"), "ㅇ": ("NG", "NG"), "ㅈ": ("T", "T"), "ㅊ": ("T", "T"),
          "ㅋ": ("K", "K"), "ㅌ": ("T", "T"), "ㅍ": ("P", "P"), "ㅎ": ("T", "H")}


def _syl(ch):
    code = ord(ch) - 0xAC00
    if not (0 <= code < 11172):
        return None
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def romanize(text: str, cons: int, vow: int, woo: bool = False) -> str:
    out = []
    prev_vowel = False
    for ch in text:
        d = _syl(ch)
        if d is None:
            if ch.isalnum():
                out.append(ch.upper())
            prev_vowel = False
            continue
        c, v, j = d
        cs = CHO_R[c][cons]
        if c == "ㅅ" and v in ("ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ"):
            cs = "SH" if cons else "S"
        vs = JUNG_R[v][vow]
        if woo and c == "ㅇ" and v == "ㅜ":
            vs = "WOO"
        if c == "ㅇ" and v == "ㅣ" and vow == 1:
            vs = "EE"
        out.append(cs + vs + JONG_R[j][cons])
    return "".join(out)


def variants(name: str) -> list[str]:
    s = re.sub(r"제\s*\d+\s*호?", " ", name)
    s = re.sub(r"\d+\s*호", " ", s)
    s = re.sub(r"\d+", " ", s)
    toks = s.split()
    if toks and toks[-1].endswith("호") and len(toks[-1]) >= 2 and len(toks) + len(toks[-1]) > 3:
        toks[-1] = toks[-1][:-1]
    t = "".join(toks)
    if not t:
        return []
    vs = {romanize(t, 0, 0), romanize(t, 1, 1), romanize(t, 0, 1), romanize(t, 1, 0),
          romanize(t, 1, 1, woo=True), romanize(t, 0, 0, woo=True)}
    return [v for v in vs if v]
