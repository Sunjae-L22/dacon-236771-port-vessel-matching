# author: opus
"""Jamo-level Korean typos (copied from opus/stage2/jamo_fingerprint.py) for build107 training copies."""

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
V = {"ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅓ": "ㅗ", "ㅗ": "ㅓ", "ㅡ": "ㅜ", "ㅜ": "ㅡ", "ㅒ": "ㅖ", "ㅖ": "ㅒ", "ㅙ": "ㅞ", "ㅞ": "ㅙ", "ㅚ": "ㅞ"}
C = {"ㄱ": "ㅋ", "ㅋ": "ㄱ", "ㄷ": "ㅌ", "ㅌ": "ㄷ", "ㅂ": "ㅍ", "ㅍ": "ㅂ", "ㅈ": "ㅊ", "ㅊ": "ㅈ", "ㅅ": "ㅆ", "ㅆ": "ㅅ",
     "ㄲ": "ㄱ", "ㄸ": "ㄷ", "ㅃ": "ㅂ", "ㅉ": "ㅈ"}


def split(ch):
    o = ord(ch) - 0xAC00
    return o // 588, (o % 588) // 28, o % 28


def join(a, b, c):
    return chr(0xAC00 + a * 588 + b * 28 + c)


def jamo_typo(s, rng):
    pos = [i for i, ch in enumerate(s) if "가" <= ch <= "힣"]
    for _ in range(10):
        if not pos:
            return s
        i = pos[int(rng.integers(len(pos)))]
        a, b, c = split(s[i])
        op = int(rng.integers(3))
        if op == 0 and JUNG[b] in V:
            b = JUNG.index(V[JUNG[b]])
        elif op == 1 and CHO[a] in C:
            a = CHO.index(C[CHO[a]])
        elif op == 2:
            c = 0 if c else JONG.index(["ㄴ", "ㅇ", "ㄹ", "ㅅ"][int(rng.integers(4))])
        else:
            continue
        return s[:i] + join(a, b, c) + s[i + 1:]
    return s
