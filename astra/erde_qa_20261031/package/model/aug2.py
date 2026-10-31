"""Keyboard-adjacent hangul typos (2-beolsik layout) and syllable deletion, for compound-damage augmentation."""
import random
import translit as T
CHO, JUNG, JONG = T.CHO, T.JUNG, T.JONG
def _dec(c): o = ord(c) - 0xAC00; return o // 588, (o % 588) // 28, o % 28
def _enc(a, b, c): return chr(0xAC00 + a * 588 + b * 28 + c)
def _ish(c): return '가' <= c <= '힣'
_ROWS = [list('ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ'), list('ㅁㄴㅇㄹㅎㅗㅓㅏㅣ'), list('ㅋㅌㅊㅍㅠㅜㅡ')]
NB = {}
for ri, row in enumerate(_ROWS):
    for ci, k in enumerate(row):
        s = set()
        for dc in (-1, 1):
            if 0 <= ci + dc < len(row): s.add(row[ci + dc])
        for dr in (-1, 1):
            if 0 <= ri + dr < 3:
                for dc in (0, 1, -1):
                    if 0 <= ci + dc < len(_ROWS[ri + dr]): s.add(_ROWS[ri + dr][ci + dc])
        NB[k] = s
def _sub(ch, rng):
    a, b, c = _dec(ch); w = rng.random()
    if w < 0.45:
        cand = [x for x in NB.get(CHO[a], ()) if x in CHO]
        if cand: a = CHO.index(rng.choice(cand))
    elif w < 0.9:
        cand = [x for x in NB.get(JUNG[b], ()) if x in JUNG]
        if cand: b = JUNG.index(rng.choice(cand))
    else:
        cur = JONG[c]
        cand = [x for x in NB.get(cur, ()) if x in JONG] if cur else ['ㄴ', 'ㄹ', 'ㅇ']
        if cand: c = JONG.index(rng.choice(cand))
    return _enc(a, b, c)
def kbd_typo(name, rng, k=1):
    if not name: return name
    s = list(name); idx = [i for i, ch in enumerate(s) if _ish(ch)]
    for i in rng.sample(idx, min(k, len(idx))): s[i] = _sub(s[i], rng)
    return ''.join(s)
def syl_del(name, rng, k=1):
    if not name: return None
    s = list(name); idx = [i for i, ch in enumerate(s) if _ish(ch)]
    if len(idx) < k + 3: return None
    for i in sorted(rng.sample(idx, k), reverse=True): del s[i]
    return ''.join(s)
def ko_noise(name, rng):
    """1-2 typos, keyboard-adjacent or phonetically similar."""
    import aug
    k = 1 if rng.random() < 0.6 else 2
    out = name
    for _ in range(k):
        out = kbd_typo(out, rng, 1) if rng.random() < 0.5 else aug.ko_typo(out, rng)
    return out
