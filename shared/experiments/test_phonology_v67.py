"""Test enhanced phonology (woo/hyun, -se->j, ch->kh, glide vowel equivalence) on held-out queries."""
from __future__ import annotations
import bisect
import csv
import math
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(unicodedata.normalize("NFD", "/Users/sunjae/Desktop/항만"))
sys.path.insert(0, str(ROOT / "shared/experiments/pinned/build31"))
import ko_translit_argon as KA


def norm_glide_lat(s: str) -> str:
    """Normalize Korean-English romanization variants (woo->u, hyun->hyeon, yo->yeo, yu->u at start)."""
    s = s.lower()
    s = re.sub(r"woo", "u", s)
    s = re.sub(r"hyun", "hyeon", s)
    s = re.sub(r"hyoung|hyung", "hyeong", s)
    s = re.sub(r"young|yung", "yeong", s)
    s = re.sub(r"([aeiou])se\b", r"\1j", s)
    return s


for ko_q, en_c in [
    ("우현", "Woohyun Sky"),
    ("우현 카스이", "Woohyun Sky"),
    ("우리 썬", "Woori Sun"),
    ("와이엔 여수", "YN Yeosu"),
    ("켐로드 로즈", "Chemroad Rose"),
    ("써니 아카시아", "Sunny Acacia"),
    ("케이에스 썬글로리", "KS Sunglory"),
    ("아스펜 익스프레스", "Aspen Express"),
    ("글로비스 슈피리어", "Glovis Superior"),
    ("에스아이티씨 웨이하이", "SITC Weihai"),
    ("카카를로스", "Don Carlos"),
]:
    kr = KA.romanize_hangul(ko_q.replace(" ", ""), drop_eu=True)
    en_body, _ = KA.extract_digits_and_clean_en(en_c, join_hyphen=True)
    e2k = KA.en_to_ko_phonetic_lat(KA.expand_en_acronyms(en_body), keep_precons_r=False)
    print(f"ko='{ko_q}' -> kr='{kr}' (glide='{norm_glide_lat(kr)}') | en='{en_c}' -> e2k='{e2k}' (glide='{norm_glide_lat(e2k)}')")
