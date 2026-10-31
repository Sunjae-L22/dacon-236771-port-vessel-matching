"""Row-local diagnostic predicates; no target labels or cross-query statistics."""
import re
import unicodedata

CONDITIONS = (
    'tonnage_gap', 'short_roman', 'latin', 'digits', 'mixed', 'no_space',
    'suffix_ho', 'sister', 'flag_disagree', 'type_disagree', 'partial_name',
)

def predicates(name, first_name, second_name, first, second, features):
    c = {n: i for i, n in enumerate(features)}
    get = lambda n: float(first[c[n]])
    nxt = lambda n: float(second[c[n]]) if second is not None else get(n)
    name = unicodedata.normalize('NFC', str(name)).strip()
    tokens = lambda s: re.findall(r'[A-Z0-9]+', str(s).upper())
    a, b = tokens(first_name), tokens(second_name)
    latin = bool(re.search(r'[A-Za-z]', name))
    digits = bool(re.search(r'[0-9]', name))
    return dict(
        tonnage_gap=get('q_gt_missing') == 0 and get('c_gt_missing') == 0 and get('best_gap') > .10,
        # Raw Romanized-query / English-name length, NOT Hangul transliteration length.
        short_roman=get('len_q') / max(get('len_c'), 1) < .7,
        latin=latin, digits=digits, mixed=latin or digits,
        no_space=not bool(re.search(r'\s', name)), suffix_ho=name.endswith('호'),
        sister=bool(a and b and len(a[0]) >= 2 and a[0] == b[0]),
        # Learned probability disagreements, NOT verified country/type mismatches.
        flag_disagree=get('flag_p') < .1 and nxt('flag_p') > get('flag_p'),
        type_disagree=get('type_p') < .1 and nxt('type_p') > get('type_p'),
        partial_name=get('ko_partial_sim') >= .85 and get('ko_trans_sim') < .7,
    )
