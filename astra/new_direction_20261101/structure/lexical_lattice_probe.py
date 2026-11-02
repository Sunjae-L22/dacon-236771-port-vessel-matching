"""Five-fold real-query lexical-context coverage probe, no prefit rankers.

Only exactly spaced/equal-length bilingual tokens induce correspondences.
This tests a narrow, falsifiable alternative to per-token top1 translation.
It is NOT a C_vote accuracy evaluation or a transliteration/noise model.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import product
import csv, hashlib, json, math, re, unicodedata

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def toks(value):
    value = unicodedata.normalize('NFC', value).upper()
    return re.findall(r'[A-Z]+|[가-힣]+|[0-9]+', value)

def fold(vid):
    return int(hashlib.sha256(('lexical-lattice-v1:' + vid).encode()).hexdigest()[:8], 16) % 5

def main():
    cat = list(csv.DictReader((ROOT/'data/open/vessel_catalog.csv').open(encoding='utf-8-sig')))
    train = list(csv.DictReader((ROOT/'data/open/train.csv').open(encoding='utf-8-sig')))
    byid = {r['vessel_id']: r for r in cat}
    korean = [r for r in train if any('가' <= c <= '힣' for c in r['vessel_name'])]
    summary = Counter()
    rows = []
    mappings = []
    for f in range(5):
        fit = [r for r in korean if fold(r['target_vessel_id']) != f]
        fitids = {r['target_vessel_id'] for r in train if fold(r['target_vessel_id']) != f}
        held = [r for r in korean if fold(r['target_vessel_id']) == f]
        counts = defaultdict(Counter)
        seen = set()
        for r in fit:
            source = toks(r['vessel_name'])
            target = toks(byid[r['target_vessel_id']]['vessel_name'])
            if len(source) != len(target):
                continue
            for a,b in zip(source,target):
                key = (r['target_vessel_id'], a, b)
                if key not in seen and any('가' <= c <= '힣' for c in a):
                    counts[a][b] += 1
                    seen.add(key)
        index = defaultdict(list)
        for r in cat:
            if r['vessel_id'] not in fitids:
                index[tuple(toks(r['vessel_name']))].append(r['vessel_id'])
        ambiguous = {a: dict(v) for a,v in counts.items() if len(v) > 1}
        mappings.append({'fold': f, 'fit_korean_vessels': len({r['target_vessel_id'] for r in fit}),
                         'ambiguous_tokens': ambiguous})
        for r in held:
            summary['heldout_rows'] += 1
            source = toks(r['vessel_name'])
            options = [counts.get(t, {}) if any('가' <= c <= '힣' for c in t) else {t:1} for t in source]
            record = {'fold': f, 'vessel_id': r['target_vessel_id'], 'query': r['vessel_name'],
                      'target_name': byid[r['target_vessel_id']]['vessel_name']}
            if not options or any(not x for x in options):
                summary['unsupported_unseen_token'] += 1
                record['status'] = 'unsupported_unseen_token'
                rows.append(record)
                continue
            summary['full_token_coverage'] += 1
            has_amb = any(len(x)>1 for x in options)
            summary['full_coverage_with_ambiguity'] += has_amb
            top1 = tuple(sorted(x, key=lambda t: (-x[t], t))[0] for x in options)
            bare = index.get(top1, [])
            candidates = {}
            for seq in product(*(list(x) for x in options)):
                p = math.prod(options[i][t]/sum(options[i].values()) for i,t in enumerate(seq))
                for vid in index.get(seq, []):
                    candidates[vid] = max(candidates.get(vid, 0), p)
            ordered = sorted(candidates, key=lambda vid: (-candidates[vid], vid))
            target = r['target_vessel_id']
            summary['top1_translation_target_in_catalog_match'] += target in bare
            summary['lattice_target_in_catalog_match'] += target in candidates
            added = target in candidates and target not in bare
            summary['lattice_extra_target_coverage'] += added
            summary['lattice_unique_target_match'] += list(candidates) == [target]
            summary['lattice_top_rank_correct'] += bool(ordered and ordered[0] == target)
            record.update(status='covered', ambiguous=has_amb, top1_translation=list(top1),
                          top1_candidate_ids=bare, lattice_candidates=ordered,
                          added_target=added)
            rows.append(record)
    result = {'counts': dict(summary), 'protocol': 'Five folds hashed by vessel; dictionaries fitted only on four folds; all fit vessels excluded from catalog retrieval; genuine Korean queries unchanged; no pretrained model, KoCat, English noise, GPU, metadata features, or thresholds',
              'scope': 'Exact token correspondence lexical lattice; an upper bound only on this mechanism, not all structure-aware methods. No C_vote predictions, so no C_vote rescue/damage or Public extrapolation.',
              'source_train_sha256': hashlib.sha256((ROOT/'data/open/train.csv').read_bytes()).hexdigest(),
              'source_catalog_sha256': hashlib.sha256((ROOT/'data/open/vessel_catalog.csv').read_bytes()).hexdigest()}
    (OUT/'LEXICAL_RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT/'LEXICAL_ROWS.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (OUT/'LEXICAL_AMBIGUITY.json').write_text(json.dumps(mappings, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
