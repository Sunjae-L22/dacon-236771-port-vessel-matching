"""External fixed-weight challenge. Pure row-local decision + optional real GPU views.
No labels, fitting, test aggregation, submission writing, or baseline edits.
"""
import sys
sys.dont_write_bytecode = True
import math
import unicodedata

FIELDS = ('vessel_name', 'gross_tonnage', 'international_gross_tonnage', 'vessel_type', 'flag_name')
MARGIN = 1.0
BATCH = 4
MAX_CANDIDATES = 63


def clean_row(row):
    return {k: row.get(k) for k in FIELDS}


def hangul(name):
    return any('\uac00' <= c <= '\ud7a3' for c in unicodedata.normalize('NFC', str(name)))


def route(name, gap, votes):
    if set(votes) != {'ko', 'en', 'hs'} or any(not isinstance(v, str) or not v.strip() for v in votes.values()):
        raise ValueError('all three real component outputs required')
    a, b, ours = (votes[k] for k in ('ko', 'hs', 'en'))
    if not hangul(name):
        return ours
    if not math.isfinite(float(gap)):
        raise ValueError('nonfinite Korean 110 gap')
    if gap >= 4.0:
        return ours
    if b == ours or a == b or a == ours:
        return b if b in (ours, a) else a
    return a


def shortlist(packet, catalog, excluded):
    pool = packet['pool']
    indexed = {}
    for x in pool:
        vid = x['id']
        if vid in indexed or vid not in catalog:
            raise ValueError('duplicate pool ID or unknown catalog ID')
        if any(not math.isfinite(float(x[k])) for k in ('raw', 'adjusted', 'sim_key', 'sim_llm')):
            raise ValueError('invalid candidate score')
        if x['raw'] < 0 or x['adjusted'] < 0:
            raise ValueError('probabilities must be nonnegative')
        indexed[vid] = x
    candidates = set()
    sources = {}
    for field, limit in (('adjusted', 20), ('raw', 20), ('sim_key', 10), ('sim_llm', 10)):
        ranked = sorted((x for x in pool if x['id'] not in excluded), key=lambda x: (-x[field], x['id']))[:limit]
        # Zero similarities carry no retrieval evidence; probability lists retain zeros for traceability.
        for x in ranked:
            if field.startswith('sim_') and x[field] <= 0:
                continue
            candidates.add(x['id'])
            sources.setdefault(x['id'], []).append(field)
    for role, vid in packet['votes'].items():
        if vid not in catalog:
            raise ValueError('component output outside catalog')
        if vid not in excluded:
            candidates.add(vid)
            sources.setdefault(vid, []).append('voter:' + role)
    ids = sorted(candidates)
    if len(ids) > MAX_CANDIDATES:
        raise AssertionError('fixed shortlist budget exceeded')
    return ids, sources


def decide(packet, ids, scores):
    """No labels accepted; native model gaps are engineering gates, not calibrated risk."""
    original = route(packet['row'].get('vessel_name'), packet['gap110'], packet['votes'])
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError('invalid shortlist')
    if set(scores) != {'qwen:evidence', 'encoder:phonetic'}:
        raise ValueError('exact two frozen views required')
    wins, gaps = {}, {}
    for name, values in scores.items():
        if len(values) != len(ids) or any(not math.isfinite(float(s)) for s in values):
            raise ValueError('missing/nonfinite neural scores: abort, never fallback')
        order = sorted(range(len(ids)), key=lambda i: (-float(values[i]), ids[i]))
        wins[name] = ids[order[0]]
        gaps[name] = float(values[order[0]]) - float(values[order[1]])
    if len(set(wins.values())) != 1:
        return {'prediction': original, 'reason': 'view_disagreement', 'winners': wins, 'gaps': gaps}
    winner = next(iter(wins.values()))
    if min(gaps.values()) < MARGIN:
        return {'prediction': original, 'reason': 'native_gap_below_1', 'winners': wins, 'gaps': gaps}
    modified = dict(packet['votes'], ko=winner)
    prediction = route(packet['row'].get('vessel_name'), packet['gap110'], modified)
    return {'prediction': prediction, 'challenger_ko': winner,
            'reason': 'accepted' if prediction != original else 'same_routed_prediction',
            'winners': wins, 'gaps': gaps}


def challenge(packet, catalog, excluded, views):
    """packet.pool: [{id, raw, adjusted, sim_key, sim_llm}], votes en/ko/hs.
    views: {name: callable(row, list[catalog-record], beams)->list[float]}.
    Component provenance/ZIP hashes and dataset completeness are runner responsibilities.
    """
    row = clean_row(packet['row'])
    original = route(row.get('vessel_name'), packet['gap110'], packet['votes'])
    # Validate every real voter before any protected/English/locked early return.
    for role, vid in packet['votes'].items():
        if vid not in catalog:
            raise ValueError('component output outside official catalog: ' + role)
    out = {'sample_id': str(packet['sample_id']), 'baseline': original, 'prediction': original}
    if not hangul(row.get('vessel_name')):
        return dict(out, reason='english')
    if packet['gap110'] >= 4:
        return dict(out, reason='protected110')
    if packet['votes']['en'] == packet['votes']['hs']:
        return dict(out, reason='router_locked_en_hs')
    ids, sources = shortlist(packet, catalog, excluded)
    if len(ids) < 2:
        return dict(out, reason='insufficient_nontrain_candidates')
    if set(views) != {'qwen:evidence', 'encoder:phonetic'}:
        raise ValueError('exact frozen-view contract required')
    beams = packet.get('llm_beams', [])
    if not isinstance(beams, (list, tuple)) or any(not isinstance(s, str) for s in beams):
        raise ValueError('bad query-only transliterations')
    scores = {}
    for name in sorted(views):
        values = []
        for start in range(0, len(ids), BATCH):
            batch = ids[start:start + BATCH]
            got = list(views[name](row, [clean_row(catalog[vid]) for vid in batch], beams))
            if len(got) != len(batch):
                raise ValueError('truncated scorer batch')
            values.extend(float(x) for x in got)
        scores[name] = values
    result = decide(dict(packet, row=row), ids, scores)
    return dict(out, **result, shortlist=ids, sources=sources, scores=scores)


def install_capture(neural, emit):
    """Call immediately after EncoderViewsEnsemble.load, BEFORE script's masked_score_groups.
    Leave source files/hash checks untouched. emit receives each unchanged v20 row trace.
    Production caller must enrich votes/gap from all-three-success baseline output.
    """
    raw_call = neural.baseline.score_groups
    original = neural.score_records
    state = {}
    def raw_capture(blocks):
        result = raw_call(blocks)
        state['raw'] = [x.copy() for x in result]
        return result
    neural.baseline.score_groups = raw_capture
    def score_records(rows, signals, blocks):
        masked_call = neural.baseline.score_groups
        state.clear()
        def adjusted_capture(bs):
            result = masked_call(bs)
            state['adjusted'] = [x.copy() for x in result]
            return result
        neural.baseline.score_groups = adjusted_capture
        try:
            output = original(rows, signals, blocks)
        finally:
            neural.baseline.score_groups = masked_call
        if not (len(rows) == len(signals) == len(output) == len(state['raw']) == len(state['adjusted'])):
            raise ValueError('capture length mismatch')
        for row, sig, raw, adjusted, final in zip(rows, signals, state['raw'], state['adjusted'], output):
            pool = []
            for j, ci in enumerate(sig['cand']):
                pool.append({'id': str(neural.r.ids[ci]), 'raw': float(raw[j]), 'adjusted': float(adjusted[j]),
                             'sim_key': float(sig['sim_key'][j]), 'sim_llm': float(sig['sim_llm'][j])})
            order = sorted(range(len(pool)), key=lambda j: (-float(adjusted[j]), j))[:20]
            mass = sum(adjusted[j] for j in order)
            if mass <= 0 or not math.isfinite(float(mass)):
                raise ValueError('invalid adjusted candidate mass')
            skipped = len(order) < 2 or (adjusted[order[0]] - adjusted[order[1]]) / mass > neural.weight/(1-neural.weight)+1e-6
            emit({'sample_id': str(row['sample_id']), 'row': clean_row(row), 'pool': pool,
                  'llm_beams': list(sig.get('llm_beams', ())), 'baseline_top20': [pool[j]['id'] for j in order],
                  'baseline_neural_skipped': bool(skipped), 'v20_final_scores': [float(x) for x in final]})
        return output
    neural.score_records = score_records
    return neural
