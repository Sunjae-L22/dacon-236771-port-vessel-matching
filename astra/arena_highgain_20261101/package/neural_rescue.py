"""Frozen two-view rescue; no training, truth, cross-row state, or submission writer.

Inference packet v1 is captured from the actual post-mask/post-KoW v20 ensemble.
Only this module's run_one is the candidate; capture does not change baseline.
"""
import math
import hashlib
import json
import sys
import unicodedata
sys.dont_write_bytecode = True

ROW_KEYS = {'vessel_name', 'gross_tonnage', 'international_gross_tonnage',
            'vessel_type', 'flag_name', 'call_sign', 'sample_id'}
PACKET_KEYS = {'sample_id', 'row', 'ids', 'base', 'post', 'signals', 'beams',
               'excluded', 'v20', 'en', 'hs', 'gap', 'components_ok'}
SIGNALS = ('sim_key', 'sim_llm', 'sim_compact', 'sim_key_word')


def packet_fingerprint(packet):
    payload = json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def hangul(row):
    return any('\uac00' <= c <= '\ud7a3' for c in unicodedata.normalize('NFC', str(row.get('vessel_name', ''))))


def route(row, gap, v20, en, hs):
    if not hangul(row) or gap >= 4:
        return en
    if hs == en or v20 == hs or v20 == en:
        return hs if hs in (en, v20) else v20
    return v20


def validate(p):
    if set(p) != PACKET_KEYS or set(p['row']) - ROW_KEYS:
        raise ValueError('Unexpected/missing packet fields; labels are not inference inputs')
    if p['components_ok'] is not True:
        raise ValueError('One or more baseline components failed')
    for field in ('v20', 'en', 'hs'):
        if not isinstance(p[field], str) or not p[field].strip():
            raise ValueError(f'Missing or malformed voter ID: {field}')
    ids = p['ids']
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) for i in ids):
        raise ValueError('Invalid candidate IDs')
    if p['v20'] not in ids or p['v20'] in p['excluded']:
        raise ValueError('Invalid/excluded original winner')
    for name in ('base', 'post'):
        if len(p[name]) != len(ids) or any(not math.isfinite(x) or x < 0 for x in p[name]):
            raise ValueError('Malformed probabilities')
        mass = sum(p[name])
        allowed_mass = sum(x for i, x in zip(ids, p[name]) if i not in p['excluded'])
        if not math.isfinite(mass) or mass <= 0 or not math.isfinite(allowed_mass) or allowed_mass <= 0:
            raise ValueError(f'Zero or invalid probability mass: {name}')
    if not math.isfinite(p['gap']):
        raise ValueError('Missing gap')
    if set(p['signals']) != set(SIGNALS):
        raise ValueError('Missing retrieval signals')
    for vals in p['signals'].values():
        if len(vals) != len(ids) or any(not math.isfinite(x) for x in vals):
            raise ValueError('Malformed retrieval signals')
    if p['sample_id'] != str(p['row'].get('sample_id')):
        raise ValueError('Row identity mismatch')


def rank(ids, score):
    return sorted(ids, key=lambda i: (-score[i], i))


def build_pool(p):
    """A maximum of 97 candidates: original winner +32 base +4*16 retrieval.

    Retrieval features promote candidates independently of the ranker's top20.
    Duplicate candidates never gain weight. Zero-similarity features add nothing.
    """
    excluded = set(p['excluded'])
    allowed = [i for i in p['ids'] if i not in excluded]
    base = dict(zip(p['ids'], p['base']))
    chosen = {p['v20'], *rank(allowed, base)[:32]}
    for name in SIGNALS:
        score = dict(zip(p['ids'], p['signals'][name]))
        chosen.update(i for i in rank(allowed, score)[:16] if score[i] > 0)
    return sorted(chosen)


def score_pairs(scorer, row, ids, catalog, beams):
    """Actual scorer.score(row, list[dict], beams); max batch 8, canonical ID order."""
    out = {}
    for start in range(0, len(ids), 8):
        part = ids[start:start+8]
        # Identity is only a mapping key, never included in neural model text.
        docs = [{k: catalog[i].get(k) for k in ROW_KEYS - {'sample_id', 'call_sign'}} for i in part]
        vals = list(scorer.score(row, docs, beams))
        if len(vals) != len(part) or any(not math.isfinite(float(x)) for x in vals):
            raise ValueError('Nonfinite/missing scorer results')
        out.update((i, float(x)) for i, x in zip(part, vals))
    return out


def run_one(p, catalog, encoder, qwen, *, enabled=True):
    validate(p)
    for field in ('v20', 'en', 'hs'):
        if p[field] not in catalog:
            raise ValueError(f'Voter ID is not in official catalog: {field}')
    for i in p['ids']:
        if i not in catalog:
            raise ValueError('Unknown catalog candidate')
    original = route(p['row'], p['gap'], p['v20'], p['en'], p['hs'])
    out = {'sample_id': p['sample_id'], 'baseline': original, 'candidate': original,
           'packet_sha256': packet_fingerprint(p),
           'v20_before': p['v20'], 'v20_after': p['v20'], 'changed': False,
           'reason': 'disabled', 'pool': [], 'shortlist': [], 'encoder': {}, 'qwen': {}}
    if not enabled:
        return out
    if not hangul(p['row']) or p['gap'] >= 4:
        out['reason'] = 'protected_route'
        return out
    if p['en'] == p['hs']:
        out['reason'] = 'fixed_majority'
        return out
    pool = build_pool(p)
    out['pool'] = pool
    if len(pool) < 2:
        out['reason'] = 'insufficient_candidates'
        return out
    enc = score_pairs(encoder, p['row'], pool, catalog, p['beams'])
    order = rank(pool, enc)
    base = dict(zip(p['ids'], p['base']))
    # 8 neural proposals +20 base leaders + original winner => at most29 Qwen pairs.
    shortlist = sorted({*order[:8], *rank(pool, base)[:20], p['v20']})
    qw = score_pairs(qwen, p['row'], shortlist, catalog, p['beams'])
    qo = rank(shortlist, qw)
    out.update(shortlist=shortlist, encoder=enc, qwen=qw)
    # A fixed one-logit separation in each independently trained view. Not a
    # calibrated confidence or a claimed safety guarantee.
    out['encoder_margin'] = enc[order[0]] - enc[order[1]]
    out['qwen_margin'] = qw[qo[0]] - qw[qo[1]]
    if order[0] != qo[0]:
        out['reason'] = 'view_disagreement'
        return out
    if out['encoder_margin'] < 1.0 or out['qwen_margin'] < 1.0:
        out['reason'] = 'weak_separation'
        return out
    winner = qo[0]
    out['v20_after'] = winner
    out['candidate'] = route(p['row'], p['gap'], winner, p['en'], p['hs'])
    out['changed'] = out['candidate'] != original
    out['reason'] = 'dual_view_rescue' if out['changed'] else 'same_final'
    out['winner_base_rank'] = rank([i for i in p['ids'] if i not in p['excluded']], base).index(winner)+1
    return out


def verify_result(p, result):
    """Post-hoc pairing check; no truth input and no model rerun."""
    validate(p)
    if result.get('sample_id') != p['sample_id'] or result.get('packet_sha256') != packet_fingerprint(p):
        raise ValueError('Result does not belong to this exact packet')
    baseline = route(p['row'], p['gap'], p['v20'], p['en'], p['hs'])
    if result.get('baseline') != baseline or result.get('v20_before') != p['v20']:
        raise ValueError('Result baseline disagrees with original router')
    after = result.get('v20_after')
    if after not in p['ids'] or after in p['excluded']:
        raise ValueError('Result chose an unavailable/excluded v20 candidate')
    expected = route(p['row'], p['gap'], after, p['en'], p['hs'])
    if result.get('candidate') != expected or result.get('changed') is not (expected != baseline):
        raise ValueError('Result prediction/change flag disagrees with router')
    eligible = hangul(p['row']) and p['gap'] < 4 and p['en'] != p['hs']
    if not eligible and (after != p['v20'] or result['candidate'] != baseline):
        raise ValueError('Result changed a protected/out-of-scope row')


def install_capture(ensemble_class, excluded_ids, emit):
    """External monkeypatch, retaining original file hashes and return arrays.

    emit receives only trace-stage fields; kit adds en, hs, gap, components_ok
    after successfully joining the three real components by sample_id.
    Call before baseline script.run; restore() afterwards, including exceptions.
    """
    original = ensemble_class.score_records

    def scalar(value):
        if hasattr(value, 'item'):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def captured(self, rows, signals, blocks):
        saved = self.baseline.score_groups
        bases = []

        def spy(current):
            result = saved(current)
            bases.extend(x.copy() for x in result)
            return result

        self.baseline.score_groups = spy
        try:
            result = original(self, rows, signals, blocks)
        finally:
            self.baseline.score_groups = saved
        if not (len(rows) == len(signals) == len(bases) == len(result)):
            raise ValueError('Trace alignment failed')
        for row, sig, base, post in zip(rows, signals, bases, result):
            ids = [str(self.r.ids[int(j)]) for j in sig['cand']]
            ok = [j for j, i in enumerate(ids) if i not in excluded_ids]
            if not ok:
                raise ValueError('Every candidate is excluded')
            # Mirror outer script masking and numpy argmax's first occurrence.
            winner = ids[max(ok, key=lambda j: float(post[j]))]
            emit({'sample_id': str(row['sample_id']),
                  'row': {k: scalar(row[k]) for k in ROW_KEYS if k in row},
                  'ids': ids, 'base': [float(x) for x in base],
                  'post': [float(x) for x in post],
                  'signals': {k: [float(x) for x in sig[k]] for k in SIGNALS},
                  'beams': list(sig.get('llm_beams', ())),
                  'excluded': sorted(set(ids) & set(excluded_ids)), 'v20': winner})
        return result

    ensemble_class.score_records = captured

    def restore():
        ensemble_class.score_records = original
    return restore
