"""Frozen two-view candidate challenge. No labels, fitting or batch state.

Use existing EncoderViewsEnsemble views, after the original source-hash checks.
GPU calls are chunked within one query, never across queries. Original top20
probabilities are retained unless both frozen models choose the same challenger.
"""
import math
import unicodedata

FIELDS = ("vessel_name", "gross_tonnage", "international_gross_tonnage", "vessel_type", "flag_name")
CAP, BASE_K, DIVERSE_K, FINAL_K, BATCH = 128, 64, 16, 8, 8
MARGIN = 1.0  # raw logits, an experimental threshold, NOT calibrated confidence


def clean(row):
    return {k: row.get(k) for k in FIELDS}


def hangul(row):
    return any("가" <= c <= "힣" for c in unicodedata.normalize("NFC", str(row.get("vessel_name", ""))))


def route(row, votes):
    """Exactly the original router for successful, complete component outputs."""
    a, b, c = (votes[k] for k in ("v20", "hsw", "p112"))
    gap = float(votes["gap110"])
    if hangul(row) and not math.isfinite(gap):
        raise ValueError("nonfinite Korean gap")
    if not hangul(row) or gap >= 4:
        return c
    if b == c or a == b or a == c:
        return b if b in (c, a) else a
    return a


def scope(row, votes):
    for k in ("v20", "hsw", "p112"):
        if not isinstance(votes.get(k), str) or not votes[k]:
            raise ValueError("failed or incomplete voter")
    route(row, votes)
    if not hangul(row):
        return "english"
    if float(votes["gap110"]) >= 4:
        return "protected"
    if votes["hsw"] == votes["p112"]:
        return "router_pinned"
    return None


def ranked(indices, values, ids):
    if any(not math.isfinite(float(values[i])) for i in indices):
        raise ValueError("nonfinite score")
    return sorted(indices, key=lambda i: (-float(values[i]), str(ids[i])))


def select_pool(ids, base, signals, incumbent, excluded=()):
    if len(ids) != len(set(ids)) or len(ids) != len(base):
        raise ValueError("duplicate IDs or score shape mismatch")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in base):
        raise ValueError("invalid baseline probabilities")
    if sum(float(v) for v in base) <= 0:
        raise ValueError("zero baseline probability mass")
    excluded = set(excluded)
    valid = [i for i, x in enumerate(ids) if x not in excluded and float(base[i]) > 0]
    if incumbent not in ids or ids.index(incumbent) not in valid:
        raise ValueError("incumbent missing/excluded: fail closed")
    order = [ids.index(incumbent)] + ranked(valid, base, ids)[:BASE_K]
    for field in ("sim_llm", "sim_key", "sim_compact", "sim_key_word"):
        values = signals[field]
        if len(values) != len(ids):
            raise ValueError("signal shape mismatch")
        order += ranked(valid, values, ids)[:DIVERSE_K]
    selected = list(dict.fromkeys(order))[:CAP]
    return sorted(selected, key=lambda i: ids[i])  # deterministic GPU grouping


def score_chunks(scorer, row, records, beams):
    out = []
    for start in range(0, len(records), BATCH):
        chunk = records[start:start+BATCH]
        values = [float(v) for v in scorer.score(clean(row), [clean(c) for c in chunk], beams)]
        if len(values) != len(chunk) or any(not math.isfinite(v) for v in values):
            raise ValueError("failed/nonfinite neural scorer")
        out.extend(values)
    return out


def challenge(row, ids, records, base, signals, votes, encoder, qwen, excluded=(), enabled=True):
    """Return (v20_id, trace). records/ids/base/signals are aligned full-pool arrays.

    Only the five whitelisted feature fields reach either scorer. Truth, sample_id,
    target_vessel_id, and catalog IDs cannot be model features. Scorer exceptions
    intentionally propagate; the runner must abort, never convert failure to HOLD.
    """
    before = route(row, votes)
    trace = {"baseline_v20": votes["v20"], "baseline_final": before, "votes": dict(votes)}
    reason = scope(row, votes)
    if reason or not enabled:
        trace.update(reason=reason or "disabled", candidate_v20=votes["v20"], candidate_final=before)
        return votes["v20"], trace
    if len(records) != len(ids):
        raise ValueError("catalog alignment mismatch")
    pool = select_pool(ids, base, signals, votes["v20"], excluded)
    inc = ids.index(votes["v20"])
    es = score_chunks(encoder, row, [records[i] for i in pool], signals.get("llm_beams", ()))
    enc = dict(zip(pool, es))
    erank = sorted(pool, key=lambda i: (-enc[i], ids[i]))
    finalist = sorted(set(erank[:FINAL_K] + [inc]), key=lambda i: ids[i])
    qs = score_chunks(qwen, row, [records[i] for i in finalist], signals.get("llm_beams", ()))
    q = dict(zip(finalist, qs))
    qrank = sorted(finalist, key=lambda i: (-q[i], ids[i]))
    winner = erank[0]
    reason = "view_disagreement"
    chosen = inc
    if winner == inc:
        reason = "incumbent_wins"
    elif qrank[0] == winner:
        # Runner-up margins avoid choosing between near-tied alternatives, not
        # merely comparing a strong challenger to an especially weak incumbent.
        em = enc[winner] - enc[erank[1]]
        qm = q[winner] - q[qrank[1]]
        if em >= MARGIN and qm >= MARGIN:
            chosen, reason = winner, "two_view_challenge"
        else:
            reason = "insufficient_margin"
    changed_votes = dict(votes, v20=ids[chosen])
    # Original v20 uses numpy's stable sort on the ORIGINAL pool positions.
    # Canonical ID tie-breaking belongs only to our experimental admission.
    top20 = sorted(range(len(ids)), key=lambda i: -float(base[i]))[:20]
    top_sum = sum(float(base[i]) for i in top20)
    conditional_gap = ((float(base[top20[0]]) - float(base[top20[1]]))/top_sum
                       if len(top20) > 1 else 1.0)
    trace.update(reason=reason, candidate_v20=ids[chosen], candidate_final=route(row, changed_votes),
                 pool_ids=[ids[i] for i in pool], finalist_ids=[ids[i] for i in finalist],
                 encoder_logits={ids[i]: enc[i] for i in pool}, qwen_logits={ids[i]: q[i] for i in finalist},
                 base_top20=[ids[i] for i in top20], base_top20_cond_gap=conditional_gap,
                 original_neural_skipped=conditional_gap > .25/.75+1e-6,
                 full_pool_ids=list(ids))
    return ids[chosen], trace


def install_instance(model, votes_by_id, excluded, emit, enabled=True):
    """Exact external integration hook: call after EncoderViewsEnsemble.load.

    May install before script.py sets masked_score_groups: the wrapper accesses
    model.baseline at call time, preserving train exclusion + KO_W=4 processing.
    votes_by_id is immutable first-pass component output keyed by sample_id.
    Only catalog IDs, original model objects, and rows/signals/blocks are consumed.
    """
    import numpy as np
    original = model.score_records
    views = {view.name: view for view, _ in model.views}
    encoder, qwen = views["encoder:phonetic"], views["qwen:evidence"]

    def wrapped(rows, signals, blocks):
        output = original(rows, signals, blocks)
        bases = model.baseline.score_groups(blocks)
        if not (len(rows) == len(signals) == len(blocks) == len(output) == len(bases)):
            raise ValueError("row alignment mismatch")
        result = []
        for row, sig, local, base in zip(rows, signals, output, bases):
            sid = str(row["sample_id"])
            votes = votes_by_id[sid]
            ids = [str(model.r.ids[i]) for i in sig["cand"]]
            # Match script.py's post-neural exclusion, including its all-bad
            # exception. Neural softmax can assign positive mass to excluded
            # zero-base candidates, so raw local argmax is not the v20 output.
            bad = np.asarray([x in excluded for x in ids], dtype=bool)
            replay = np.asarray(local, dtype=np.float64)
            if bad.any() and not bad.all():
                replay = np.where(bad, -np.inf, replay)
            actual = ids[int(np.argmax(replay))]
            if actual != votes["v20"]:
                raise ValueError("unchanged baseline replay mismatch: " + sid)
            records = [model.r.catalog.iloc[i].to_dict() for i in sig["cand"]]
            chosen, trace = challenge(row, ids, records, base, sig, votes, encoder, qwen, excluded, enabled)
            trace["sample_id"] = sid
            emit(trace)
            if chosen == actual:
                result.append(local)
            else:
                new = np.zeros(len(local), dtype=np.float64)
                new[ids.index(chosen)] = 1.0
                result.append(new)
        return result
    model.score_records = wrapped
    return model
