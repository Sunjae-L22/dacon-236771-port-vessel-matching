"""POSTHOC ONLY: run after frozen inference artifacts are closed. Never imported by inference."""
def diagnosis(packet, result, truth):
    pool={p['id'] for p in packet['pool']}
    top20=set(packet['baseline_top20'])
    baseline=result['baseline']
    return {
        'truth_in_original_pool': truth in pool,
        'truth_rank_below20': truth in pool and truth not in top20,
        'truth_in_challenge_shortlist': truth in set(result.get('shortlist',[])),
        'baseline_neural_skipped': packet['baseline_neural_skipped'],
        'baseline_v20_correct': packet['votes']['ko']==truth,
        'baseline_route_overrode_correct_v20': packet['votes']['ko']==truth and baseline!=truth,
        'baseline_neural_misranking': truth in top20 and not packet['baseline_neural_skipped'] and packet['votes']['ko']!=truth,
        'neural_misranking_note':'final mixed v20 ranking; inspect baseline per-view logits to distinguish model error from convex-mixture suppression',
        'rescue': baseline!=truth and result['prediction']==truth,
        'harm': baseline==truth and result['prediction']!=truth,
    }
