"""Offline evaluator ONLY. Labels never imported by challenge.py."""
def classify(trace, truth):
    if trace['baseline_final'] == truth:
        stage = 'baseline_correct'
    elif trace['baseline_v20'] == truth:
        stage = 'route_override'
    elif 'full_pool_ids' not in trace:
        stage = 'out_of_scope_needs_baseline_trace'
    elif truth not in trace['full_pool_ids']:
        stage = 'candidate_absent'
    elif truth not in trace['base_top20']:
        stage = 'rank_below20'
    elif trace['original_neural_skipped']:
        stage = 'neural_skipped_confidence'
    else:
        stage = 'neural_misranking'
    return dict(stage=stage,
                challenger_admitted=truth in trace.get('pool_ids', []),
                qwen_admitted=truth in trace.get('finalist_ids', []),
                rescued=trace['baseline_final'] != truth and trace['candidate_final'] == truth,
                harmed=trace['baseline_final'] == truth and trace['candidate_final'] != truth)
