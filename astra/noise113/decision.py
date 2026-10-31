"""Astra: pure per-row decisions; there is no mutable cross-row state."""
import numpy as np
MARGIN=4.0

def choose(ids, baseline_mean, allowed, features, feature_names, rule, alternate_mean=None, baseline_order=None):
    order=list(baseline_order) if baseline_order is not None else sorted(allowed,key=lambda j:(-float(baseline_mean[j]),int(ids[j])))
    assert order
    first=order[0]
    gap=float(baseline_mean[order[0]]-baseline_mean[order[1]]) if len(order)>1 else 99.0
    hard=gap<MARGIN
    tonnage_missing=bool(features[first,feature_names.index('q_gt_missing')]) if 'q_gt_missing' in feature_names else False
    chosen=first;eligible=False
    if hard:
        if rule in ('noise','noise_with_tonnage'):
            eligible=rule=='noise' or not tonnage_missing
            if eligible:
                assert alternate_mean is not None
                chosen=min(allowed,key=lambda j:(-float(alternate_mean[j]),int(ids[j])))
        elif rule=='name_top3':
            col=feature_names.index('ko_combined_score');top=order[:3]
            proposed=max(top,key=lambda j:(float(features[j,col]),-top.index(j)))
            eligible=bool(features[proposed,col]>features[first,col]+.15)
            if eligible:chosen=proposed
        elif rule=='second_name':
            if len(order)>1:
                col=feature_names.index('ko_combined_score');second=order[1]
                eligible=bool(features[second,col]>features[first,col]+.15)
                if eligible:chosen=second
        else:raise ValueError(rule)
    return chosen,dict(gap=gap,hard=hard,eligible=eligible,tonnage_missing=tonnage_missing,baseline_index=int(ids[first]),selected_index=int(ids[chosen]))
