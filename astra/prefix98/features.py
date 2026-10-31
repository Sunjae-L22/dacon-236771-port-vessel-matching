"""Three additive short-prefix features; original candidate pool/features stay intact."""
import unicodedata
import numpy as np

FEATURE_NAMES=['ko_short_tail_prefix','ko_short_token_prefix_coverage','ko_short_tail_gain']

def nonempty_prefix(candidate,query):
    return bool(query) and bool(candidate) and candidate.startswith(query)

def query_prefix_ratio(query,candidate):
    if not query or not candidate:return 0.
    matched=0
    for q,c in zip(query,candidate):
        if q!=c:break
        matched+=1
    return matched/len(query)

def extra_features(matcher,name,candidate_indices,original_tail):
    """Read-only inputs, independent float32 extra columns, no score/candidate override.

    Dictionary and catalog state must be the exact matched incumbent context.
    Empty translated/roman strings never establish prefix evidence.
    """
    name=unicodedata.normalize('NFC',str(name or ''));n=len(candidate_indices)
    old=np.asarray(original_tail,dtype=np.float32);assert old.shape==(n,)
    out=np.zeros((n,3),dtype=np.float32)
    if not any('\uac00'<=c<='\ud7a3' for c in name) or not n:return out
    query=matcher.translate_ko(name);tokens=query['tok_projections']
    if not tokens:return out
    for j,index in enumerate(candidate_indices):
        candidate_tokens=matcher.cat_tok_e2k[int(index)]
        tail=0.
        if len(tokens)>=2 and len(candidate_tokens)>=2:
            qt,qr,qc=tokens[-1];ct,ce1,ce2,cc=candidate_tokens[-1]
            if 1<=len(qr)<=4:
                prefix=max(query_prefix_ratio(qt,ct),query_prefix_ratio(qr,ce1),query_prefix_ratio(qr,ce2))
                tail=.88*prefix
                # Same vowel-class adjustment as original build70 tail calculation.
                if qr and ct and ((qr[0] in 'aeiou')!=(ct[0] in 'aeiou')):tail*=.78
        coverage=0.
        for qt,qr,qc in tokens:
            if not (1<=len(qr)<=3):continue
            best=0.
            for ct,ce1,ce2,cc in candidate_tokens:
                if len(ct)<2:continue
                matched=(nonempty_prefix(ct,qt) or nonempty_prefix(ce1,qr) or nonempty_prefix(ce2,qr))
                if matched:
                    score=.86
                    if bool(qr and qr[0] in 'aeiou')!=bool(ct and ct[0] in 'aeiou'):score*=.85
                    best=max(best,score)
            coverage+=best
        out[j,0]=tail;out[j,1]=coverage/len(tokens)
    out[:,2]=np.maximum(out[:,0]-old,0.)
    assert np.isfinite(out).all() and np.all(out>=0.)
    return out

def append_features(base,names,matcher,name,indices):
    assert base.dtype==np.float32 and base.shape==(len(indices),len(names))
    extra=extra_features(matcher,name,indices,base[:,names.index('ko_tail_sim')])
    return np.column_stack((base,extra)).astype(np.float32,copy=False)
