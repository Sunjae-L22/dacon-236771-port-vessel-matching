# author: opus  — usage: python3 shared/src/eval_ko.py full|nodict
import sys; sys.path.insert(0,'/Users/sunjae/Desktop/항만/shared/src')
import ko_translit_argon as A
_DICT = list(A._UNIVERSAL_KO_SUBWORDS)  # opus_lib clears it on import
import cv, opus_lib as L, numpy as np
tr,ca=cv.load(); f=cv.folds(tr); ix=L.Index(ca,None)
mode=sys.argv[1]
A._UNIVERSAL_KO_SUBWORDS = [] if mode=='nodict' else _DICT
hits={1:[],5:[]}
for k in range(5):
    km=A.KoMatcher(ca, tr[f!=k])
    va=tr[(f==k)&tr.vessel_name.map(L.is_hangul)]
    for r in va.itertuples():
        t=ix.id2i[r.target_vessel_id]; pool=set()
        for v in {L.positive(r.gross_tonnage),L.positive(r.international_gross_tonnage)}:
            if np.isfinite(v):
                lo,hi=np.searchsorted(ix.gt_sorted,[v*0.88,v*1.12]); pool.update(ix.gt_sorted_idx[lo:hi].tolist())
        pool=list(pool); s=km.score_batch(r.vessel_name,pool)[:,-1]
        order=[pool[i] for i in np.argsort(-s,kind='stable')]
        for kk in hits: hits[kk].append(t in order[:kk])
print(mode,{k:round(np.mean(v),4) for k,v in hits.items()}, len(hits[1]))
