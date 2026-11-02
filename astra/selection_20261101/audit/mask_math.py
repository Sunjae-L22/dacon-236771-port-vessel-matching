"""CPU-only independent algebra check. No competition data/model is read."""
import json
from pathlib import Path
import numpy as np

def softmax(x):
    e=np.exp(x-np.max(x,axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)

def compare(raw,weights,allowed):
    raw=np.asarray(raw,float);weights=np.asarray(weights,float);allowed=np.asarray(allowed,bool)
    p=softmax(raw);mass=p[:,allowed].sum(axis=1)
    old=(weights[:,None]*p[:,allowed]).sum(axis=0);old/=old.sum()
    conditional=softmax(raw[:,allowed])
    new=(weights[:,None]*conditional).sum(axis=0)/weights.sum()
    effective=weights*mass;effective/=effective.sum()
    reconstructed=(effective[:,None]*conditional).sum(axis=0)
    assert np.allclose(old,reconstructed,atol=1e-13)
    return dict(old=old.tolist(),new=new.tolist(),allowed_mass=mass.tolist(),
                old_effective_weights=effective.tolist(),fixed_weights=(weights/weights.sum()).tolist(),
                old_winner=int(old.argmax()),new_winner=int(new.argmax()))

def main():
    # Candidate2 is forbidden. Expert0 prefers allowed A conditionally but assigns
    # nearly all unconditioned probability to the forbidden candidate.
    raw=np.array([[2.,0.,10.],[0.,1.,-5.]])
    toy=compare(raw,[1,1],[True,True,False])
    assert toy['old_winner']!=toy['new_winner']
    rng=np.random.default_rng(11)
    for _ in range(200):
        r=rng.normal(size=(5,9))*4;w=np.array([3,3,3,1,1]);a=np.array([True]*6+[False]*3)
        reference=compare(r,w,a)
        perm=rng.permutation(9);permuted=compare(r[:,perm],w,a[perm])
        # Reorder allowed candidate arrays back to the original allowed order.
        original_ids=np.flatnonzero(a);permuted_ids=perm[a[perm]]
        restore=[list(permuted_ids).index(i) for i in original_ids]
        assert np.allclose(reference['new'],np.asarray(permuted['new'])[restore])
        # With fixed raw features, manipulating forbidden logits cannot change new.
        changed=r.copy();changed[:,~a]=rng.normal(size=(5,3))*20
        assert np.allclose(reference['new'],compare(changed,w,a)['new'])
    equal_mass=compare([[2,0,1],[0,2,1]],[3,1],[True,True,False])
    assert np.allclose(equal_mass['old'],equal_mass['new'])
    no_mask=compare(raw,[1,1],[True,True,True])
    assert np.allclose(no_mask['old'],no_mask['new'])
    # Numerical note: direct softmax over allowed logits remains well-defined
    # even when their probabilities under the full universe underflow to zero.
    safe=softmax(np.array([[0.,-1.]]));assert np.isfinite(safe).all()
    result={'tests':'PASS','random_identity_permutation_decoy_cases':200,'toy':toy,
            'interpretation':'Old is Bayes-conditioning a mixture, new is fixed-weight mixture of conditional experts. Ranking can change; which is more accurate depends on labels. No correctness gain established.',
            'invariance_scope':'Fixed raw feature scores only; production candidate-relative features may themselves change if candidates are removed.'}
    Path(__file__).with_name('MATH_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
