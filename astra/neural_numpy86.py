"""Author: Astra. NumPy inference for the trained character/jamo encoder."""
import unicodedata, zlib
from functools import lru_cache
import numpy as np
from scipy.special import erf
import opus_lib as B
N=32768

@lru_cache(maxsize=4096)
def features(name):
    s=' '.join(unicodedata.normalize('NFKC',str(name)).lower().split())
    roman=B.romanize(B.ko_clean(s)) if B.is_hangul(s) else B.acronym_expand(s)
    projections=[('c',unicodedata.normalize('NFD',s)),('r',roman),('p',B.phon_key(roman,ko=B.is_hangul(s))),('v',B.vowel_key(roman))]
    ids=[]
    for tag,text in projections:
        words=text.split(); strings=words+[''.join(words)]
        for word in strings:
            word='^'+word+'$'
            for n in (1,2,3,4):
                for i in range(len(word)-n+1): ids.append(zlib.crc32((tag+word[i:i+n]).encode())%N)
    return tuple(ids or [0])

class Encoder:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as f: self.w={k:f[k] for k in f.files}
    def __call__(self,names):
        # Always process one string at a time: no query-batch dependent arithmetic.
        out=[]
        for name in names:
            v=self.w['emb.weight'][list(features(name))].mean(axis=0)
            h=v@self.w['proj.0.weight'].T+self.w['proj.0.bias']
            h=.5*h*(1+erf(h/np.sqrt(np.float32(2))))
            v=v+h@self.w['proj.2.weight'].T+self.w['proj.2.bias']
            out.append(v/max(float(np.linalg.norm(v)),1e-12))
        return np.stack(out).astype(np.float32)

def vectors(model,names): return model(names)
