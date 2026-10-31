"""Astra: optional per-instance, bounded pure-computation cache for fixed KoMatcher.

Only attach after all dictionary fitting/catalog setup is finished. Detach before
any mutation of the matcher's dictionary or catalog arrays. This cache stores no
row metadata and changes neither fitting data nor any matching calculation.
"""
from collections.abc import Mapping
from functools import lru_cache
from types import MethodType, MappingProxyType

def freeze(value):
    if isinstance(value,Mapping):return tuple(sorted((key,freeze(item)) for key,item in value.items()))
    if isinstance(value,(tuple,list)):return tuple(freeze(item) for item in value)
    if isinstance(value,(str,int,float,bool,type(None))):return value
    raise TypeError(f'Unsupported mutable qinfo field: {type(value).__name__}')

class FrozenQInfo(Mapping):
    """Full immutable qinfo fingerprint, with precomputed hash for repeated scoring."""
    __slots__=('_fingerprint','_hash','_mapping')
    def __init__(self,info):
        object.__setattr__(self,'_fingerprint',freeze(info))
        object.__setattr__(self,'_hash',hash(self._fingerprint))
        object.__setattr__(self,'_mapping',MappingProxyType(dict(self._fingerprint)))
    def __setattr__(self,key,value):raise TypeError('FrozenQInfo is immutable')
    def __getitem__(self,key):return self._mapping[key]
    def __iter__(self):return iter(self._mapping)
    def __len__(self):return len(self._mapping)
    def __hash__(self):return self._hash
    def __eq__(self,other):
        if isinstance(other,FrozenQInfo):return self._fingerprint==other._fingerprint
        if isinstance(other,Mapping):return self._fingerprint==freeze(other)
        return NotImplemented

class InstanceMemo:
    """Attach cache closures to exactly one already-fitted matcher instance."""
    def __init__(self,matcher,translation_limit=2048,score_limit=50000):
        self.matcher=matcher;self.original_translate=matcher.translate_ko;self.original_score=matcher._score_from_qinfo
        self.keys=tuple(matcher.FEATURE_NAMES);self.attached=False
        @lru_cache(maxsize=translation_limit)
        def translated(name):return FrozenQInfo(self.original_translate(name))
        @lru_cache(maxsize=score_limit)
        def scored(info,idx):
            result=self.original_score(info,idx)
            assert set(result)==set(self.keys)
            return tuple(result[key] for key in self.keys)
        self.translated=translated;self.scored=scored
        def translate_call(instance,name):return self.translated(name)
        def score_call(instance,info,idx):
            frozen=info if isinstance(info,FrozenQInfo) else FrozenQInfo(info)
            return dict(zip(self.keys,self.scored(frozen,int(idx))))
        self.translate_call=MethodType(translate_call,matcher);self.score_call=MethodType(score_call,matcher)
    def attach(self):
        self.matcher.translate_ko=self.translate_call;self.matcher._score_from_qinfo=self.score_call;self.attached=True
    def detach(self,clear=True):
        self.matcher.translate_ko=self.original_translate;self.matcher._score_from_qinfo=self.original_score;self.attached=False
        if clear:self.clear()
    def clear(self):self.translated.cache_clear();self.scored.cache_clear()
    def stats(self):return {'translations':self.translated.cache_info()._asdict(),'scores':self.scored.cache_info()._asdict()}
