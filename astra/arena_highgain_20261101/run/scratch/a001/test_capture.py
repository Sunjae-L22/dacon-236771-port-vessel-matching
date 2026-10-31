"""Execute real encoder_views.py control flow with mocked neural weights, no GPU claims."""
import sys
sys.dont_write_bytecode=True
import importlib.util
import json
import types
from pathlib import Path
import numpy as np
from neural_challenge import install_capture

ROOT=Path('/Users/sunjae/Desktop/항만/astra/arena_20261101/baseline/model/ko/model/src')
def mod(name, **attrs):
    m=types.ModuleType(name)
    for k,v in attrs.items():setattr(m,k,v)
    sys.modules[name]=m
mod('src');mod('src.common',is_hangul=lambda x:True)
mod('src.encoder_pair',EncoderPairScorer=object)
mod('src.encoder_phonetics',EncoderPhoneticScorer=object)
mod('src.ensemble',Ensemble=object)
mod('src.features',FEATURE_NAMES=[])
def mix(base,scores,weights,weight):
    s=np.exp(scores[0]-max(scores[0]));return .75*base+.25*s/s.sum()
mod('src.pair_views',PairViewScorer=object,mix_views=mix)
spec=importlib.util.spec_from_file_location('real_encoder_views',ROOT/'encoder_views.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Baseline:
    def score_groups(self,blocks):return [np.array([.50,.30,.20]) for _ in blocks]
class FakeCatalog:
    @property
    def iloc(self):return self
    def __getitem__(self,i):return self
    def to_dict(self):return {'vessel_name':'candidate'}
class View:
    def score(self,row,candidates,beams):return np.array([0.,4.,-1.])
r=types.SimpleNamespace(ids=np.array(['A','B','C']),catalog=FakeCatalog())
rows=[{'sample_id':'s1','vessel_name':'알파','target_vessel_id':'C'}]
signals=[{'cand':np.array([0,1,2]),'sim_key':np.array([1,.5,.1]),'sim_llm':np.array([.9,.4,.1]),'llm_beams':['Alpha']}]
def make():return m.EncoderViewsEnsemble(Baseline(),[(View(),1)],r)
def production_mask(n):
    old=n.baseline.score_groups
    def masked(blocks):return [x*np.array([1.,1.1,1.])/(x*np.array([1.,1.1,1.])).sum() for x in old(blocks)]
    n.baseline.score_groups=masked
plain=make();production_mask(plain);expected=plain.score_records(rows,signals,[None])
trace=[];wrapped=make();install_capture(wrapped,trace.append);production_mask(wrapped)
actual=wrapped.score_records(rows,signals,[None])
assert np.array_equal(expected[0],actual[0])
assert trace[0]['pool'][0]['raw']==.5
assert trace[0]['pool'][0]['adjusted']!=.5
assert 'target_vessel_id' not in trace[0]['row']
assert np.array_equal(actual[0],wrapped.score_records(rows,signals,[None])[0])
print(json.dumps({'real_source_control_flow':True,'mocked_neural_weights':True,'bit_exact_capture_replay':True,'raw_and_adjusted_distinct':True,'repeated_call_equal':True}))
