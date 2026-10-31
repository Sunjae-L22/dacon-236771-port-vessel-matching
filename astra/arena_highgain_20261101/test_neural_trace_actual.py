"""Actual frozen score_records source; tiny CPU scorer doubles, no GPU claim."""
from pathlib import Path
import sys,types,importlib.util,json
import numpy as np,pandas as pd
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'package'))
from trace_hooks import attach_neural_trace
src=types.ModuleType('src');src.__path__=[];sys.modules['src']=src
for mod,attrs in {'common':{'is_hangul':lambda s:any('가'<=c<='힣' for c in str(s))},'encoder_pair':{'EncoderPairScorer':object},'encoder_phonetics':{'EncoderPhoneticScorer':object},'ensemble':{'Ensemble':object},'features':{'FEATURE_NAMES':[]},'pair_views':{'PairViewScorer':object}}.items():
 m=types.ModuleType('src.'+mod)
 for k,v in attrs.items():setattr(m,k,v)
 sys.modules[m.__name__]=m
def mix(bp,scores,weights,w):
 probs=[np.exp(s-np.max(s))/np.exp(s-np.max(s)).sum() for s in scores]
 return (1-w)*bp+w*np.average(probs,weights=weights,axis=0)
sys.modules['src.pair_views'].mix_views=mix
p=R.parent/'arena_20261101/baseline/model/ko/model/src/encoder_views.py';sp=importlib.util.spec_from_file_location('src.encoder_views',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
class Scorer:
 def __init__(self,name):self.name=name;self.calls=0
 def score(self,row,candidates,beams):self.calls+=1;return np.array([1.,2.,0.])[:len(candidates)]
scorers=[Scorer('qwen:evidence'),Scorer('encoder:phonetic')]
r=types.SimpleNamespace(ids=np.array(['A','B','C']),catalog=pd.DataFrame({'vessel_id':['A','B','C']}))
b=types.SimpleNamespace(score_groups=lambda blocks:[np.asarray(x).copy() for x in blocks]);model=m.EncoderViewsEnsemble(b,[(s,1) for s in scorers],r)
rows=[{'sample_id':'active','vessel_name':'한글'},{'sample_id':'skip','vessel_name':'한글'},{'sample_id':'one','vessel_name':'한글'}]
sigs=[{'cand':np.array([0,1,2])},{'cand':np.array([0,1,2])},{'cand':np.array([0])}];blocks=[np.array([.4,.35,.25]),np.array([.9,.05,.05]),np.array([1.])]
import neural_rescue as nr
for sig in sigs:
 for key in nr.SIGNALS:sig[key]=np.zeros(len(sig['cand']))
expected=model.score_records(rows,sigs,blocks);before=[s.calls for s in scorers]
out=R/'capture_qa/actual_class_trace.jsonl';out.unlink(missing_ok=True);attach_neural_trace(out)
packets=[];nr.install_capture(m.EncoderViewsEnsemble,set(),packets.append)
got=model.score_records(rows,sigs,blocks);assert all(np.array_equal(a,b) for a,b in zip(expected,got));assert [s.calls-before[i] for i,s in enumerate(scorers)]==[1,1]
for p,s in zip(packets,got):
 assert p['v20']==p['ids'][int(np.argmax(s))]
 p.update(en='A',hs='B',gap=1.,components_ok=True);nr.validate(p)
rev=model.score_records(rows[::-1],sigs[::-1],blocks[::-1]);assert all(np.array_equal(a,b) for a,b in zip(got,rev[::-1]))
for i in range(3):assert np.array_equal(got[i],model.score_records(rows[i:i+1],sigs[i:i+1],blocks[i:i+1])[0])
tr=[json.loads(x) for x in out.read_text().splitlines()][:3];assert len(tr[0]['views'])==2 and tr[1]['neural_skipped'] and tr[2]['neural_skipped']
print('PASS actual frozen score_records: active/skip/singleton, exact output equality, no extra scorer calls, reverse/subset invariance')
