"""Exhaustive small graph audit of the actual pure decision function; no GPU."""
import itertools,json,sys,copy,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from decision import decide

def make(pool,states):
 cache={};comparisons=[];edges=set()
 for (a,b),state in zip(itertools.combinations(pool,2),states):
  ab=f'{a}/{b}';ba=f'{b}/{a}';comparisons.append({'a':a,'b':b,'ab':ab,'ba':ba})
  choices={-1:('B','A'),0:('A','A'),1:('A','B')}[state]
  cache[ab]={'choice':choices[0]};cache[ba]={'choice':choices[1]}
  if state:edges.add((a,b) if state==1 else (b,a))
 return {'cvote':pool[0],'hard':True,'pool':list(pool),'comparisons':comparisons},cache,edges
counts={'graphs':0,'pool_permutations':0,'comparison_reversals':0,'format_cases':0,'protected_cases':0}
for n in (2,3,4):
 pool=[f'V{i}' for i in range(n)]
 for states in itertools.product((-1,0,1),repeat=n*(n-1)//2):
  row,cache,edges=make(pool,states);before=copy.deepcopy((row,cache));out=decide(row,cache)
  winners=[a for a in pool if all((a,b) in edges for b in pool if b!=a)]
  assert len(winners)<=1;expected=winners[0] if winners else row['cvote']
  assert out['prediction']==expected and out['condorcet_winner']==(winners[0] if winners else None)
  assert out['decisive_pairs']==len(edges)
  for perm in itertools.permutations(pool):
   alt={**row,'pool':list(perm)};got=decide(alt,cache);assert got==out;counts['pool_permutations']+=1
  swapped=[]
  for p in row['comparisons'][::-1]:swapped.append({'a':p['b'],'b':p['a'],'ab':p['ba'],'ba':p['ab']})
  assert decide({**row,'comparisons':swapped},cache)==out;counts['comparison_reversals']+=1
  assert (row,cache)==before
  assert decide({**row,'hard':False},{})['prediction']==row['cvote'];counts['protected_cases']+=1;counts['graphs']+=1
# Every possible valid/invalid AB/BA combination on2candidates.
for ca,cb in itertools.product(('A','B',None,'X'),repeat=2):
 row,cache,edges=make(['X','Y'],[0]);cache['X/Y']['choice']=ca;cache['Y/X']['choice']=cb;out=decide(row,cache)
 winner='X' if (ca,cb)==('A','B') else 'Y' if (ca,cb)==('B','A') else None
 assert out['condorcet_winner']==winner;counts['format_cases']+=1
# Same name prompt makes AB/BA cachekey identical; deterministic letter gives disagreement.
row={'cvote':'X','hard':True,'pool':['X','Y'],'comparisons':[{'a':'X','b':'Y','ab':'same','ba':'same'}]}
for letter in ('A','B',None):
 out=decide(row,{'same':{'choice':letter}});assert out['condorcet_winner'] is None and out['prediction']=='X'
# Missing cache entries fail explicitly, not by borrowing a prior row's variables.
try:decide(row,{})
except KeyError:pass
else:raise AssertionError('Expected explicit incomplete-cache failure')
# No persistent state across unrelated successive rows.
r1,c1,_=make(['U','V','W'],[1,1,0]);r2,c2,_=make(['K','L'],[-1]);a=decide(r1,c1);b=decide(r2,c2);assert decide(r1,c1)==a and decide(r2,c2)==b
result={'pass':True,'counts':counts,'mutation_detected':False,'same_name_abstention':True,'missing_cache_behavior':'explicit KeyError, entire evaluation stops','source_sha256':hashlib.sha256((HERE.parent/'decision.py').read_bytes()).hexdigest(),'scope':'Pure decision implementation, exhaustive2/3/4-node directed-or-tied graphs; no neural score guarantee'}
(HERE/'GRAPH_PROPERTIES.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
