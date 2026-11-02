"""Pure ID-level tournament decision; no model calls, labels or shared state."""
def decide(row,cache):
 out={'prediction':row['cvote'],'condorcet_winner':None,'wins':{},'decisive_pairs':0,'draw_pairs':0,'invalid_pairs':0,'order_disagreement_pairs':0}
 if not row['hard']:return out
 pool=list(row['pool']);wins={v:0 for v in pool}
 for p in row['comparisons']:
  ab=cache[p['ab']];ba=cache[p['ba']]
  ca=ab.get('choice');cb=ba.get('choice')
  if ca not in ('A','B') or cb not in ('A','B'):
   out['invalid_pairs']+=1;out['draw_pairs']+=1;continue
  wa=p['a'] if ca=='A' else p['b'];wb=p['b'] if cb=='A' else p['a']
  if wa!=wb:out['order_disagreement_pairs']+=1;out['draw_pairs']+=1;continue
  wins[wa]+=1;out['decisive_pairs']+=1
 winners=[v for v in pool if wins[v]==len(pool)-1]
 assert len(winners)<=1,'Multiple strict Condorcet winners indicate a graph bug'
 if len(winners)==1:out['prediction']=out['condorcet_winner']=winners[0]
 out['wins']=wins
 return out
