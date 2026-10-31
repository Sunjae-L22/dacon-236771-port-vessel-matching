"""Astra: fit-only Korean lexical denoising. No catalog Korean generation/table."""
import re,unicodedata
from collections import defaultdict
class Restorer:
    def __init__(self,matcher,max_edits=2,max_views=3):
        self.max_edits=max_edits;self.max_views=max_views
        vocab={k for k in matcher.exact_word_dict if re.fullmatch('[가-힣]+',k) and 2<=len(k)<=32}
        vocab.update(k for k,v in matcher.learned_subwords if re.fullmatch('[가-힣]+',k) and 2<=len(k)<=32)
        self.exact=vocab;self.rev=defaultdict(set)
        for word in sorted(vocab):
            if len(word)<4:continue
            for i in range(len(word)):
                self.rev[word[:i]+word[i+1:]].add(word)
                self.rev[word[:i+1]+word[i]+word[i+1:]].add(word)
            for i in range(len(word)-1):
                if word[i]!=word[i+1]:self.rev[word[:i]+word[i+1]+word[i]+word[i+2:]].add(word)
        self.maxlen=min(33,max(map(len,vocab),default=0)+1)
    def restore(self,name):
        name=unicodedata.normalize('NFC',str(name));n=len(name)
        # State: input offset, total edit count -> distinct complete outputs so far.
        states={(0,0):{''}}
        for i in range(n):
            for e in range(self.max_edits+1):
                prefixes=states.get((i,e),())
                if not prefixes:continue
                transitions=[];c=name[i]
                if not ('가'<=c<='힣') or (c=='호' and i==n-1):transitions.append((i+1,0,c))
                for j in range(i+1,min(n,i+self.maxlen)+1):
                    part=name[i:j]
                    if not ('가'<=part[-1]<='힣'):break
                    if part in self.exact:transitions.append((j,0,part))
                    if e<self.max_edits and part in self.rev:
                        transitions.extend((j,1,w) for w in sorted(self.rev[part]))
                for j,cost,word in transitions:
                    ne=e+cost
                    if ne>self.max_edits:continue
                    bucket=states.setdefault((j,ne),set())
                    # Cap ambiguous states above the decision limit. Stable lexical ordering.
                    for pre in sorted(prefixes):
                        if len(bucket)>=16:break
                        bucket.add(pre+word)
        for e in range(self.max_edits+1):
            views=sorted(states.get((n,e),set()))
            if views:
                if e==0 or len(views)>self.max_views:return dict(views=[],edits=e,reason='already_covered' if e==0 else 'ambiguous')
                return dict(views=views,edits=e,reason='restored')
        return dict(views=[],edits=None,reason='out_of_vocabulary')
