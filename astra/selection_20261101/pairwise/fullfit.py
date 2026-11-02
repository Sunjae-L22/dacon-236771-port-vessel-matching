"""Fit the already-frozen pair recipe on both existing OOF development folds."""
import sys,pickle,json,time
sys.dont_write_bytecode=True
import run as R
R.checktime()
rows=[]
for f in (2,4):rows.extend(pickle.loads((R.HERE/f'fold{f}_features.pkl').read_bytes()))
t=time.monotonic();m,ships=R.fit(rows,'2plus4')
(R.HERE/'FULLFIT.json').write_text(json.dumps({'source_folds':[2,4],'feature_width':187,'trees':m.num_trees(),'vessels':len(ships),'seconds':time.monotonic()-t,'threshold':.95,'parameter_changes':False,'type':'Combined OOF features, not in-sample full-fit112 training features','model':'verifier_fitfold2plus4.txt'},indent=2)+'\n')
