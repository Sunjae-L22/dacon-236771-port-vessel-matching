"""Explicitly post-ablation-selected single-column-removal follow-up."""
import sys,json,time
sys.dont_write_bytecode=True
import ablation as A
t=time.monotonic();A.R.checktime()
m,ships=A.fit(A.FOLDS[2]+A.FOLDS[4],'no_score_all19','2plus4',False,False)
(A.HERE/'FOLLOWUP_FULLFIT.json').write_text(json.dumps({'cell':'no_score_all19','selected_after_ablation':True,'original_primary_rejected':True,'trees':m.num_trees(),'feature_width':m.num_feature(),'vessels':len(ships),'threshold':.95,'parameter_search':False,'seconds':time.monotonic()-t,'model':'ablation/no_score_all19/fit2plus4.txt'},indent=2)+'\n')
