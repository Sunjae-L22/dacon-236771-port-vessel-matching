from pathlib import Path
import pandas as pd,types,numpy as np
def run(test_path,catalog_path,output_path):
 x=pd.read_csv(test_path,dtype=str,keep_default_na=False);preds=[]
 ix=types.SimpleNamespace(ids=['A','B','C'])
 for row in x.to_dict('records'):
  cs=None
  if cs is not None:
   preds.append(ix.ids[cs])
   continue
  idx=[0,1,2];score=[0.,0.,0.];best=ix.ids.index(row['en']);score[best]=1.
  allowed=[best]+[i for i in idx if i!=best];row_gap=float(row['gap']);use_kc=row_gap<4
  preds.append(ix.ids[best])
 o=pd.DataFrame({'sample_id':x.sample_id,'target_vessel_id':preds,'ko_gap':x.gap})
 o.iloc[::-1].to_csv(output_path,index=False)
