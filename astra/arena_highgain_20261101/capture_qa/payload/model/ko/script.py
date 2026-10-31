from pathlib import Path
import pandas as pd,os
def run(test_path,catalog_path,output_path):
 tag=Path(__file__).parent.name
 if os.getenv('FAIL_KO')=='1' and tag=='ko':raise SystemExit(7)
 x=pd.read_csv(test_path,dtype=str,keep_default_na=False)
 o=pd.DataFrame({'sample_id':x.sample_id,'target_vessel_id':x[tag]})
 if tag=='en':o['ko_gap']=x.gap
 o.iloc[::-1].to_csv(output_path,index=False)
