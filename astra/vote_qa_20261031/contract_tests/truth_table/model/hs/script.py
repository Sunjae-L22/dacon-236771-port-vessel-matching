from pathlib import Path
import pandas as pd,time,os
def run(test_path,catalog_path,output_path):
 p=Path(__file__).parent; tag=p.name; root=p.parents[1]
 (root/(tag+'_started')).write_text(str(time.time()))
 mode=os.environ.get('STUB_MODE','normal')
 if mode=='fail_en' and tag=='en': raise SystemExit(7)
 if mode=='fail_en': time.sleep(1.5)
 else: time.sleep(.15)
 (root/(tag+'_after_delay')).write_text(str(time.time()))
 d=pd.read_csv(test_path,dtype=str,keep_default_na=False)
 o=pd.DataFrame({'sample_id':d.sample_id,'target_vessel_id':d[tag]})
 if tag=='en': o['ko_gap']=d.gap
 if mode=='missing_en' and tag=='en': o=o.iloc[1:]
 if mode=='missing_ko' and tag=='ko': o=o.iloc[1:]
 if mode=='duplicate_en' and tag=='en': o=pd.concat([o,o.iloc[:1]])
 if mode=='missing_gap' and tag=='en': o=o.drop(columns='ko_gap')
 o.iloc[::-1].to_csv(output_path,index=False)
 (root/(tag+'_done')).write_text(str(time.time()))
