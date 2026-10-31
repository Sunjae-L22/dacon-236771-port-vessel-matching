"""Real Popen stubs test the unchanged router including error behavior."""
from pathlib import Path
import sys,json,subprocess,os,time,itertools,importlib.util,shutil
import pandas as pd
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;DEST=HERE/'contract_tests';DEST.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location('builder',HERE/'sources/build_vote.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
STUB='''from pathlib import Path
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
'''

def run_case(name,df,mode='normal'):
    p=DEST/name
    if p.exists():shutil.rmtree(p)
    p.mkdir()
    (p/'script.py').write_text(m.ROUTER.format(gap=4.,tie='ko'))
    for tag in ['en','ko','hs']:
        x=p/'model'/tag;x.mkdir(parents=True,exist_ok=True);(x/'script.py').write_text(STUB)
    (p/'data').mkdir(exist_ok=True);(p/'tmp').mkdir(exist_ok=True)
    df.to_csv(p/'data/test.csv',index=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',STUB_MODE=mode,TMPDIR=str(p/'tmp'))
    t=time.monotonic();v=subprocess.run([sys.executable,'script.py'],cwd=p,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (p/'stdout.log').write_text(v.stdout);(p/'stderr.log').write_text(v.stderr)
    o=pd.read_csv(p/'output/submission.csv',dtype=str,keep_default_na=False) if (p/'output/submission.csv').exists() else None
    return p,v,o,time.monotonic()-t

def oracle(r):
    import unicodedata
    if not any('가'<=c<='힣' for c in unicodedata.normalize('NFC',r.vessel_name)):return r.en
    if float(r.gap)>=4:return r.en
    vals=[r.ko,r.hs,r.en]
    return next((v for v in vals if vals.count(v)>=2),r.ko)

rows=[]
for h,g,(a,b,c) in itertools.product(['한글','ENGLISH','한글'],['3.999','4','4.001','nan'],itertools.product('ABC',repeat=3)):
    rows.append(dict(sample_id=f'{len(rows):05}',vessel_name=h,gap=g,en=c,ko=a,hs=b))
df=pd.DataFrame(rows).sample(frac=1,random_state=1701)
p,v,o,secs=run_case('truth_table',df);assert v.returncode==0,v.stderr
assert o.sample_id.tolist()==df.sample_id.tolist()
assert o.target_vessel_id.tolist()==[oracle(r) for r in df.itertuples()]
starts=[float((p/f'{t}_started').read_text()) for t in ['en','ko','hs']]
ends=[float((p/f'{t}_done').read_text()) for t in ['en','ko','hs']]
assert max(starts)<min(ends)
sub=df.sample(n=70,random_state=17).iloc[::-1];_,v,o,_=run_case('shuffled_subset',sub);assert v.returncode==0
assert o.target_vessel_id.tolist()==[oracle(r) for r in sub.itertuples()]
eng=df[df.vessel_name=='ENGLISH'].iloc[:5];p,v,o,_=run_case('all_english',eng);assert v.returncode==0
assert not (p/'ko_started').exists() and not (p/'hs_started').exists()
_,v,o,_=run_case('empty',df.iloc[:0]);empty_fails=v.returncode!=0
errors={}
probe=pd.DataFrame([dict(sample_id='00001',vessel_name='ENGLISH',gap='nan',en='C',ko='A',hs='B'),dict(sample_id='00002',vessel_name='한글',gap='2',en='C',ko='A',hs='B'),dict(sample_id='00003',vessel_name='다른이름',gap='2',en='C',ko='A',hs='B')])
for mode in ['fail_en','missing_en','missing_ko','duplicate_en','missing_gap']:
    test=probe.iloc[::-1] if mode=='missing_ko' else probe
    p,v,o,elapsed=run_case(mode,test,mode)
    errors[mode]=dict(returncode=v.returncode,output_written=o is not None,output=[] if o is None else o.to_dict('records'),stderr_tail=v.stderr[-700:])
    if mode=='fail_en':
        starts={t:float((p/f'{t}_started').read_text()) for t in ['en','ko','hs']}
        ends={t:float((p/f'{t}_after_delay').read_text()) for t in ['ko','hs']}
        errors[mode]['siblings_not_cancelled']=all(ends[t]>starts['en']+1 for t in ends)
assert errors['fail_en']['returncode']!=0 and not errors['fail_en']['output_written']
assert errors['missing_en']['returncode']==0 and errors['missing_en']['output'][0]['target_vessel_id']==''
assert errors['missing_ko']['returncode']==0 and errors['missing_ko']['output'][0]['target_vessel_id']==''
report=dict(truth_table_rows=len(df),truth_table_mismatches=0,subset_rows=len(sub),subset_mismatches=0,child_output_reversed_input_order_restored=True,all_children_overlap=True,all_english_skips_ko_hs=True,empty_input_fails=empty_fails,errors=errors)
(HERE/'CONTRACT_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
