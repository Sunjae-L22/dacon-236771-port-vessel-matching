"""Author: Astra. Check the preserved English route against immutable submission79."""
from pathlib import Path
import tempfile,zipfile,subprocess,os,sys,json,unicodedata,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/src'))
import local_check as C

def main():
    test=C.corrupted_test();test=test[~test.vessel_name.map(lambda s:C.L.is_hangul(unicodedata.normalize('NFC',str(s))))].sample(200,random_state=88)
    env={k:v for k,v in os.environ.items() if k!='KO_DICT'};env.update(PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    outputs={};times={}
    for n in [79,87]:
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            with zipfile.ZipFile(ROOT/f'submissions/{n}_submit.zip') as z:z.extractall(d)
            (d/'data').mkdir();(d/'data/vessel_catalog.csv').write_bytes((ROOT/'data/open/vessel_catalog.csv').read_bytes())
            test.drop(columns='target_vessel_id').to_csv(d/'data/test.csv',index=False)
            start=time.time();p=subprocess.run([sys.executable,'script.py'],cwd=d,env=env,capture_output=True,text=True)
            assert p.returncode==0,p.stderr
            outputs[n]=pd.read_csv(d/'output/submission.csv').set_index('sample_id').target_vessel_id.sort_index();times[n]=time.time()-start
    mismatch=int((outputs[79]!=outputs[87]).sum());assert mismatch==0,mismatch
    result=dict(rows=200,mismatches=mismatch,seconds=times,reference=79,candidate=87,source='train-derived corrupted English rows; no test data')
    (ROOT/'astra/english87_parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
