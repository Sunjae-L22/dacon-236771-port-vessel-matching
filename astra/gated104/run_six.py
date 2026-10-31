"""Astra: two simultaneous local training jobs, all children start in /tmp."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys,json,time
OUT=Path(__file__).resolve().parent
def one(seed):
    with (OUT/f'train_{seed}.log').open('w') as log:
        result=subprocess.run([sys.executable,str(OUT/'train_g2.py'),'--seed',str(seed)],cwd='/tmp',stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'seed{seed} failed; read train_{seed}.log')
    return seed
if __name__=='__main__':
    assert (OUT/'train_manifest.json').exists()
    started=time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(one,s) for s in [0,5000,1000,2000,6000,7000]]
        for f in as_completed(futures):print('SIX_MODEL_PROGRESS',f.result(),'seconds',round(time.time()-started),flush=True)
    print('SIX_MODELS_COMPLETE',round(time.time()-started),flush=True)
