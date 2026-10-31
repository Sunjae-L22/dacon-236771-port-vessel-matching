from pathlib import Path
import subprocess,sys,argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
OUT=Path(__file__).resolve().parent
def job(scope,seed):
    dest=OUT/'light'/scope/f'seed{seed}'
    if (dest/'manifest.json').exists():return
    with (OUT/f'light_{scope}_{seed}.log').open('w') as f:
        p=subprocess.run([sys.executable,str(OUT/'train_light.py'),scope,str(seed)],cwd='/tmp',stdout=f,stderr=subprocess.STDOUT)
    assert p.returncode==0,(scope,seed)
    print('LIGHT_JOB_DONE',scope,seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope');p.add_argument('--seeds',default='0,5000,1000,2000,6000,7000');p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(job,a.scope,int(s)) for s in a.seeds.split(',')]):f.result()
