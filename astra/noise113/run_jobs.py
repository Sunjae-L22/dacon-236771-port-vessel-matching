"""Astra: two CPU jobs at a time, no shared source writes, children cwd=/tmp."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
def job(scope,seed):
    for stage in ['prepare','train']:
        command=[sys.executable,str(OUT/'train.py'),scope,str(seed)]+(['--prepare'] if stage=='prepare' else [])
        log=OUT/f'{scope}_{seed}_{stage}.log'
        with log.open('w') as f:
            p=subprocess.run(command,cwd='/tmp',stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'{stage} failed: {log}')
        print('JOB_DONE',scope,seed,stage,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope');p.add_argument('--seeds',default='0,5000,1000,2000,6000,7000');a=p.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(job,a.scope,int(s)) for s in a.seeds.split(',')]
        for f in as_completed(fs):f.result()
