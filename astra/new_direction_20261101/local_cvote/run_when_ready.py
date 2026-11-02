from pathlib import Path
import time,sys,subprocess,os,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];deadline=datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp()
start=time.time()
while not (HERE/'PREPARE.json').exists():
 if time.time()>=deadline or time.time()-start>1800:raise RuntimeError('Preparation wait cutoff')
 time.sleep(5)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONHASHSEED='0',TMPDIR=str(HERE/'tmp'))
for test,out in [('v20_needed.csv','v20_pred.csv'),('v20_checks.csv','v20_checks_pred.csv')]:
 if time.time()>=deadline:raise RuntimeError('07:00 cutoff')
 with (HERE/(out+'.log')).open('w') as f:subprocess.run([sys.executable,str(HERE/'run_v20.py'),str(HERE/'eval'/test),str(HERE/'eval'/out)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
subprocess.run([sys.executable,str(HERE/'finish_comparison.py')],env=env,check=True)
