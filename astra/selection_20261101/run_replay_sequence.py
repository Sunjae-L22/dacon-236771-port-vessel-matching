from pathlib import Path
import os,sys,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];env=dict(os.environ,PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
for strategy in ('mask','context'):
 with (HERE/(strategy+'_replay.log')).open('w') as f:subprocess.run([sys.executable,str(HERE/'replay_mask.py')],env=dict(env,ASTRA_STRATEGY=strategy),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
subprocess.run([sys.executable,str(HERE/'evaluate_variants.py')],env=env,check=True)
# Original-reference confirmation inference; no new candidate and no labels passed.
with (HERE/'confirm/capture.log').open('w') as f:subprocess.run([sys.executable,str(HERE/'run_capture.py'),str(HERE/'confirm/hard_test.csv'),str(HERE/'confirm/v20_hard_pred.csv')],env=dict(env,ASTRA_CAPTURE_DIR=str(HERE/'confirm/capture')),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
