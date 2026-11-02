"""No submissions. Run frozen C_vote and one predeclared offline experiment before 07:00 KST."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time,zipfile,shutil,traceback,datetime,re
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
EXPECTED='601c71ab3bc87ee5128305261c1708c432eb75dca53258e4d3841cce813a97d3'
CUTOFF=datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp()
RUN_DEADLINE=None

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):
   remaining();h.update(b)
 return h.hexdigest()
def remaining():
 n=CUTOFF-time.time()
 if n<=0:raise RuntimeError('07:00 KST cutoff reached: no new experiment execution')
 if RUN_DEADLINE is not None:
  n=min(n,RUN_DEADLINE-time.monotonic())
  if n<=0:raise RuntimeError('Whole diagnostic run exceeded 1800-second watchdog')
 return n
def run(argv,cwd,log,cap=1800):
 seconds=min(remaining(),cap);started=time.monotonic()
 receipt={'budget_seconds':seconds,'state':'STARTED','command':[str(x) for x in argv]}
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with Path(log).open('w') as out:
  p=None
  try:
   p=subprocess.Popen([str(x) for x in argv],cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   rc=p.wait(timeout=seconds);receipt.update(state='COMPLETE' if rc==0 else 'FAILED',returncode=rc)
  except BaseException:
   if p is not None:
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    p.wait()
   receipt.update(state='FAILED_OR_INTERRUPTED',returncode=p.returncode if p is not None else None);raise
  finally:
   receipt['elapsed_seconds']=time.monotonic()-started;Path(log).with_suffix('.stage.json').write_text(json.dumps(receipt,indent=2))
 if rc:raise RuntimeError(f'Failed stage {Path(log).name}, rc={rc}; see log')
 return time.monotonic()-started

def extract(source,dest):
 remaining()
 if sha(source)!=EXPECTED:raise ValueError('C_vote.zip SHA256 mismatch; do not substitute another package')
 with zipfile.ZipFile(source) as z:
  total=sum(x.file_size for x in z.infolist())
  if shutil.disk_usage(dest.parent).free<total+2*1024**3:raise RuntimeError('Insufficient space for original weights plus 2 GiB reserve')
  dest.mkdir(exist_ok=False)
  for i in z.infolist():
   remaining();p=dest/i.filename
   if not p.resolve().is_relative_to(dest.resolve()) or ((i.external_attr>>16)&0o170000)==0o120000:raise ValueError('Unsafe ZIP member')
   if i.is_dir():p.mkdir(parents=True,exist_ok=True)
   else:
    p.parent.mkdir(parents=True,exist_ok=True)
    with z.open(i) as source_member,p.open('wb') as target:
     for block in iter(lambda:source_member.read(8*1024*1024),b''):
      remaining();target.write(block)
 if not (dest/'script.py').exists():raise ValueError('Unexpected baseline ZIP layout')

def same_predictions(a,b):
 import pandas as pd
 x=pd.read_csv(a,dtype=str).set_index('sample_id');y=pd.read_csv(b,dtype=str).set_index('sample_id')
 if not set(y.index)<=set(x.index):raise ValueError('Replay row mismatch')
 return int((x.loc[y.index,'target_vessel_id']!=y.target_vessel_id).sum())

def main():
 global RUN_DEADLINE
 started=time.monotonic();RUN_DEADLINE=started+1800
 p=argparse.ArgumentParser();p.add_argument('--cvote',required=True);p.add_argument('--work',required=True);a=p.parse_args()
 work=Path(a.work).resolve();work.mkdir(parents=True,exist_ok=False);out=work/'results';out.mkdir();timings={};status={'status':'RUNNING','public_submissions':0,'model_zip_created':False,'cutoff_utc':'2026-10-31T22:00:00Z','baseline_sha256':EXPECTED,'evidence':'TRAIN_TOKEN_COMPOSITION_SYNTHETIC','independent_holdout':False,'submission_eligible':False}
 try:
  remaining()
  import torch,pandas as pd
  if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for original C_vote; CPU channel adds no GPU model')
  hardware=dict(gpu=torch.cuda.get_device_name(0),gpu_memory=torch.cuda.get_device_properties(0).total_memory,torch=torch.__version__,python=sys.version,logical_cpus=os.cpu_count())
  (out/'hardware.json').write_text(json.dumps(hardware,indent=2));status['actual_L4']=bool(re.search(r'\bL4\b',hardware['gpu']))
  for name,expected in json.loads((HERE/'PACKAGE_MANIFEST.json').read_text())['files'].items():
   if sha(HERE/name)!=expected:raise ValueError(f'Kit content changed: {name}')
  shutil.copy(HERE/'PACKAGE_MANIFEST.json',out/'PACKAGE_MANIFEST.json');shutil.copy(HERE/'data/manifest.json',out/'data_manifest.json')
  payload=work/'payload';t=time.monotonic();extract(a.cvote,payload);timings['verify_and_extract']=time.monotonic()-t
  for name,expected in json.loads((HERE/'CPU_SNAPSHOT_HASHES.json').read_text())['files'].items():
   if sha(payload/name)!=expected:raise ValueError(f'CPU snapshot differs from actual C_vote: {name}')
  status['cpu_snapshot_matches_original_payload']=True
  test=HERE/'data/test.csv';cat=HERE/'data/vessel_catalog.csv'
  timings['original_baseline']=run([sys.executable,HERE/'capture_baseline.py',payload,test,cat,out/'baseline'],HERE,out/'baseline.log')
  timings['cpu_channel_and_replay']=run([sys.executable,HERE/'run_channel.py',payload,test,cat,out/'baseline',out/'experiment'],HERE,out/'channel.log',cap=300)
  from evaluate import evaluate
  report=evaluate(HERE/'data',out/'baseline',out/'experiment',out/'evaluation')
  remaining();status.update(status='COMPLETE',paired_summary=report['cohorts'],clean_harm=report['clean_harm'],offline_screen_pass=report['offline_screen_pass'])
  status['production_3000_time_measured']=False;status['production_600_invariance_measured']=False
 except BaseException as e:
  status.update(status='FAILED_OR_STOPPED',error=str(e));(out/'error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
 finally:
  status['total_diagnostic_seconds']=time.monotonic()-started;status['whole_run_budget_seconds']=1800
  status['stage_receipts']={p.stem:json.loads(p.read_text()) for p in out.glob('*.stage.json')};status['timings_seconds']=timings;status['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
  archive=work/'ASTRA_CHANNEL_RESULTS.zip'
  with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
   for f in sorted(out.rglob('*')):
    if f.is_file() and 'tmp' not in f.relative_to(out).parts:z.write(f,f.relative_to(out))
  print(json.dumps(status,ensure_ascii=False,indent=2));print(f'RETURN_FILE={archive}',flush=True)
 if status['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
