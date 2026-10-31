from pathlib import Path
import sys,zipfile,hashlib,time,shutil,json,subprocess
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'package'));import run_all as m
D=R/'runner_guard_qa';D.mkdir(exist_ok=True)
m.CUTOFF=time.time()-1
try:m.remaining();raise AssertionError('cutoff ignored')
except RuntimeError:pass
m.CUTOFF=time.time()+300
m.RUN_DEADLINE=time.monotonic()-1
try:m.remaining();raise AssertionError('whole-run budget ignored')
except RuntimeError:pass
m.RUN_DEADLINE=None
for kind,member in [('valid','script.py'),('unsafe','../escape.py')]:
 z=D/f'{kind}.zip'
 with zipfile.ZipFile(z,'w') as f:f.writestr(member,'# fixture')
 m.EXPECTED=hashlib.sha256(z.read_bytes()).hexdigest();dest=D/kind
 if dest.exists():shutil.rmtree(dest)
 if kind=='valid':m.extract(z,dest);assert (dest/'script.py').exists()
 else:
  try:m.extract(z,dest);raise AssertionError('unsafe path accepted')
  except ValueError:pass
assert not (D/'escape.py').exists()
m.EXPECTED='bad'
try:m.extract(D/'valid.zip',D/'wrong');raise AssertionError('hash mismatch accepted')
except ValueError:pass
for name,code,cap in [('failed','raise SystemExit(7)',2),('timeout','import time\ntime.sleep(5)',.1)]:
 script=D/f'{name}.py';script.write_text(code);log=D/f'{name}.log'
 try:m.run([sys.executable,script.resolve()],D.resolve(),log.resolve(),cap=cap);raise AssertionError('failed stage accepted')
 except (RuntimeError,subprocess.TimeoutExpired):pass
 receipt=json.loads(log.with_suffix('.stage.json').read_text());assert receipt['elapsed_seconds']>=0 and receipt['state']!='COMPLETE'
print('PASS deadline refusal, frozen hash enforcement, safe ZIP extraction, traversal rejection')
