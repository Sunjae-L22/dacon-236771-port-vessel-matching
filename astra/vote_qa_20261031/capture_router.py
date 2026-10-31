"""Run unchanged router; preserve child outputs only for independent QA."""
import importlib.util,sys,shutil,json,time
from pathlib import Path
sys.dont_write_bytecode=True
script,test,catalog,dest=map(Path,sys.argv[1:]);dest.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location('router',script);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
original=m.finish;events=[]
def finish(job):
    result=original(job)
    if job is not None:
        pkg,p,dst,n,t=job
        shutil.copy(dst,dest/f'{pkg}_pred.csv');shutil.copy(Path(dst).with_name(f'{pkg}_test.csv'),dest/f'{pkg}_test.csv')
        events.append(dict(pkg=pkg,pid=p.pid,rows=n,start=t,observed_finished=time.time()))
    return result
m.finish=finish
m.run(str(test),str(catalog),str(dest/'pred.csv'))
(dest/'processes.json').write_text(json.dumps(events,indent=2))
