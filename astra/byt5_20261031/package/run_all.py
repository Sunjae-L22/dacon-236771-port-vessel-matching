"""One-shot GPU experiment. No Public upload and no post-hoc threshold search."""
import os,sys,json,time,hashlib,subprocess,zipfile,datetime,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
REVISION='68377bdc18a2ffec8a0533fef03b1c513a4dd49d'

def run(args,log):
    env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    Path(log).write_text('')
    p=subprocess.Popen(args,cwd=HERE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout:
        print(line,end='',flush=True)
        with Path(log).open('a') as f:f.write(line)
    rc=p.wait()
    if rc:raise RuntimeError(f'exit {rc}: {log}')

def allowed_time():
    # User cutoff: after Nov1 07:00 KST, verification/selection only, no new training/build.
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    cutoff=datetime.datetime(2026,11,1,7,tzinfo=now.tzinfo)
    if now>=cutoff:raise RuntimeError('07:00 KST candidate cutoff reached. Do not start new training/build.')

def pack_results(out):
    with zipfile.ZipFile(out/'RESULTS_RETURN_TO_ASTRA.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if not p.is_file() or p.suffix not in ['.json','.jsonl','.csv','.log','.txt','.md']:continue
            if any(part in ['model','pretrained'] for part in p.relative_to(out).parts):continue
            z.write(p,str(p.relative_to(out)))

def main(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);status=dict(status='RUNNING',public_submissions=0,revision=REVISION)
    try:
        allowed_time()
        for filename in ['data_manifest.json','package_manifest.json']:
            manifest=json.loads((HERE/filename).read_text())
            for name,sha in manifest['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,name
        import torch,transformers
        assert torch.cuda.is_available(),'Select GPU runtime first'
        assert transformers.__version__=='4.57.6',transformers.__version__
        free,total=torch.cuda.mem_get_info()
        assert free>=16*1024**3,f'Only {free/1024**3:.1f}GB free. Release your completed previous notebook model before rerunning; this kit never kills another process.'
        status.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,transformers=transformers.__version__)
        (out/'environment.json').write_text(json.dumps(status,indent=2))
        from huggingface_hub import snapshot_download
        pretrained=snapshot_download('google/byt5-small',revision=REVISION,allow_patterns=['config.json','generation_config.json','pytorch_model.bin'])
        reports={}
        # Always evaluate both folds, even if the first fails. Stop before full training on failure.
        for fold in [2,4]:
            allowed_time();name=f'fold{fold}';dest=out/name;dest.mkdir()
            run([sys.executable,str(HERE/'train.py'),'--data',str(HERE/f'data/train_{name}.jsonl'),'--out',str(dest),'--pretrained',pretrained],out/f'{name}_train.log')
            run([sys.executable,str(HERE/'evaluate.py'),'--data',str(HERE/f'data/eval_{name}.jsonl'),'--model',str(dest/'model'),'--out',str(dest/'eval')],out/f'{name}_eval.log')
            reports[name]=json.loads((dest/'eval/report.json').read_text())
        status['fold_reports']=reports
        if not all(r['minimum_gate'] for r in reports.values()):
            status['status']='HOLD_VALIDATION_FAILED';return
        allowed_time();dest=out/'full';dest.mkdir()
        run([sys.executable,str(HERE/'train.py'),'--data',str(HERE/'data/train_full.jsonl'),'--out',str(dest),'--pretrained',pretrained],out/'full_train.log')
        allowed_time()
        from build import build
        candidate=out/'builds/astra110_byt5_UNREVIEWED.zip';status['build']=build(HERE/'baseline110.zip',dest/'model',candidate)
        from qa import qa
        status['qa']=qa(HERE/'baseline110.zip',candidate,out/'qa');status['status']='QA_PASSED_AWAIT_ASTRA_OPUS_REVIEW'
    except Exception as e:
        status.update(status='ERROR_NO_SUBMISSION',error=repr(e));(out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        (out/'FINAL_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2));pack_results(out)
        print('FINAL',status['status'],'RETURN',out/'RESULTS_RETURN_TO_ASTRA.zip',flush=True)
    return status

if __name__=='__main__':main(sys.argv[1])
