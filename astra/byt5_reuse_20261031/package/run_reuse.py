"""Find exact prior Drive run, reuse read-only checkpoints, evaluate without submissions."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,zipfile,time,shutil,traceback,datetime
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def command(args,log):
    env=dict(os.environ,PYTHONHASHSEED='0',KO_DICT='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    p=subprocess.Popen(args,cwd='/tmp',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout:
        print(line,end='',flush=True)
        with Path(log).open('a') as f:f.write(line)
    if p.wait():raise RuntimeError(f'Failed: {log}')

def main(previous_root,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);status=dict(status='RUNNING_REUSE_ONLY',public_submissions=0,new_training_steps=0,candidate_built=False)
    try:
        manifest=json.loads((HERE/'input_manifest.json').read_text());package=json.loads((HERE/'package_manifest.json').read_text())
        for m in [manifest,package]:
            for name,digest in m['files'].items():assert sha(HERE/name)==digest,name
        matches=[p.parent for p in Path(previous_root).glob('run_*/FINAL_STATUS.json') if sha(p)==manifest['expected_previous_status_sha256']]
        assert matches,'Previous exact run not found. Keep the previous Drive run with fold2/model and fold4/model.'
        previous=sorted(matches)[-1];status['previous_run']=str(previous)
        import torch,transformers
        assert torch.cuda.is_available(),'Select GPU runtime first'
        assert transformers.__version__=='4.57.6'
        status.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,transformers=transformers.__version__)
        reports={};hashes={}
        for fold in [2,4]:
            src=previous/f'fold{fold}/model';assert src.is_dir()
            complete=json.loads((previous/f'fold{fold}/COMPLETE.json').read_text());assert complete['steps']==1500 and not complete['smoke_only']
            assert complete['train_file_sha256']==sha(HERE/f'data/train_fold{fold}.jsonl')
            # Copy once to ephemeral local disk. The Drive checkpoint is never opened for writing.
            local=HERE/'reused_weights'/f'fold{fold}';local.mkdir(parents=True,exist_ok=False);hashes[f'fold{fold}']={}
            for p in sorted(src.iterdir()):
                if not p.is_file():continue
                dest=local/p.name;shutil.copyfile(p,dest);hashes[f'fold{fold}'][p.name]=sha(dest)
            dest=out/f'fold{fold}';dest.mkdir()
            for phase in ['generate','baseline','score','evaluate']:
                print('PHASE',fold,phase,flush=True)
                if phase in ['generate','score']:args=[sys.executable,'-u',str(HERE/'gpu_phase.py'),phase,'--fold',str(fold),'--model',str(local),'--out',str(dest)]
                else:args=[sys.executable,'-u',str(HERE/('baseline_phase.py' if phase=='baseline' else 'evaluate_reuse.py')),str(fold),str(dest)]
                command(args,out/f'fold{fold}_{phase}.log')
            reports[f'fold{fold}']=json.loads((dest/'report.json').read_text())
        status.update(status='REUSE_EVALUATION_COMPLETE_NO_SUBMISSION',reports=reports,reused_model_sha256=hashes)
    except Exception as e:
        status.update(status='ERROR_NO_SUBMISSION',error=repr(e));(out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        (out/'FINAL_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(out/'REUSE_RESULTS_RETURN_TO_ASTRA.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p.suffix in ['.json','.jsonl','.log','.txt']:z.write(p,str(p.relative_to(out)))
        print('RETURN',out/'REUSE_RESULTS_RETURN_TO_ASTRA.zip',flush=True)

if __name__=='__main__':main(*sys.argv[1:])
