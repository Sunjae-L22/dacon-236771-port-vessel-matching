"""Exploratory 5-fold extension after the inconclusive, failed two-fold pilot gate."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def main():
    if (OUT/'early_stop.json').exists():
        print('EARLY_STOP_FUTILITY already recorded; no further training is authorized.',flush=True)
        return
    pilot=OUT/'report.json'
    plan=dict(created_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),status='exploratory_extension_after_inconclusive_confirmation',original_two_fold_report_sha256=hashlib.sha256(pilot.read_bytes()).hexdigest(),original_pilot_gate='FAILED; preserved unchanged',new_folds=[1,2,3],training='One prespecified inverse sqrt target query-frequency method; 70 existing features; 600 trees; seed84; 2 threads; one process at a time',aggregate_gate=dict(minimum_all5_vessel_mean_gain=.005,bootstrap95_lower_strictly_positive=True,native_all_rows_no_loss=True,maximum_single_fold_loss=.01,trunc_noise_mean_no_loss=True,original_mean_no_loss=True),limitations=['This extension follows an inconclusive confirmation and is exploratory.', 'TRAIN versus LLM source mix changes with vessel weights.', 'Full-data Public79 ensemble is not the paired validation comparator.'])
    path=OUT/'extension_plan.json'
    if path.exists():
        old=json.loads(path.read_text());plan['created_kst']=old['created_kst'];assert old==plan
    else:path.write_text(json.dumps(plan,indent=2))
    print('PLAN_FROZEN',plan['created_kst'],flush=True)
    env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for fold in plan['new_folds']:
        report=OUT/f'fold{fold}/report.json'
        if not report.exists():
            with (OUT/f'fold{fold}.log').open('w') as log:
                print('START',fold,flush=True)
                subprocess.run([sys.executable,'-u',str(ROOT/'astra/balance94.py'),'--fold',str(fold)],cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        r=json.loads(report.read_text());print('DONE',fold,r['stress_delta'],r['native_delta'],flush=True)
        matrix=OUT/f'fold{fold}/matrix70.npy'
        if matrix.exists():matrix.unlink()
    subprocess.run([sys.executable,str(OUT/'collect_extension.py')],cwd='/tmp',env=env,check=True)

if __name__=='__main__':main()
