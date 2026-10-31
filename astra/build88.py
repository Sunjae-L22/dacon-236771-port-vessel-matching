"""Author: Astra. Gate a new Korean ranker without rank_name; preserve 79 English."""
from pathlib import Path
import json,hashlib,zipfile
from build87 import SCRIPT
ROOT=Path(__file__).resolve().parents[1]

def main():
    report=json.loads((ROOT/'astra/rankless87_oof_report.json').read_text())
    assert report['eligible_for_full_training'],'Rankless OOF gate rejected this candidate'
    dst=ROOT/'submissions/88_submit.zip'
    if dst.exists():raise FileExistsError(dst)
    script=SCRIPT.replace('Astra87','Astra88').replace('ko_stats87.pkl','ko_stats88.pkl').replace('ko_ranker87.txt','ko_ranker88.txt')
    script=script.replace('    assert ko_model.feature_name()==A.FEATURES',"    assert ko_model.feature_name()==[n for n in A.FEATURES if n!='rank_name']\n    ko_columns=[A.FEATURES.index(n) for n in ko_model.feature_name()]")
    script=script.replace('scores=ko_model.predict(F,num_threads=1)','scores=ko_model.predict(F[:,ko_columns],num_threads=1)')
    compile(script,'script.py','exec')
    with zipfile.ZipFile(ROOT/'submissions/79_submit.zip') as z:items={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
    items['script.py']=script.encode();items['model/astra_lib.py']=(ROOT/'shared/experiments/pinned/build84_astra/astra_lib.py').read_bytes()
    items['model/ko_ranker88.txt']=(ROOT/'astra/rankless87_full/ranker.txt').read_bytes();items['model/ko_stats88.pkl']=(ROOT/'astra/full84/stats.pkl').read_bytes()
    with zipfile.ZipFile(dst,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(items.items()):z.writestr(n,b)
    receipt=dict(base='79 English route, new Korean-only ranker without rank_name',ko_dict=0,features=70,neural=False,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size,local_check='PENDING',public_lb=None,validation='Paired 5-fold single-seed Korean-only models; primary confirmation folds3-4; not direct heldout validation against full-data 79 ensemble')
    (ROOT/'astra/build88_manifest.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
