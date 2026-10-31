"""Astra: consolidate verified repair evidence and completed exploratory trials."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]

def read(rel):
    p=ROOT/rel
    return json.loads(p.read_text()) if p.exists() else None

def main():
    manifest=read('astra/build91_manifest.json');qa=read('astra/check91_contract.json')
    if manifest.get('submission_status')=='REJECTED_PUBLIC_REGRESSION':
        raise RuntimeError('Historical 91 finalizer is retired: Public0.9023454095 regressed; do not promote it.')
    path=ROOT/'submissions/91_submit.zip'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
    assert qa['status']=='PASS' and manifest['local_check']=='PASS'
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    result={
      'incumbent':{'submission':79,'public_lb':'0.9418190937'},
      'recommended_submission':91,
      'submission91_public_lb':None,
      'repaired_failures':[
        '88 server error: Arrow-backed DataFrame pickle required pyarrow. Builtin-only payload in89+.',
        '88/89 dictionary train OFF versus inference ON. Actual constructor-level scope fixed in90+.',
        'Canonical NFD Korean was sent to non-Korean extra retrieval. Row-level NFC boundary fixed in91.'
      ],
      'submission91':manifest,
      'submission91_qa':qa,
      'train_serving_feature_parity':read('astra/check90_contract.json'),
      'unicode_diagnostic':read('astra/unicode93/report.json'),
      'experiments':{
        'fixed87_88_average':read('astra/fusion91/report.json'),
        'learned_sequence_alignment':read('astra/sequence90/report.json'),
        'metadata_likelihood':read('astra/meta92/report.json'),
        'translated_string_features':read('astra/translated90_report.json'),
        'vessel_frequency_weighting_pilot':read('astra/balance94/report.json'),
        'vessel_frequency_extension':read('astra/balance94/extension_report.json'),
        'positive_candidate_injection':read('astra/positive96/report.json'),
        'training_cache_benchmark':read('astra/cache95/report.json'),
        'coldship_training_fold0':read('astra/coldship91/fold0/report.json'),
        'coldship_training_fold4':read('astra/coldship91/fold4/report.json'),
        'coldship_two_fold_confirmation':read('astra/coldship91/two_fold_report.json')
      },
      'limits':[
        '91 has no Public score; 0.98 is not achieved.',
        'Synthetic stress validation uses train-derived vessels and is not a forecast of hidden test accuracy.',
        'Outer folds were reused across experiments; method confirmation does not mean a pristine new holdout.',
        'Paired local baselines are single-seed Korean rankers, not the full-data Public79 ensemble.'
      ],
      'gpu':'No Colab runtime allocated or GPU compute used in this follow-up.'
    }
    (ROOT/'astra/results89_91.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for num,update in {
      87:{'submission_readiness':'BLOCKED: same Arrow pickle and dictionary contract defects as88'},
      88:{'preferred_replacement_submission':91},
      89:{'submission_readiness':'CONTROL_ONLY: dictionary mismatch remains'},
      90:{'preferred_replacement_submission':91}
    }.items():
        p=ROOT/f'astra/build{num}_manifest.json';d=json.loads(p.read_text());d.update(update);p.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({'recommended_submission':91,'public_lb':None,'qa':'PASS','bytes':path.stat().st_size,'sha256':manifest['sha256']},indent=2))

if __name__=='__main__':main()
