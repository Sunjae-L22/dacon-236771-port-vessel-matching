from pathlib import Path
import hashlib,json,zipfile,datetime,ast
R=Path(__file__).resolve().parent;P=R/'package'
for name in ['run_candidate.py','EXPERIMENT.md','SOURCE_PROVENANCE.json','run_all.py','ASTRA_HIGHGAIN_L4.ipynb']:
 assert (P/name).is_file(),f'Missing final file: {name}'
for p in P.rglob('*.py'):ast.parse(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={p.relative_to(P).as_posix():sha(p) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='PACKAGE_MANIFEST.json' and '__pycache__' not in p.parts}
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'OFFLINE_EVALUATION_NOT_SUBMISSION','public_submissions':0,'cutoff_utc':'2026-10-31T22:00:00Z','files':files}
(P/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
out=R/'ASTRA_HIGHGAIN_KIT.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name in sorted(files):z.write(P/name,name)
 z.write(P/'PACKAGE_MANIFEST.json','PACKAGE_MANIFEST.json')
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 assert 'script.py' not in z.namelist() and not any(n.startswith('model/') for n in z.namelist())
report={'file':str(out.resolve()),'bytes':out.stat().st_size,'sha256':sha(out),'members':len(files)+1,'archive_crc_ok':True,'not_submission_zip':True,'contains_model_weights':False,'GPU_execution_unrun':True}
(R/'DELIVERY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
