from pathlib import Path
import sys,json,zipfile,hashlib,shutil,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
src=Path(__import__('os').environ['DRIVE_PROJECT']+'/combo_kit/out/C_vote.zip');expected='601c71ab3bc87ee5128305261c1708c432eb75dca53258e4d3841cce813a97d3';dest=HERE/'payload_mps';weights=('model/ko/model/encoder/model.safetensors','model/ko/model/llm/model.safetensors','model/ko/model/qwen/model.safetensors')
with zipfile.ZipFile(src) as z:
 before=json.loads((ROOT/'astra/arena_20261101/CVOTE_MEMBERS.json').read_text());old={r['name']:(r['size'],r['crc']) for r in before};now={r.filename:(r.file_size,r.CRC) for r in z.infolist()};assert old==now,'Source ZIP member metadata changed'
 need=sum(z.getinfo(n).file_size for n in weights);free=shutil.disk_usage(HERE).free
 if free<need+src.stat().st_size+2*1024**3:raise RuntimeError('Insufficient conservative disk reserve for source hydration and extracted weights')
 if dest.exists():raise RuntimeError('Destination exists, no overwrite')
 shutil.copytree(ROOT/'astra/arena_20261101/baseline',dest,ignore=shutil.ignore_patterns('__pycache__'))
 t=time.time();h=hashlib.sha256()
 with src.open('rb') as f:
  total=0
  for chunk in iter(lambda:f.read(8*1024**2),b''):
   h.update(chunk);total+=len(chunk)
   if total%(512*1024**2)==0:print('HASH_BYTES',total,'seconds',round(time.time()-t),flush=True)
 assert h.hexdigest()==expected,'Original C_vote source hash mismatch'
 print('ORIGINAL_ZIP_SHA256_VERIFIED',h.hexdigest(),flush=True)
 for name in weights:
  p=dest/name
  with z.open(name) as inp,p.open('wb') as out:
   shutil.copyfileobj(inp,out,length=8*1024**2)
  print('EXTRACTED',name,p.stat().st_size,'seconds',round(time.time()-t),flush=True)
 edits=[]
 for name in ('llm_translit.py','encoder_pair.py','neural_pair.py'):
  p=dest/'model/ko/model/src'/name;s=p.read_text();oldhash=hashlib.sha256(p.read_bytes()).hexdigest()
  if name=='llm_translit.py':a,b='device: str = "cuda"','device: str = "mps"'
  else:a,b='device="cuda"','device="mps"'
  assert s.count(a)==1;(p).write_text(s.replace(a,b));edits.append({'file':str(p.relative_to(dest)),'before':oldhash,'after':hashlib.sha256(p.read_bytes()).hexdigest(),'edit':'default CUDA device to MPS only'})
 report={'source_zip_sha256':expected,'source_member_metadata_match':True,'only_source_edits':edits,'scope':'local MPS reference using original weights; not identical CUDA numerical execution or server-time measurement','seconds':time.time()-t,'free_bytes_after':shutil.disk_usage(HERE).free}
 (HERE/'PREPARE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
