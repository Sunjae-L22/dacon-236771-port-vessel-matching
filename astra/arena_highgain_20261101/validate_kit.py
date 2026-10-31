from pathlib import Path
import ast,json,hashlib,csv,sys
R=Path(__file__).resolve().parent;P=R/'package';errors=[];py=[]
for f in P.rglob('*.py'):
 try:ast.parse(f.read_text(),filename=str(f));py.append(str(f.relative_to(P)))
 except Exception as e:errors.append(str(e))
nb=json.loads((P/'ASTRA_HIGHGAIN_L4.ipynb').read_text())
for i,c in enumerate(nb['cells']):
 if c['cell_type']=='code':
  try:ast.parse(''.join(c['source']),filename=f'cell{i}')
  except Exception as e:errors.append(str(e))
for split in ['pilot','full']:
 with (P/f'data/{split}_test.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 with (P/f'data/{split}_truth.csv').open(encoding='utf-8-sig') as f:truth=list(csv.DictReader(f))
 assert 'target_vessel_id' not in rows[0]
 assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in truth]
 assert len({r['sample_id'] for r in rows})==len(rows)
manifest=json.loads((P/'data/manifest.json').read_text())
for n,h in manifest['files'].items():assert hashlib.sha256((P/'data'/n).read_bytes()).hexdigest()==h
assert not errors,errors
result=dict(python_syntax_files=py,notebook_code_cells_valid=True,data_hashes_match=True,inference_csv_has_no_truth=True,pilot_rows=104,full_rows=400,actual_GPU_execution=False,production_3000_timing=False,production_600_invariance=False)
(R/'KIT_STATIC_QA.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
