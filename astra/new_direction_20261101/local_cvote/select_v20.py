from pathlib import Path
import csv,json,hashlib
HERE=Path(__file__).resolve().parent;D=HERE/'eval'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
rows=read(D/'test.csv');en={r['sample_id']:r for r in read(D/'en_pred.csv')};hs={r['sample_id']:r for r in read(D/'hs_pred.csv')};sem={r['sample_id']:r for r in map(json.loads,(D/'semantic112.jsonl').read_text().splitlines())};needed=[]
for r in rows:
 s=r['sample_id'];k=any('가'<=c<='힣' for c in r['vessel_name']);hard=k and float(en[s]['ko_gap'])<4
 if hard and sem[s]['changed']:needed.append(r)
checks=sorted(needed,key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:6]
def write(name,rr):
 with (D/name).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rr)
write('v20_needed.csv',needed);write('v20_checks.csv',list(reversed(checks)));write('v20_pilot.csv',needed[:1]);(D/'V20_SELECTION.json').write_text(json.dumps({'total_rows':len(rows),'v20_needed':len(needed),'reason':'all E5-modified rows; unchanged rows equal original C_vote by construction. Net rescue/harm only, not overall C_vote accuracy. No labels read','checks':len(checks)},indent=2));print('NEEDED',len(needed))
