"""CPU-only structural QA; no outer-fold evaluation and no training expansion."""
from pathlib import Path
import hashlib,json,random,subprocess,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent;PACKAGE=ROOT/'package'
sys.path.insert(0,str(PACKAGE))
from model import normalize,load_checkpoint,score_pairs
from train import VesselCycle,group,corrupt

def main():
    data=json.loads((PACKAGE/'data/pairs.json').read_text());catalog=json.loads((PACKAGE/'data/catalog.json').read_text());tune=json.loads((PACKAGE/'data/tune.json').read_text());config=json.loads((PACKAGE/'config.json').read_text());smoke=json.loads((ROOT/'smoke/complete.json').read_text())
    assert smoke['status']=='COMPLETE' and not smoke['outer2_read']
    coverage={}
    for source in ['native','llm','catalog']:
        cycle=VesselCycle([r for r in data if r['source']==source],random.Random(97))
        for _ in range(len(cycle.keys)):cycle.take()
        assert len(cycle.seen_targets)==len(cycle.keys);coverage[source]=len(cycle.keys)
    # Explicit equal-name positives must never become negative labels.
    toy=[{'name':name,'norm':normalize(name),'phon':name.lower()} for name in ['BLUE SKY','Blue-Sky','BLUE OCEAN','RED SKY','OTHER NAME']]
    row={'q':'블루 스카이','t':0,'lex':[1,2],'phon':[3,4]}
    positive_labels_checked=False
    for seed in range(50):
        pairs,labels,valid,weight=group(row,toy,random.Random(seed),5)
        for (_,candidate),positive,allowed in zip(pairs,labels,valid):
            if candidate['norm']==toy[0]['norm']:assert positive and allowed;positive_labels_checked=True
    assert positive_labels_checked
    # All three Jamo classes are supported; examples are generated from train data.
    kinds={corrupt(data[0]['q'],random.Random(seed))[1] for seed in range(200)}
    assert 'jamo_edit' in kinds
    model,tokenizer,_=load_checkpoint(ROOT/'smoke/best.pt','cpu')
    query=data[0]['q'];i=data[0]['t'];batch=[(query,catalog[i]),(''.join(__import__('unicodedata').normalize('NFD',query)),catalog[i])]
    scores=score_pairs(model,tokenizer,batch,'cpu',2);assert abs(scores[0]-scores[1])<1e-5
    # Exercise allowed-mask handling, ties, top-k, and exact weight-zero baseline parity.
    probe=[]
    for row in tune[:2]:
        r=dict(row);ids=r['candidates'];r['baseline_scores']=[0.]*len(ids);r['allowed']=[j!=0 for j in range(len(ids))];r['baseline_pred']=min(i for j,i in enumerate(ids) if r['allowed'][j]);probe.append(r)
    probe.append({'q':tune[0]['q'],'t':1,'sample_id':'QA_TIE_OUTSIDE_INPUT_TOP100','variant':'orig','candidates':list(range(150,0,-1)),'baseline_scores':[0.]*150,'allowed':[True]*150,'baseline_pred':1})
    source=ROOT/'qa_candidates.jsonl';source.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in probe)+'\n');dest=ROOT/'qa_evaluate.json'
    command=[sys.executable,str(PACKAGE/'evaluate.py'),'--checkpoint',str(ROOT/'smoke/best.pt'),'--data',str(PACKAGE/'data'),'--candidates',str(source),'--out',str(dest),'--split','tune4','--weight','0','--device','cpu']
    p=subprocess.run(command,cwd='/tmp',text=True,capture_output=True);assert p.returncode==0,p.stderr
    evaluated=json.loads(dest.read_text());assert all(r['pred']==r['baseline_pred'] for r in evaluated['predictions'])
    bad=[v for v in command];bad[bad.index('tune4')]='outer2';where=bad.index('--weight');del bad[where:where+2]
    denied=subprocess.run(bad,cwd='/tmp',text=True,capture_output=True);assert denied.returncode!=0 and 'Freeze fusion weight' in denied.stderr
    report={'status':'PASS','cpu_training_steps':2,'cpu_reference_batch_max_error':smoke['batch_reference_max_error'],'nfc_nfd_score_difference':abs(scores[0]-scores[1]),'vessel_cycle_coverage':coverage,'identical_name_positive_mask_passed':True,'jamo_augmentation_present':True,'allowed_mask_weight0_tie_break_passed':True,'top100_contains_full_baseline_winner_even_if_input_position150':True,'outer2_without_fixed_weight_rejected':True,'outer2_evaluated':False,'parameter_count':sum(p.numel() for p in model.parameters()),'files_sha256':{str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
    (ROOT/'qa_report.json').write_text(json.dumps(report,indent=2)+'\n')
    target=ROOT/'neural97b_package.zip'
    if target.exists():raise FileExistsError(target)
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(PACKAGE))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    (ROOT/'package_manifest.json').write_text(json.dumps({'status':'CPU_QA_PASSED_GPU_UNTESTED','path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'qa':'qa_report.json'},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2));print(target,target.stat().st_size,hashlib.sha256(target.read_bytes()).hexdigest())

if __name__=='__main__':main()
