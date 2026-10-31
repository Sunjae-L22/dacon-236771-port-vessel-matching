from pathlib import Path
import sys,json,subprocess,tempfile,os,shutil
ROOT=Path(__file__).resolve().parent;PKG=ROOT/'package';sys.path.insert(0,str(PKG))
import baseline_phase as BP

def main():
    reports=[]
    for fold in [2,4]:
        ps=[json.loads(s) for s in (PKG/f'data/eval_fold{fold}.jsonl').read_text().splitlines()];qs=json.loads((PKG/f'data/queries_fold{fold}.json').read_text())
        selected=[]
        for hard,clean in [(True,True),(True,False),(False,True),(False,False)]:
            hits=[i for i,p in enumerate(ps) if (p['gap']<4)==hard and p['variant'].startswith('clean')==clean]
            if hits:selected+=hits[:2]
        selected=list(dict.fromkeys(selected));packets=[ps[i] for i in selected];queries=[qs[i] for i in selected]
        cat,ix,fx,models=BP.context(fold);excluded=set(ix.cs_map.values());generated={}
        for p,q in zip(packets,queries):
            if p['gap']>=4:continue
            cand=fx.add(BP.B.query_candidates(ix,q),q);new=next(i for i in range(len(ix.ids)) if i not in cand and i not in excluded)
            generated[p['name']]=[dict(name=str(cat.iloc[new].vessel_name).upper(),ids=[str(ix.ids[new])])]
        with tempfile.TemporaryDirectory(prefix='astra_reuse_fixture_') as d:
            work=Path(d);(work/'data').mkdir();(work/'baseline').symlink_to(PKG/'baseline',target_is_directory=True);(work/'data/vessel_catalog.csv').symlink_to(PKG/'data/vessel_catalog.csv')
            shutil.copy2(PKG/'baseline_phase.py',work/'baseline_phase.py')
            outputs=[]
            for label in ['normal','changed_labels']:
                pp=[dict(p,target=('NOT_A_CATALOG_ID' if label=='changed_labels' else p['target'])) for p in packets]
                qq=[dict(q,target_vessel_id=('NOT_A_CATALOG_ID' if label=='changed_labels' else q['target_vessel_id'])) for q in queries]
                (work/f'data/eval_fold{fold}.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pp));(work/f'data/queries_fold{fold}.json').write_text(json.dumps(qq,ensure_ascii=False))
                out=work/label;out.mkdir();(out/'generated.json').write_text(json.dumps(generated,ensure_ascii=False))
                subprocess.run([sys.executable,str(work/'baseline_phase.py'),str(fold),str(out)],cwd='/tmp',check=True)
                rows=[json.loads(s) for s in (out/'expanded_packets.jsonl').read_text().splitlines()];outputs.append(rows)
                if label=='normal':report=json.loads((out/'baseline_report.json').read_text())
            for a,b in zip(*outputs):
                for key in ['candidates','retrieval_pred','beam_ids','original_full_ids','expanded_full_ids','new_candidates']:assert a[key]==b[key],key
            assert all(r['retrieval_pred']==r['baseline'] for r in outputs[0] if r['gap']>=4)
            assert any(r['new_candidates']>0 for r in outputs[0])
            saved=ROOT/'local_smoke'/f'fold{fold}';saved.mkdir(parents=True,exist_ok=True)
            shutil.copy2(work/'normal/expanded_packets.jsonl',saved/'expanded_packets.jsonl')
            report.update(fold=fold,fixture_rows=len(packets),label_change_prediction_mismatches=0,easy_changes=0,new_candidate_path_tested=True)
            reports.append(report)
    (ROOT/'BASELINE_SMOKE.json').write_text(json.dumps(reports,indent=2));print(reports)

if __name__=='__main__':main()
