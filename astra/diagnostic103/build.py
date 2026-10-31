"""Build unnumbered diagnostic masks from immutable 93. Never writes submissions/."""
from pathlib import Path
import hashlib
import json
import zipfile
import argparse

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SELECTED = ['tonnage_gap', 'sister', 'partial_name', 'mixed']
INSERT = '''        # Diagnostic-only output mask; baseline rank, English path and models unchanged.
        if L.is_hangul(q):
            allowed = [j for j in order if idx[j] not in train_targets] or order
            lead = allowed[0]
            runner = allowed[1] if len(allowed) > 1 else None
            gap = float(score[lead] - score[runner]) / len(models["ranker"]) if runner is not None else 99.0
            flags = predicates(q, ix.names_raw[idx[lead]],
                ix.names_raw[idx[runner]] if runner is not None else "",
                F60[lead], F60[runner] if runner is not None else None, L.FEATURES)
            if not (gap < 4.0 and flags[PROBE_CONDITION]):
                best = min(train_targets)  # known train ID; assumes test targets exclude train
'''

def sha(b): return hashlib.sha256(b).hexdigest()

def main(conditions=None):
    source = ROOT/'submissions/93_submit.zip'
    manifest=OUT/'build_manifest.json'
    reports=json.loads(manifest.read_text()) if manifest.exists() else []
    with zipfile.ZipFile(source) as z:
        base=z.read('script.py').decode()
        assert base.count('        preds.append(ix.ids[best])') == 1
        assert 'KO = {"ranker": 1.0}' in base
        for condition in (conditions or SELECTED):
            code=base.replace('import opus_lib_65 as L65',
                'import opus_lib_65 as L65\nfrom diagnostic_conditions import predicates\nPROBE_CONDITION = '+repr(condition))
            code=code.replace('        preds.append(ix.ids[best])', INSERT+'        preds.append(ix.ids[best])')
            if condition == 'hard_all':
                code=code.replace('gap < 4.0 and flags[PROBE_CONDITION]', 'gap < 4.0')
            # Force determinism even if an inherited environment sets a different seed.
            code=code.replace('os.environ.get("PYTHONHASHSEED") is None', 'os.environ.get("PYTHONHASHSEED") != "0"')
            compile(code,'script.py','exec')
            p=ROOT/'astra/builds'/f'probe93_hard_{condition}_20261030.zip'
            assert not p.exists(), p
            with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as w:
                for info in z.infolist():
                    if info.filename != 'script.py': w.writestr(info,z.read(info))
                w.writestr('script.py',code)
                w.writestr('model/diagnostic_conditions.py',(OUT/'conditions.py').read_bytes())
            with zipfile.ZipFile(p) as w:
                assert all(w.read(n)==z.read(n) for n in z.namelist() if n!='script.py')
            reports.append(dict(condition=condition,path=str(p),sha256=sha(p.read_bytes()),bytes=p.stat().st_size,
                base93_sha256=sha(source.read_bytes()),all_existing_payload_bytes_preserved=True,
                purpose='diagnostic only; intentionally masks Korean predictions',
                dummy='minimum catalog index among stats cs_map train targets; requires train/test disjointness'))
    (OUT/'build_manifest.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False))
    print(json.dumps(reports,indent=2,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--condition',choices=SELECTED+['hard_all'])
    args=parser.parse_args()
    main([args.condition] if args.condition else None)
