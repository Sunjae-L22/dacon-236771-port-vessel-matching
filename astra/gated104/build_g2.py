"""Astra: provisional G2 package. Publication to astra/builds requires evaluation/review."""
from pathlib import Path
import hashlib,json,pickle,zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
GATE='            gap = float(s[allowed[0]] - s[allowed[1]]) if len(allowed) > 1 else 99.0'
BLOCK='''        if L.is_hangul(q):
            s = score / len(models["ranker"])
            order = sorted(range(len(idx)), key=lambda j: (-float(s[j]), idx[j]))
            allowed = [j for j in order if idx[j] not in train_targets] or order
'''+GATE+'''
            if gap < 4.0:
                alt = sum(m.predict(F60) for m in g2alt) / len(g2alt)
                mixed = 0.5 * s + 0.5 * alt
                chosen = min(allowed, key=lambda j: (-float(mixed[j]), idx[j]))
                best = idx[chosen]
'''

def main():
    assert GATE in (ROOT/'shared/experiments/build_102_probe.py').read_text()
    p=OUT/'g2_hard_mean93_101_20261030.zip'
    assert not p.exists()
    with zipfile.ZipFile(ROOT/'submissions/93_submit.zip') as a,zipfile.ZipFile(ROOT/'submissions/101_submit.zip') as b:
        sa,sb=(pickle.loads(z.read('model/stats.pkl')) for z in (a,b))
        for key in ('cs_map','flag_p','type_p','ko_pairs','alias_rows'):
            assert json.dumps(sa[key],sort_keys=True,default=str)==json.dumps(sb[key],sort_keys=True,default=str),key
        for name in ('model/opus_lib.py','model/ko_translit_argon.py'):
            assert a.read(name)==b.read(name),name
        script=a.read('script.py').decode()
        needle='    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}'
        script=script.replace(needle,'    g2alt = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("g2alt_*.txt"))]\n    assert len(g2alt) == len(models["ranker"]) == 6\n'+needle)
        assert script.count('        preds.append(ix.ids[best])')==1
        script=script.replace('        preds.append(ix.ids[best])',BLOCK+'        preds.append(ix.ids[best])')
        compile(script,'script.py','exec')
        with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
            for n in a.infolist():
                if n.filename!='script.py':z.writestr(n,a.read(n))
            for k in range(6):z.writestr(f'model/g2alt_{k}.txt',b.read(f'model/ranker_{k}.txt'))
            z.writestr('script.py',script)
        with zipfile.ZipFile(p) as z:
            assert all(z.read(n)==a.read(n) for n in a.namelist() if n!='script.py')
    manifest=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
        base93_sha256=hashlib.sha256((ROOT/'submissions/93_submit.zip').read_bytes()).hexdigest(),
        alt101_sha256=hashlib.sha256((ROOT/'submissions/101_submit.zip').read_bytes()).hexdigest(),
        gate_expression_byte_equal_to102=True,existing93_payload_preserved=True,
        intervention='Only Korean mean93 margin <4: equal-weight mean of six93 and six101 rankers on unchanged candidates/features',
        status='provisional; requires matched fold2 evaluation and QA before recommendation')
    (OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
