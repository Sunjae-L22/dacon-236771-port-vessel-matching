"""Astra: replace only hard Korean decisions on immutable110, with auditable row-local guards."""
import argparse
import copy
from common import *

BLOCK='''        before = best
        if L.is_hangul(q):
            allowed = [j for j in order if idx[j] not in train_targets] or order
            base_mean = score / len(models["ranker"])
            _, scope = choose(idx, base_mean, allowed, F60, L.FEATURES, RULE, base_mean, baseline_order=allowed)
            alternate = (sum(m.predict(F60) for m in noise_models) / len(noise_models)) if RULE.startswith("noise") and scope["eligible"] else None
            selected, audit = choose(idx, base_mean, allowed, F60, L.FEATURES, RULE, alternate, baseline_order=allowed)
            best = idx[selected]
            assert audit["baseline_index"] == before
            assert best == before or audit["eligible"]
            trace(row, "ko", ix.ids[before], ix.ids[best], audit)
        else:
            trace(row, "en", ix.ids[before], ix.ids[best], dict(hard=False, eligible=False, gap=None))
'''
TRACE='''
def trace(row, route, baseline, selected, audit):
    path = os.environ.get("ASTRA_TRACE")
    if path:
        import json
        payload = dict(sample_id=row["sample_id"], route=route, baseline=str(baseline), selected=str(selected), **audit)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\\n")

'''

def main(rule,variant=None):
    model_root=OUT/variant if variant else OUT
    tag=f'{variant}_{rule}' if variant else rule
    report=json.loads((model_root/'fold2'/f'{rule}_report.json').read_text())
    assert report['clean_harm_zero'] and report['eligible_scope_verified'],report
    assert report['noise']['net']>0,report
    confirmation=json.loads((model_root/'fold4'/f'{rule}_report.json').read_text())
    assert confirmation['eligible_for_next_stage'],confirmation
    if rule.startswith('noise'):
        for seed in SEEDS:
            p=model_root/'full'/f'seed{seed}/ranker.txt';m=lgb.Booster(model_file=str(p));assert m.num_trees()==600 and m.num_feature()==60
    source=ROOT/'submissions/110_submit.zip'
    name=f'astra110_{tag}_20261031.zip';dest=ROOT/'astra/builds'/name;assert not dest.exists()
    with zipfile.ZipFile(source) as z:
        script=z.read('script.py').decode()
        script=script.replace('import opus_lib_65 as L65','import opus_lib_65 as L65\nfrom astra_decision import choose\nRULE = '+repr(rule))
        script=script.replace('def run(test_path=',TRACE+'def run(test_path=')
        needle='    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}'
        script=script.replace(needle,'    noise_models = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("noise113_*.txt"))]\n    assert not RULE.startswith("noise") or len(noise_models) == len(models["ranker"]) == 6\n'+needle)
        needle='        preds.append(ix.ids[best])';assert script.count(needle)==1
        script=script.replace(needle,BLOCK+needle)
        script=script.replace('            preds.append(ix.ids[cs])','            trace(row, "call_sign", ix.ids[cs], ix.ids[cs], dict(hard=False, eligible=False, gap=None))\n            preds.append(ix.ids[cs])')
        compile(script,'script.py','exec')
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as w:
            for info in z.infolist():
                if info.filename!='script.py':w.writestr(copy.copy(info),z.read(info))
            w.writestr('script.py',script);w.writestr('model/astra_decision.py',(OUT/'decision.py').read_bytes())
            if rule.startswith('noise'):
                for k,seed in enumerate(SEEDS):w.write(model_root/'full'/f'seed{seed}/ranker.txt',f'model/noise113_{k}.txt')
        with zipfile.ZipFile(dest) as w:assert all(z.read(n)==w.read(n) for n in z.namelist() if n!='script.py')
    dump(OUT/f'build_{tag}.json',dict(rule=rule,model_variant=variant or 'three_extra_copies',path=str(dest),sha256=sha(dest),base110_sha256=sha(source),bytes=dest.stat().st_size,
        existing110_payload_bytes_preserved=True,clean_harm=report['clean']['harm'],noise_net=report['noise']['net'],status='requires full local_check and scope QA'))
    print(dest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rule',choices=['noise','noise_with_tonnage','name_top3','second_name']);p.add_argument('--variant',choices=['light']);a=p.parse_args();main(a.rule,a.variant)
