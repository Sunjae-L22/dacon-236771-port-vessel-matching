"""Build an UNREVIEWED candidate; original110 payloads remain byte identical."""
from pathlib import Path
import zipfile,hashlib,ast,json

def replace_once(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)

def patched_script(original):
    s=replace_once(original,'import unicodedata','import unicodedata\nimport json, subprocess, tempfile')
    s=replace_once(s,'    preds = []','    preds = []\n    neural_packets = []')
    s=replace_once(s,'    for row in test.to_dict("records"):', '    for row_number, row in enumerate(test.to_dict("records")):')
    s=replace_once(s,'        preds.append(ix.ids[best])', '''        preds.append(ix.ids[best])
        if L.is_hangul(q):
            allowed_order = [j for j in order if idx[j] not in train_targets] or order
            denom = len(models["ranker"])
            gap = (float(score[allowed_order[0]]) - float(score[allowed_order[1]])) / denom if len(allowed_order) > 1 else 99.0
            if gap < 4:
                candidates = [dict(index=int(idx[j]), id=str(ix.ids[idx[j]]), name=str(catalog.iloc[idx[j]].vessel_name), score=float(score[j]) / denom) for j in allowed_order[:100]]
                assert candidates[0]["id"] == str(ix.ids[best])
                neural_packets.append(dict(row_number=row_number, name=q, baseline=str(ix.ids[best]), gap=gap, candidates=candidates))''')
    s=replace_once(s,'    out = Path(out)', '''    original_preds = list(preds)
    neural_results = []
    if neural_packets:
        with tempfile.TemporaryDirectory(prefix="astra_byt5_") as temp:
            temp = Path(temp)
            inp, result = temp / "packets.json", temp / "results.json"
            inp.write_text(json.dumps(neural_packets, ensure_ascii=False))
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
            subprocess.run([sys.executable, str(HERE / "model" / "worker.py"), str(HERE / "model" / "byt5"), str(inp), str(result)], env=env, check=True)
            neural_results = json.loads(result.read_text())
            assert len(neural_results) == len(neural_packets)
            for packet, r in zip(neural_packets, neural_results):
                assert packet["row_number"] == r["row_number"] and packet["baseline"] == r["baseline"]
                assert r["pred"] in {c["id"] for c in packet["candidates"]}
                assert r["eligible"] or r["pred"] == r["baseline"]
                preds[r["row_number"]] = r["pred"]
    if os.environ.get("ASTRA_TRACE"):
        detail = {r["row_number"]: r for r in neural_results}
        trace = []
        for i, (r, old, new) in enumerate(zip(test.to_dict("records"), original_preds, preds)):
            name = unicodedata.normalize("NFC", str(r.get("vessel_name") or ""))
            trace.append(dict(sample_id=str(r["sample_id"]), baseline=str(old), pred=str(new), is_ko=bool(L.is_hangul(name)), eligible=detail.get(i, {}).get("eligible", False), gap=detail.get(i, {}).get("gap")))
        Path(os.environ["ASTRA_TRACE"] + "." + str(len(test)) + ".json").write_text(json.dumps(trace, ensure_ascii=False))
    out = Path(out)''')
    ast.parse(s);return s

def build(base,model,out):
    base,model,out=Path(base),Path(model),Path(out);here=Path(__file__).resolve().parent
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    assert (model/'config.json').exists() and list(model.glob('*.safetensors'))
    with zipfile.ZipFile(base) as src,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as dest:
        for n in src.namelist():dest.writestr(n,patched_script(src.read(n).decode()).encode() if n=='script.py' else src.read(n))
        for n in ['neural.py','worker.py']:dest.write(here/n,'model/'+n)
        for p in sorted(model.iterdir()):
            if p.is_file():dest.write(p,'model/byt5/'+p.name)
    unchanged=[]
    with zipfile.ZipFile(base) as src,zipfile.ZipFile(out) as dest:
        for n in src.namelist():
            if n=='script.py':continue
            assert src.read(n)==dest.read(n);unchanged.append(n)
    report=dict(zip=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,original_payloads_unchanged=unchanged,status='UNREVIEWED_NOT_FOR_SUBMISSION')
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    import sys
    build(*sys.argv[1:])
