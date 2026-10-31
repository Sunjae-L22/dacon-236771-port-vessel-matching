"""Author: Astra. Experimental neural fusion on 79, independent of retrieval change 84."""
from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]

def main():
    dst=ROOT/'submissions/85_submit.zip'
    if dst.exists():raise FileExistsError(dst)
    full=json.loads((ROOT/'astra/ngram85_full/manifest.json').read_text())
    assert full['steps']==1200 and full['weight']==.1
    with zipfile.ZipFile(ROOT/'submissions/79_submit.zip') as z: items={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
    s=items['script.py'].decode()
    s=s.replace('if os.environ.get("PYTHONHASHSEED") is None:','if os.environ.get("PYTHONHASHSEED") != "0":')
    s=s.replace('import lightgbm as lgb','os.environ["OMP_NUM_THREADS"]="1"\nos.environ["OPENBLAS_NUM_THREADS"]="1"\nimport lightgbm as lgb')
    s=s.replace('import opus_lib_65 as L65','import opus_lib_65 as L65\nimport neural85 as NN\nimport torch')
    s=s.replace('    preds = []','    neural = NN.Encoder().eval()\n    neural.load_state_dict(torch.load(HERE / "model" / "neural85.pt", map_location="cpu", weights_only=True))\n    catalog_vectors = NN.vectors(neural, ix.names_raw)\n    preds = []')
    s=s.replace('m.predict(F)','m.predict(F, num_threads=1)').replace('m.predict(F65)','m.predict(F65, num_threads=1)')
    old='        preds.append(ix.ids[best])'
    assert s.count(old)==1
    s=s.replace(old,'''        if L.is_hangul(q):
            short = [j for j in order[:20] if idx[j] not in train_targets]
            if len(short) > 1:
                with torch.inference_mode():
                    query_vector = neural([str(row.get("vessel_name") or "")])[0].numpy()
                neural_score = (catalog_vectors[[idx[j] for j in short]] * query_vector).sum(axis=1)
                base_score = np.asarray([score[j] for j in short], dtype=np.float64)
                base_z = (base_score-base_score.mean()) / max(float(base_score.std()), 1e-6)
                neural_z = (neural_score-neural_score.mean()) / max(float(neural_score.std()), 1e-6)
                fused = base_z + 0.1 * neural_z
                winner = min(range(len(short)), key=lambda k: (-float(fused[k]), idx[short[k]]))
                best = idx[short[winner]]
        preds.append(ix.ids[best])''')
    items['script.py']=s.encode();items['model/neural85.py']=(ROOT/'astra/ngram_encoder84.py').read_bytes();items['model/neural85.pt']=(ROOT/'astra/ngram85_full/best.pt').read_bytes()
    with zipfile.ZipFile(dst,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(items.items()):z.writestr(n,b)
    (ROOT/'astra/build85_manifest.json').write_text(json.dumps(dict(base='79_submit.zip',change='Korean-only character/jamo contrastive fusion; fixed tune weight 0.1',sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size,local_check='PENDING',public_lb=None,evidence='small 52-vessel pilot; confidence interval includes zero; experimental'),indent=2))
    print(dst,dst.stat().st_size)

if __name__=='__main__':main()
