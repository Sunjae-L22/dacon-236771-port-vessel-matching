"""Author: Astra. Export the same trained neural model without torch at inference."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,zipfile,hashlib,unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import ngram_encoder84 as PT
import neural_numpy86 as NN

def main():
    dst=ROOT/'submissions/86_submit.zip'
    if dst.exists():raise FileExistsError(dst)
    out=ROOT/'astra/ngram86_numpy';out.mkdir(exist_ok=True)
    weights=torch.load(ROOT/'astra/ngram85_full/best.pt',map_location='cpu',weights_only=True)
    np.savez_compressed(out/'weights.npz',**{k:v.numpy() for k,v in weights.items()})
    m=NN.Encoder(out/'weights.npz');ref=PT.Encoder().eval();ref.load_state_dict(weights)
    cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv');tr=pd.read_csv(ROOT/'data/open/train.csv')
    names=cat.vessel_name.fillna('').astype(str).tolist()
    probe=names[::79]+tr.vessel_name.fillna('').astype(str).tolist()[::12]
    probe += [unicodedata.normalize('NFD',s) for s in probe[-100:]]
    reference=PT.vectors(ref,probe);actual=m(probe)
    error=float(np.max(np.abs(reference-actual)))
    assert error<1e-5,error
    np.save(out/'catalog_vectors.npy',m(names),allow_pickle=False)
    with zipfile.ZipFile(ROOT/'submissions/85_submit.zip') as z:items={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
    s=items['script.py'].decode().replace('import neural85 as NN\nimport torch','import neural86 as NN')
    s=s.replace('    neural = NN.Encoder().eval()\n    neural.load_state_dict(torch.load(HERE / "model" / "neural85.pt", map_location="cpu", weights_only=True))\n    catalog_vectors = NN.vectors(neural, ix.names_raw)', '    neural = NN.Encoder(HERE / "model" / "neural86.npz")\n    catalog_vectors = np.load(HERE / "model" / "catalog_vectors86.npy", allow_pickle=False)\n    assert hashlib.sha256("\\n".join(ix.ids).encode()).hexdigest() == "'+hashlib.sha256('\n'.join(cat.vessel_id).encode()).hexdigest()+'"')
    s=s.replace('import os','import os\nimport hashlib',1)
    s=s.replace('                with torch.inference_mode():\n                    query_vector = neural([str(row.get("vessel_name") or "")])[0].numpy()','                query_vector = neural([str(row.get("vessel_name") or "")])[0]')
    assert 'torch' not in s and 'neural85' not in s
    items.pop('model/neural85.py');items.pop('model/neural85.pt')
    items['script.py']=s.encode();items['model/neural86.py']=(ROOT/'astra/neural_numpy86.py').read_bytes()
    items['model/neural86.npz']=(out/'weights.npz').read_bytes();items['model/catalog_vectors86.npy']=(out/'catalog_vectors.npy').read_bytes()
    with zipfile.ZipFile(dst,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(items.items()):z.writestr(n,b)
    receipt=dict(base='79_submit.zip',replaces='85_submit.zip (local torch embedding_bag segmentation fault)',change='same trained neural fusion, NumPy inference, catalog-only precomputed vectors',sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size,local_check='PENDING',public_lb=None,export_probe_names=len(probe),max_abs_vector_error=error,evidence='52-vessel pilot; confidence interval includes zero; experimental')
    (ROOT/'astra/build86_manifest.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
