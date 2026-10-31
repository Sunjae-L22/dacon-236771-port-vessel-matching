"""Author: Astra. Preserve submission 88 predictions without Arrow-backed pickles.

The inference payload contains only Python builtins; pandas frames are rebuilt
with object strings and explicit NumPy numeric dtypes in the server environment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'astra/build89_manifest.json'


def primitive(value):
    import numpy as np
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f'Non-portable scalar: {type(value)}')


def pack_frame(frame):
    import pandas as pd
    return {
        'columns': list(frame.columns),
        'data': [[primitive(v) for v in row] for row in frame.to_numpy(dtype=object).tolist()],
        'numeric_dtypes': {c: str(t) for c, t in frame.dtypes.items() if pd.api.types.is_numeric_dtype(t)},
    }


def restore_frame(payload):
    import pandas as pd
    frame = pd.DataFrame(payload['data'], columns=payload['columns'], dtype=object)
    for column, dtype in payload['numeric_dtypes'].items():
        frame[column] = frame[column].astype(dtype)
    return frame


def main():
    import pandas as pd
    source = ROOT / 'submissions/88_submit.zip'
    target = ROOT / 'submissions/89_submit.zip'
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        items = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
    original = pickle.loads(items.pop('model/ko_stats88.pkl'))
    portable = {name: pack_frame(frame) for name, frame in original.items()}
    blob = pickle.dumps(portable, protocol=4)
    # A builtin-only payload uses no GLOBAL / STACK_GLOBAL pickle opcodes.
    import pickletools
    globals_used = [op.name for op, _, _ in pickletools.genops(blob) if op.name in ('GLOBAL', 'STACK_GLOBAL')]
    assert not globals_used, globals_used
    assert b'pyarrow' not in blob and b'pandas' not in blob
    for name, original_frame in original.items():
        restored = restore_frame(pickle.loads(blob)[name])
        pd.testing.assert_frame_equal(original_frame, restored, check_dtype=False)
        assert all(str(dtype) == 'object' or pd.api.types.is_numeric_dtype(dtype) for dtype in restored.dtypes)
    items['model/ko_stats89.pkl'] = blob
    script = items['script.py'].decode().replace('Astra88:', 'Astra89:').replace('ko_stats88.pkl', 'ko_stats89.pkl')
    needle = "    with (HERE/'model/ko_stats89.pkl').open('rb') as f:new=pickle.load(f)\n"
    assert script.count(needle) == 1
    restoration = '''    for name, payload in list(new.items()):
        frame=pd.DataFrame(payload['data'],columns=payload['columns'],dtype=object)
        for column,dtype in payload['numeric_dtypes'].items():
            frame[column]=frame[column].astype(dtype)
        new[name]=frame
'''
    script = script.replace(needle, needle + restoration)
    compile(script, 'script.py', 'exec')
    items['script.py'] = script.encode()
    with zipfile.ZipFile(ROOT / 'submissions/79_submit.zip') as z:
        assert all(items[name] == z.read(name) for name in z.namelist() if name.startswith('model/') and not name.endswith('/'))
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, contents in sorted(items.items()):
            z.writestr(name, contents)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    report = {
        'submission': 89,
        'base': 88,
        'reason': '88 ko_stats pickle used pandas ArrowStringArray and required unavailable pyarrow',
        'change': 'Only serialize fit/ko_pairs as Python builtin payload; restore object strings and numeric dtypes at inference',
        'models_changed': False,
        'all_79_model_files_byte_identical': True,
        'original_stats_shapes': {k: list(v.shape) for k, v in original.items()},
        'roundtrip_all_values_and_missing_values_equal': True,
        'portable_pickle_global_opcodes': globals_used,
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'bytes': target.stat().st_size,
        'local_check': 'PENDING',
        'public_lb': None,
    }
    MANIFEST.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
