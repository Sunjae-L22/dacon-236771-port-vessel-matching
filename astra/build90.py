"""Author: Astra. Fix 88's Korean dictionary training/inference mismatch on portable 89."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / 'submissions/90_submit.zip'
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(ROOT / 'submissions/89_submit.zip') as z:
        items = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
    script = items['script.py'].decode().replace('Astra89:', 'Astra90:')
    needle = 'import opus_lib as L\n'
    assert script.count(needle) == 1
    # Both legacy libraries inspect this variable only at module import.
    script = script.replace(needle, "os.environ['KO_DICT']='1'\n" + needle)
    needle = "    koix=A.Index(catalog,new['fit'])\n    koix.ko=L.KA.KoMatcher(catalog,new['ko_pairs'])\n"
    assert script.count(needle) == 1
    replacement = '''    # The Korean ranker was trained with the universal dictionary disabled.
    # Each KoMatcher copies this global only at construction. The legacy English
    # matcher's learned dictionaries are already independent instance values.
    universal=L.KA._UNIVERSAL_KO_SUBWORDS
    try:
        L.KA._UNIVERSAL_KO_SUBWORDS=[]
        koix=A.Index(catalog,new['fit'])
        koix.ko=L.KA.KoMatcher(catalog,new['ko_pairs'])
    finally:
        L.KA._UNIVERSAL_KO_SUBWORDS=universal
'''
    script = script.replace(needle, replacement)
    compile(script, 'script.py', 'exec')
    items['script.py'] = script.encode()
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, contents in sorted(items.items()):
            z.writestr(name, contents)
    with zipfile.ZipFile(ROOT / 'submissions/89_submit.zip') as old, zipfile.ZipFile(destination) as new:
        assert new.testzip() is None
        assert all(old.read(n) == new.read(n) for n in old.namelist() if n != 'script.py')
    manifest = {
        'submission': 90,
        'base': 89,
        'original_design': 88,
        'change': 'Explicit legacy dictionary ON at import; scoped universal dictionary OFF while constructing new Korean matcher',
        'reason': '88 set KO_DICT=0 after module import; matcher did not read it, so inference used ON while training used OFF',
        'models_changed': False,
        'all_89_model_files_byte_identical': True,
        'korean_predictions_may_change_from_88_89': True,
        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'bytes': destination.stat().st_size,
        'training_feature_parity': 'PENDING',
        'english_parity': 'PENDING',
        'local_check': 'PENDING',
        'public_lb': None,
    }
    (ROOT / 'astra/build90_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
