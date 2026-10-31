"""Author: Astra. Canonical NFC query boundary on the validated submission 90."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    dst = ROOT / 'submissions/91_submit.zip'
    if dst.exists():
        raise FileExistsError(dst)
    with zipfile.ZipFile(ROOT / 'submissions/90_submit.zip') as z:
        items = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
    script = items['script.py'].decode().replace('Astra90:', 'Astra91:')
    needle = "        q=unicodedata.normalize('NFC',str(row.get('vessel_name') or ''))\n"
    assert script.count(needle) == 1
    script = script.replace(needle, needle + "        row['vessel_name']=q\n")
    compile(script, 'script.py', 'exec')
    items['script.py'] = script.encode()
    with zipfile.ZipFile(dst, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(items.items()):
            z.writestr(name, data)
    with zipfile.ZipFile(ROOT / 'submissions/90_submit.zip') as old, zipfile.ZipFile(dst) as new:
        assert new.testzip() is None
        assert all(old.read(name) == new.read(name) for name in old.namelist() if name != 'script.py')
    report = {
        'submission': 91,
        'base': 90,
        'original_design': 88,
        'change': 'Assign NFC-normalized vessel_name to the current row before candidate generation and features',
        'reason': 'Extra candidate search treated decomposed NFD Korean as non-Korean while base search already normalized NFC',
        'models_changed': False,
        'all_90_model_files_byte_identical': True,
        'sha256': hashlib.sha256(dst.read_bytes()).hexdigest(),
        'bytes': dst.stat().st_size,
        'nfc_prediction_parity': 'PENDING',
        'nfd_prediction_invariance': 'PENDING',
        'local_check': 'PENDING',
        'public_lb': None,
    }
    (ROOT / 'astra/build91_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
