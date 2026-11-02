from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
source=ROOT/'astra/new_direction_20261101/local_cvote/run_v20.py'
sys.path.insert(0,str(HERE));text=source.read_text();old='m.run(test_path=test,catalog_path=ROOT/\'data/open/vessel_catalog.csv\',output_path=out)'
assert text.count(old)==1
text=text.replace(old,"from capture_hooks import install\ninstall(Path(os.environ['ASTRA_CAPTURE_DIR']))\n"+old)
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
