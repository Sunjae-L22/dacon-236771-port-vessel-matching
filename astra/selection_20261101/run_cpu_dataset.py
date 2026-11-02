from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];source=ROOT/'astra/new_direction_20261101/local_cvote/run_cpu.py';s=source.read_text();a="dest=HERE/'eval'";assert s.count(a)==1;s=s.replace(a,"dest=Path(os.environ['ASTRA_EVAL_DIR'])");s=s.replace("'rows':400","'rows':sum(1 for _ in (dest/'test.csv').open())-1")
exec(compile(s,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
