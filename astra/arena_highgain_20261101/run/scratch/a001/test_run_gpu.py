"""Execute the real CLI lifecycle with mocked devices and in-memory files; no GPU claims."""
import sys
sys.dont_write_bytecode=True
import io
import json
import hashlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import run_gpu

class Lifecycle(unittest.TestCase):
    def execute(self, failure=None, invalid_voter=False):
        clock=[0.0]; present=set(); writes={}
        source=Path('/Users/sunjae/Desktop/항만/astra/arena_20261101/baseline/model/ko/model/src/encoder_views.py').read_text()
        spec={'top_k':20,'weight':.25,'views':[{'kind':'qwen','recipe':'evidence','dir':'qwen','weight':1.},
              {'kind':'encoder','recipe':'phonetic','dir':'encoder','weight':1.}],
              'source_hashes':{'encoder_views.py':hashlib.sha256(source.encode()).hexdigest()}}
        packet={'sample_id':'s','row':{'vessel_name':'알파'},'votes':{'en':'A','ko':'A','hs':'A'},'gap110':4,'pool':[]}
        if invalid_voter:packet['votes']['en']='BAD'
        data={'root/encoder_views.json':json.dumps(spec),'root/src/encoder_views.py':source,
              'root/train_vessels.json':'[]','packets':json.dumps(packet)+'\n','catalog':'catalog'}
        if failure=='source':data['root/src/encoder_views.py']='bad'
        class MemoryPath:
            def __init__(self,s):self.s=s
            def __str__(self):return self.s
            def __truediv__(self,v):return MemoryPath(self.s+'/'+v)
            def resolve(self):return self
            @property
            def suffix(self):return '.jsonl' if self.s=='output.jsonl' else ''
            @property
            def parent(self):return MemoryPath('parent')
            def exists(self):return self.s in present
            def with_suffix(self,v):return MemoryPath('output'+v)
            def mkdir(self,**kw):pass
            def read_text(self):return data[self.s]
            def read_bytes(self):return self.read_text().encode()
            def write_text(self,s):writes[self.s]=json.loads(s);present.add(self.s)
            def open(self,*a,**kw):present.add(self.s);return io.StringIO()
            def rename(self,target):present.discard(self.s);present.add(str(target))
        class Frame:
            def to_dict(self,*a):return [{'vessel_id':'A','vessel_name':'Alpha'}]
            def __len__(self):return 1
        class View:
            def __init__(self,folder,kind,recipe):
                self.name=kind+':'+recipe
                if failure=='oom':raise RuntimeError('CUDA out of memory during frozen ViewScorer load')
                if failure=='timeout':clock[0]=601
            def score(self,*a):raise AssertionError('protected scope must not score')
        torch=types.ModuleType('torch');torch.__version__='mock';torch.version=types.SimpleNamespace(cuda='mock')
        torch.set_num_threads=lambda n:None
        torch.cuda=types.SimpleNamespace(is_available=lambda:failure!='cuda',synchronize=lambda:None,
                 get_device_name=lambda:'mock',max_memory_allocated=lambda:0)
        pd=types.ModuleType('pandas');pd.read_csv=lambda *a,**kw:Frame()
        module=types.ModuleType('src.encoder_views');module.ViewScorer=View
        args=types.SimpleNamespace(model_dir=MemoryPath('root'),packets=MemoryPath('packets'),catalog=MemoryPath('catalog'),
                                 output=MemoryPath('output.jsonl'),baseline_seconds=1080)
        error=None
        with patch.object(run_gpu.argparse.ArgumentParser,'parse_args',return_value=args),patch.object(run_gpu.time,'monotonic',side_effect=lambda:clock[0]),patch.dict(sys.modules,{'torch':torch,'pandas':pd,'src.encoder_views':module}):
            try:run_gpu.main()
            except Exception as e:error=e
        return error,writes['output.jsonl.status.json'],present
    def test_model_load_oom_status(self):
        e,status,files=self.execute('oom')
        self.assertIsInstance(e,RuntimeError);self.assertEqual(status['state'],'ABORTED')
        self.assertEqual(status['stage'],'load_model:qwen');self.assertFalse(status['partial_present'])
        self.assertFalse(status['completed_output_present']);self.assertIn('out of memory',status['error'])
    def test_source_hash_failure_status(self):
        e,status,_=self.execute('source');self.assertEqual(status['state'],'ABORTED')
        self.assertEqual(status['stage'],'verify_sources');self.assertFalse(status['partial_present'])
    def test_cuda_absent_status(self):
        e,status,_=self.execute('cuda');self.assertEqual(status['state'],'ABORTED')
        self.assertIn('actual GPU',status['error']);self.assertFalse(status['partial_present'])
    def test_protected_bad_voter_abort(self):
        e,status,_=self.execute(invalid_voter=True);self.assertEqual(status['state'],'ABORTED')
        self.assertIn('official catalog',status['error']);self.assertFalse(status['completed_output_present'])
    def test_model_load_deadline_even_without_scoring(self):
        e,status,_=self.execute('timeout');self.assertIsInstance(e,TimeoutError)
        self.assertEqual(status['state'],'ABORTED');self.assertEqual(status['seconds'],601)
        self.assertFalse(status['completed_output_present'])
    def test_protected_valid_complete(self):
        e,status,_=self.execute();self.assertIsNone(e);self.assertEqual(status['state'],'COMPLETE')
        self.assertEqual(status['changed'],0);self.assertTrue(status['completed_output_present'])

if __name__=='__main__':unittest.main(verbosity=2)
