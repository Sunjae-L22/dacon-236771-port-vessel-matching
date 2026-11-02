import unittest,math,tempfile,json,csv,sys
from pathlib import Path
from channel_model import Channel,decide

P={'operation_probabilities':{'delete':.4,'dup':.3,'swap':.3},'step_probabilities':{'0':.6,'1':.3,'2':.1}}
class Safety(unittest.TestCase):
 def test_guard_and_finite(self):
  tr={'winner':'v2','score_mass_share':1.,'winner_has_deletion_only_path':False}
  self.assertEqual(decide({'vessel_name':'ABC'},'v1','',tr)[0],'v1')
  self.assertEqual(decide({'vessel_name':'가나다'},'v1',4,tr)[0],'v1')
  self.assertEqual(decide({'vessel_name':'가나다'},'v1',3.99,tr)[0],'v2')
  for x in (float('nan'),float('inf'),-1,1.01):
   with self.assertRaises(ValueError):decide({'vessel_name':'가나다'},'v1',1,dict(tr,score_mass_share=x))
 def test_exclusion_and_no_state(self):
  cat=[{'vessel_id':'v1','vessel_name':'A'},{'vessel_id':'v2','vessel_name':'B'}];table=[{'vessel_name':'A','ko0':'가나다'},{'vessel_name':'B','ko0':'라마바'}]
  m=Channel(cat,table,P,{'v1'});self.assertNotEqual(m.query('가나다')['winner'],'v1')
  before=m.query('라마바');m.query('가나다');self.assertEqual(before,m.query('라마바'))
 def test_deletion_ambiguity_veto(self):
  cat=[{'vessel_id':'v1','vessel_name':'A'}];table=[{'vessel_name':'A','ko0':'가나다라'}];m=Channel(cat,table,P)
  for q in ('가나다라','가다라','가라'):
   tr=m.query(q);self.assertTrue(tr['winner_has_deletion_only_path']);self.assertEqual(decide({'vessel_name':q},'baseline',0,tr)[0],'baseline')
  tr=m.query('가다나라');self.assertFalse(tr['winner_has_deletion_only_path']);self.assertEqual(decide({'vessel_name':'가다나라'},'baseline',0,tr)[0],'v1')
 def test_parameter_schema(self):
  bad={**P,'operation_probabilities':{'delete':.3,'dup':.3,'swap':.3,'extra':.1}}
  with self.assertRaises(ValueError):Channel([],[],bad)
if __name__=='__main__':unittest.main()
