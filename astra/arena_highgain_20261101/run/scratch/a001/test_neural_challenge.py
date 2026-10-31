import sys
sys.dont_write_bytecode = True
import copy
import unittest
from neural_challenge import challenge, shortlist, install_capture, clean_row, route

class TestChallenge(unittest.TestCase):
    def setUp(self):
        self.catalog = {k: {'vessel_name': k, 'gross_tonnage': 100, 'flag_name': 'X', 'vessel_type': 'cargo', 'target_vessel_id': 'POISON'} for k in ['A', 'B', 'C']}
        self.packet = {'sample_id': 's1', 'row': {'vessel_name': '알파', 'target_vessel_id': 'POISON'}, 'gap110': 1.0,
                       'votes': {'ko': 'A', 'en': 'B', 'hs': 'C'}, 'llm_beams': ['Alpha'],
                       'pool': [{'id': k, 'raw': v, 'adjusted': v, 'sim_key': v, 'sim_llm': v} for k,v in [('A',.8),('B',.15),('C',.05)]]}
        self.seen = []
        def view(row, docs, beams):
            self.seen.append((row, docs, beams))
            return [{'A': 0, 'B': 3, 'C': -2}[d['vessel_name']] for d in docs]
        self.views = {'qwen:evidence': view, 'encoder:phonetic': view}
    def runone(self, p=None, excluded=frozenset(), views=None):
        return challenge(p or self.packet, self.catalog, excluded, views or self.views)
    def test_positive_cap_unlock(self):
        r = self.runone()
        self.assertEqual((r['baseline'],r['prediction'],r['reason']), ('A','B','accepted'))
    def test_positive_below20(self):
        catalog = {f'V{i:02}': {'vessel_name': f'V{i:02}'} for i in range(25)}
        p = copy.deepcopy(self.packet);p['votes'] = {'ko':'V00','en':'V24','hs':'V23'}
        p['pool'] = [{'id':f'V{i:02}', 'raw': 25-i, 'adjusted':25-i, 'sim_key':25-i, 'sim_llm':25-i} for i in range(25)]
        fn = lambda row, docs, beams: [5 if x['vessel_name']=='V24' else 0 for x in docs]
        r = challenge(p,catalog,set(), {k:fn for k in self.views})
        self.assertEqual(r['prediction'],'V24');self.assertIn('voter:en',r['sources']['V24'])
    def test_raw_shortlist_rescues_nonvoter_below20(self):
        catalog={f'V{i:02}':{'vessel_name':f'V{i:02}'} for i in range(25)}
        p=copy.deepcopy(self.packet);p['votes']={'ko':'V00','en':'V01','hs':'V02'}
        p['pool']=[{'id':f'V{i:02}','raw':i+1,'adjusted':25-i,'sim_key':0,'sim_llm':0} for i in range(25)]
        fn=lambda r,d,b:[5 if x['vessel_name']=='V24' else 0 for x in d]
        result=challenge(p,catalog,set(),{k:fn for k in self.views})
        self.assertEqual(result['prediction'],'V24')
        self.assertEqual(result['sources']['V24'],['raw'])
    def test_agreement_can_harm_explicit_counterexample(self):
        # Offline assigned truth A: native agreement alone does not guarantee safety.
        result=self.runone()
        offline_truth='A'
        self.assertEqual(result['baseline'],offline_truth)
        self.assertNotEqual(result['prediction'],offline_truth)
    def test_disagreement_abstains(self):
        views = dict(self.views, **{'encoder:phonetic': lambda r,d,b: [3 if x['vessel_name']=='C' else 0 for x in d]})
        self.assertEqual(self.runone(views=views)['reason'],'view_disagreement')
    def test_margin_abstains(self):
        fn = lambda r,d,b: [.2 if x['vessel_name']=='B' else 0 for x in d]
        self.assertEqual(self.runone(views={k:fn for k in self.views})['reason'],'native_gap_below_1')
    def test_scope(self):
        p = copy.deepcopy(self.packet);p['gap110']=4
        self.assertEqual(self.runone(p)['reason'],'protected110')
        p = copy.deepcopy(self.packet);p['row']['vessel_name']='Alpha'
        self.assertEqual(self.runone(p)['reason'],'english')
        p = copy.deepcopy(self.packet);p['votes']['hs']='B'
        self.assertEqual(self.runone(p)['reason'],'router_locked_en_hs');self.assertEqual(len(self.seen),0)
    def test_exclusions(self):
        r = self.runone(excluded={'B'});self.assertNotIn('B',r['shortlist']);self.assertEqual(r['prediction'],'A')
    def test_unknown_or_missing_voter_aborts(self):
        p=copy.deepcopy(self.packet);del p['votes']['hs']
        with self.assertRaises(ValueError): self.runone(p)
        p=copy.deepcopy(self.packet);p['votes']['hs']='BAD'
        with self.assertRaises(ValueError): self.runone(p)
    def test_every_scope_rejects_invalid_official_voters(self):
        for scope in ('english','protected','locked'):
            for role in ('ko','en','hs'):
                p=copy.deepcopy(self.packet)
                if scope=='english':p['row']['vessel_name']='Alpha'
                if scope=='protected':p['gap110']=4
                if scope=='locked':p['votes']['en']=p['votes']['hs']='B'
                p['votes'][role]='NO_SUCH_CATALOG_VESSEL'
                with self.assertRaises(ValueError):self.runone(p)
        p=copy.deepcopy(self.packet);p['votes']['en']=p['votes']['hs']='NO_SUCH_CATALOG_VESSEL'
        with self.assertRaises(ValueError):self.runone(p)
        self.assertEqual(len(self.seen),0)
    def test_failed_score_aborts(self):
        for fn in [lambda r,d,b: [float('nan')]*len(d), lambda r,d,b: []]:
            with self.assertRaises(ValueError):self.runone(views={k:fn for k in self.views})
    def test_labels_not_used(self):
        r=self.runone();p=copy.deepcopy(self.packet);p['row']['target_vessel_id']='B';p['truth']='B';p['target']='C'
        self.assertEqual(r,self.runone(p))
        for row, docs, beams in self.seen:
            self.assertNotIn('target_vessel_id',row);self.assertTrue(all('target_vessel_id' not in d for d in docs))
    def test_order_subset_invariance(self):
        packets=[]
        for i in range(8):
            p=copy.deepcopy(self.packet);p['sample_id']=str(i);packets.append(p)
        expected={p['sample_id']:self.runone(p) for p in packets}
        actual={p['sample_id']:self.runone(p) for p in reversed(packets)}
        subset={p['sample_id']:self.runone(p) for p in packets[::3]}
        self.assertEqual(expected,actual);self.assertTrue(all(v==expected[k] for k,v in subset.items()))
        p=copy.deepcopy(self.packet);p['pool'].reverse();self.assertEqual(self.runone(),self.runone(p))
    def test_duplicate_pool_aborts(self):
        p=copy.deepcopy(self.packet);p['pool'].append(p['pool'][0])
        with self.assertRaises(ValueError):self.runone(p)
    def test_scoring_exception_not_swallowed(self):
        def bad(*args):raise RuntimeError('GPU OOM')
        with self.assertRaises(RuntimeError):self.runone(views={k:bad for k in self.views})
    def test_candidate_absent_can_enter_from_existing_voter(self):
        p=copy.deepcopy(self.packet);p['pool']=[x for x in p['pool'] if x['id']!='B']
        self.assertEqual(self.runone(p)['prediction'],'B')

if __name__ == '__main__':unittest.main(verbosity=2)
