import unittest, torch
import numpy as np
from radbm.metrics.spatial import superset_cost
from radbm.search.superset.priority_superset_trie_search import PrioritySupersetTrieSearch, limitrophes_suffix

class TestPrioritySupersetTrieSearch(unittest.TestCase):
    def test_priority_superset_trie_search(self):
        rng = np.random.RandomState(0xcafe)
        dcodes = torch.tensor(rng.randint(0, 2, (100, 32)), dtype=bool)
        q = torch.tensor(rng.uniform(-5, 5, (10, 32)))
        qcodes = q > 2

        dsets = [torch.where(code)[0].tolist() for code in dcodes]

        struct = PrioritySupersetTrieSearch(search_type='dfs')
        struct.batch_insert(dsets, range(len(dcodes)))

        expected = [set(torch.where(cost==0)[0].tolist()) for cost in superset_cost(qcodes[:,None], dcodes[None,:])]
        output = list(map(list, struct.batch_itersearch(q)))
        for x, out in zip(q, output):
            values = x.sort()[0]
            partials = [set(map(int, torch.where(superset_cost((x>v)[None], dcodes)==0)[0])) for v in values]
            for partial in partials:
                self.assertEqual(partial, set().union(*out[:len(partial)]))

        #test halt_cost
        list(struct.itersearch(q[0], halt_cost=1))
        
        #test search
        struct.search(q[0], halt_cost=1)
        
    def test_priority_superset_trie_search_error(self):
        with self.assertRaises(ValueError):
            PrioritySupersetTrieSearch(search_type='wrong')
            
    def test_limitrophes_suffix(self):
        self.assertEqual(limitrophes_suffix((), (1,2,3)), ())
        self.assertEqual(limitrophes_suffix((1,3,5,6), (4,)), (1,3))
        self.assertEqual(limitrophes_suffix((1,3,5,6), (5,6)), (1,3))
        self.assertEqual(limitrophes_suffix((1,3,5,6), (0,6)), (1,3,5))