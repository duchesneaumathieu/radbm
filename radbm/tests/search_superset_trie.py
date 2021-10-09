import unittest, torch
import numpy as np
from radbm.metrics.spatial import superset_cost
from radbm.search.superset import SupersetTrieSearch

class TestSupersetTrieSearch(unittest.TestCase):
    def test_superset_trie_search(self):
        rng = np.random.RandomState(0xcafe)
        dcodes = torch.tensor(rng.randint(0, 10, (1000, 64)) <= 6)
        qcodes = torch.tensor(rng.randint(0, 10, (10, 64)) <= 2)

        dsets = [torch.where(code)[0].tolist() for code in dcodes]
        qsets = [torch.where(code)[0].tolist() for code in qcodes]

        struct = SupersetTrieSearch(search_type='dfs')
        struct.batch_insert(dsets, range(len(dcodes)))

        expected = [set(torch.where(cost==0)[0].tolist()) for cost in superset_cost(qcodes[:,None], dcodes[None,:])]
        output = struct.batch_search(qsets)
        self.assertEqual(expected, output)
        
        #test halt_cost
        list(struct.itersearch(qsets[0], halt_cost=1))
        
    def test_superset_trie_search_error(self):
        with self.assertRaises(ValueError):
            SupersetTrieSearch(search_type='wrong')