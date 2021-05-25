import unittest, torch
from itertools import product
from radbm.search import HammingMultiProbing
from radbm.search.reduction import HammingReduction

class TestHammingReduction(unittest.TestCase):
    def test_hamming_reduction(self):
        fq = torch.nn.Identity()
        fd = torch.nn.Identity()
        reduction = HammingReduction(fq, fd, HammingMultiProbing())
        
        documents = 2*torch.rand(32, 8) - 1 #in [-1, 1]
        indexes = range(32)
        queries = 2*torch.rand(32, 8) - 1
        rdoc = documents > 0
        rque = queries > 0
        
        hmp = HammingMultiProbing()
        for insert_param, search_param in product(range(3), range(3)):
            reduction.clear(); hmp.clear()
            reduction.batch_insert(documents, indexes, insert_param)
            reduction = HammingReduction(fq, fd, HammingMultiProbing()).set_state(reduction.get_state())
            reduced_search = reduction.batch_search(queries, search_param)
            reduced_itersearch = list(reduction.itersearch(queries[0], yield_cost=True, halt_cost=128.)) #max cost is 255 in this case
        
            hmp.batch_insert(rdoc, indexes, insert_param)
            original_search = hmp.batch_search(rque, search_param)
            original_itersearch = list(hmp.itersearch(rque[0], yield_cost=True, halt_cost=128.))
        
            self.assertEqual(reduced_search, original_search)
            self.assertEqual(reduced_itersearch, original_itersearch)