import unittest, torch
import numpy as np
from radbm.search.radius import HammingRadiusSDS

class TestHammingRadiusSDS(unittest.TestCase):
    def test_numpy_cast(self):
        mbht = HammingRadiusSDS(64, 2)
        mbht.batch_insert(torch.zeros((10,64),dtype=torch.bool), range(10))
        
    def test_hammingradius(self):
        sds = HammingRadiusSDS(4, 2)
        #(0 in 4) + (1 in 4) + (2 in 4) = 1 + 4 + 6 = 11
        self.assertEqual(sds.masks.shape, (11, 4))
        
        documents = np.array([
            [0,0,0,0], #0
            [0,0,1,1], #1
            [0,1,0,1], #2
            [1,1,1,0], #3
        ], dtype=np.bool)
        
        queries = np.array([
            [0,0,0,0], # match with 0, 1, 2
            [1,0,0,0], # match with 0, 3
            [0,1,1,0], # match with 0, 1, 2, 3
        ], dtype=np.bool)
        
        #insert + search
        sds.batch_insert(documents, range(4))
        indexes = sds.batch_search(queries)
        expected = [{0,1,2}, {0,3}, {0,1,2,3}]
        self.assertEqual(indexes, expected)
        
        #itersearch with yield_empty=True
        indexes = list(sds.itersearch(queries[0], yield_empty=True))
        expected = [
            {0}, #radius 0
            set(), set(), set(), set(), #radius 1
            set(), set(), set(), set(), {2}, {1}, #radius 2
        ]
        self.assertEqual(indexes, expected)
        
        #itersearch with yield_empty=False
        indexes = list(sds.itersearch(queries[0], yield_empty=False))
        expected = [{0}, {2}, {1}] #same order as above
        self.assertEqual(indexes, expected)