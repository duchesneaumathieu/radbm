import unittest, torch
from radbm.search import HammingMultiProbing

class TestHammingMultiProbing(unittest.TestCase):
    def test_hamming_multi_probing(self):
        bmp = HammingMultiProbing()
        data = torch.tensor(
            [(0, 0, 0, 0),
             (1, 1, 1, 1),
             (0, 0, 1, 1),
             (1, 0, 1, 0)],
            dtype=bool
        )
        bmp.batch_insert(data, range(4), radius=0)
        retrieved = bmp.batch_search(data, radius=1) #they are at a minimum of radius 2
        expected_retrieved = [{0}, {1}, {2}, {3}]
        self.assertEqual(retrieved, expected_retrieved)
        
        #distance two only
        expected_retrieved = [{0,2,3}, {1,2,3}, {0,1,2,3}, {0,1,2,3}]
        for insert_radius, search_radius in [(1, 1), (2, 0), (0, 2)]:
            bmp.clear()
            bmp.batch_insert(data, range(4), radius=insert_radius)
            retrieved = bmp.batch_search(data, radius=search_radius)
            self.assertEqual(retrieved, expected_retrieved)
            
        #same but with __init__ parameters
        for insert_radius, search_radius in [(1, 1), (2, 0), (0, 2)]:
            bmp = HammingMultiProbing(insert_radius=insert_radius, search_radius=search_radius)
            bmp.batch_insert(data, range(4))
            retrieved = bmp.batch_search(data)
            self.assertEqual(retrieved, expected_retrieved)
            
        #type error
        bmp.clear()
        with self.assertRaises(TypeError):
            bmp.batch_insert(data.float(), range(4))
            
        with self.assertRaises(TypeError):
            bmp.batch_search(data.float())
            
        #update state
        bmp = HammingMultiProbing(insert_radius=1, search_radius=1)
        bmp.batch_insert(data, range(4))
        bmp = HammingMultiProbing().set_state(bmp.get_state())
        retrieved = bmp.batch_search(data)
        self.assertEqual(retrieved, expected_retrieved)
        
        repr(bmp) #make sure it runs
        
    def test_hamming_multi_probing_itersearch(self):
        bmp = HammingMultiProbing(insert_radius=1)
        documents = torch.tensor( #radius 1 ball
            [(0, 0, 0, 0), #1000, 0100, 0010, 0001
             (1, 1, 1, 1), #0111, 1011, 1101, 1110
             (1, 0, 1, 1), #0011, 1111, 1001, 1010
             (1, 0, 0, 0)],#0000, 1100, 1010, 1001
            dtype=bool
        )
        query = torch.tensor((0, 1, 1, 1), dtype=bool) #3, 1, 2, 4
        bmp.batch_insert(documents, range(4))
        search = list(bmp.itersearch(query, yield_cost=True, yield_empty=True, yield_duplicates=True))
        expected_search = [ #should retrieved every doc 5 (size of the ball) times in the whole search
            ({1}, 1.), #0111 d0
            ({1, 2}, 2.), #1111 d1
            ({2}, 3.), #0011 d1
            (set(), 4.), #0101 d1
            (set(), 5.), #0110 d1
            ({1,2}, 6.), #1011 d2
            ({1}, 7.), #1101 d2
            ({1}, 8.), #1110 d2
            ({0}, 9.), #0001 d2
            ({0}, 10.), #0010 d2
            ({0}, 11.), #0100 d2
            ({2,3}, 12.), #1001 d3
            ({2,3}, 13.), #1010 d3
            ({3}, 14.), #1100 d3
            ({0,3}, 15.), #0000 d3
            ({0,3}, 16.), #1000 d4
        ]
        self.assertEqual(search, expected_search)
        
        #no duplicates
        search = list(bmp.itersearch(query, yield_cost=True, yield_empty=True))
        no_dup_expected_search = [
            ({1}, 1.), #0111 d0
            ({2}, 2.), #1111 d1
            (set(), 3.), #0011 d1
            (set(), 4.), #0101 d1
            (set(), 5.), #0110 d1
            (set(), 6.), #1011 d2
            (set(), 7.), #1101 d2
            (set(), 8.), #1110 d2
            ({0}, 9.), #0001 d2
            (set(), 10.), #0010 d2
            (set(), 11.), #0100 d2
            ({3}, 12.), #1001 d3
            (set(), 13.), #1010 d3
            (set(), 14.), #1100 d3
            (set(), 15.), #0000 d3
            (set(), 16.), #1000 d4
        ]
        self.assertEqual(search, no_dup_expected_search)
        
        #no empty
        search = list(bmp.itersearch(query, yield_cost=True, yield_duplicates=True))
        no_empty_expected_search = [(cand, cost) for cand, cost in expected_search if cand]
        self.assertEqual(search, no_empty_expected_search)
        
        #no dup and no empty
        search = list(bmp.itersearch(query, yield_cost=True))
        no_dup_no_empty_expected_search = [(cand, cost) for cand, cost in no_dup_expected_search if cand]
        self.assertEqual(search, no_dup_no_empty_expected_search)
        
        #no dup, no empty, and no cost
        search = list(bmp.itersearch(query))
        no_all_expected_search = [cand for cand, cost in no_dup_no_empty_expected_search]
        self.assertEqual(search, no_all_expected_search)
        
        #cost limit (no all)
        for halt_cost in [cost for cand, cost in expected_search]:
            search = list(bmp.itersearch(query, halt_cost=halt_cost))
            halt_no_all_expected_search = [cand for cand, cost in no_dup_no_empty_expected_search if cost <= halt_cost]
            self.assertEqual(search, halt_no_all_expected_search)