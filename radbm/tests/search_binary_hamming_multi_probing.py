import unittest, torch
from radbm.search.binary import HammingMultiProbing

class TestHammingMultiProbing(unittest.TestCase):
    def test_hamming_multi_probing(self):
        bmp = HammingMultiProbing(probing='align')
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
            bmp = HammingMultiProbing(insert_radius=insert_radius, search_radius=search_radius, probing='align')
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
        bmp = HammingMultiProbing(insert_radius=1, search_radius=1, probing='align')
        bmp.batch_insert(data, range(4))
        bmp = HammingMultiProbing(probing='align').set_state(bmp.get_state())
        retrieved = bmp.batch_search(data)
        self.assertEqual(retrieved, expected_retrieved)
        
        repr(bmp) #make sure it runs
        
    def test_hamming_multi_probing_itersearch(self):
        bmp = HammingMultiProbing(insert_radius=1, probing='align')
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
            
    def test_multiple_hamming_multi_probing(self):
        with self.assertRaises(ValueError):
            HammingMultiProbing(probing='this not a valid value.')
        
        bmp = HammingMultiProbing(probing='align')
        data = torch.tensor(
            [[(0, 0, 0, 0), (1, 0, 1, 0)],
            [(1, 1, 1, 1), (0, 1, 0, 1)],
            [(0, 0, 1, 1), (1, 1, 0, 0)]],
            dtype=bool
        )
        
        with self.assertRaises(ValueError): #ndim==4 error
            bmp.batch_insert(data[None], range(3), radius=0)
        
        with self.assertRaises(ValueError): #ndim==4 error
            bmp.batch_search(data[None], radius=1)
        
        bmp.batch_insert(data, range(3), radius=0)
        retrieved = bmp.batch_search(data, radius=1) #they are at a minimum of radius 2
        expected_retrieved = [{0}, {1}, {2}]
        self.assertEqual(retrieved, expected_retrieved)
        
        #distance two only
        expected_retrieved = [{0, 2}, {1, 2}, {0, 1, 2}]
        for insert_radius, search_radius in [(1, 1), (2, 0), (0, 2)]:
            bmp.clear()
            bmp.batch_insert(data, range(3), radius=insert_radius)
            retrieved = bmp.batch_search(data, radius=search_radius)
            self.assertEqual(retrieved, expected_retrieved)
        
        #distance two only with probing='all'
        bmp = HammingMultiProbing(probing='all')
        expected_retrieved = [{0, 1, 2}, {0, 1, 2}, {0, 1, 2}]
        for insert_radius, search_radius in [(1, 1), (2, 0), (0, 2)]:
            bmp.clear()
            bmp.batch_insert(data, range(3), radius=insert_radius)
            retrieved = bmp.batch_search(data, radius=search_radius)
            self.assertEqual(retrieved, expected_retrieved)
        
        
    def test_multiple_hamming_multi_probing_itersearch(self):
        query = torch.tensor([(0, 1, 1, 1), (0, 0, 0, 1)], dtype=bool)
        documents = torch.tensor( #radius 1 ball
            [[
                (0, 0, 0, 0), #1000, 0100, 0010, 0001
                (0, 0, 1, 0), #1010, 0110, 0000, 0011
            ],
            [
                (1, 1, 1, 1), #0111, 1011, 1101, 1110
                (1, 0, 1, 1), #0011, 1111, 1001, 1010
            ],
            [
                (1, 0, 1, 1), #0011, 1111, 1001, 1010
                (1, 0, 0, 0), #0000, 1100, 1010, 1001
            ]],
            dtype=bool
        )
        align_expected_search = [
            ({1}, 1.), #k0#0 (d0) : 0111 []
            (set(), 2.), #k1#0 (d0) : 0001 []
            ({1,2}, 3.), #k0#1 (d1) : 1111 [0]
            ({1,2}, 4.), #k1#1 (d1) : 1001 [0]
            ({2}, 5.), #k0#2 (d1) : 0011 [1]
            (set(), 6.), #k1#2 (d1) : 0101 [1]
            (set(), 7.), #k0#3 (d1) : 0101 [2]
            ({0,1}, 8.), #k1#3 (d1) : 0011 [2]
            (set(), 9.), #k0#4 (d1) : 0110 [3]
            ({0,2}, 10.), #k1#4 (d1) : 0000 [3]
            ({1,2}, 11.), #k0#5 (d2) : 1011 [0,1]
            (set(), 12.), #k1#5 (d2) : 1101 [0,1]
            ({1}, 13.), #k0#6 (d2) : 1101 [0,2]
            ({1}, 14.), #k1#6 (d2) : 1011 [0,2]
            ({1}, 15.), #k0#7 (d2) : 1110 [0,3]
            ({2}, 16.), #k1#7 (d2) : 1000 [0,3]
            ({0}, 17.), #k0#8 (d2) : 0001 [1,2]
            (set(), 18.), #k1#8 (d2) : 0111 [1,2]
            ({0}, 19.), #k0#9 (d2) : 0010 [1,3]
            (set(), 20.), #k1#9 (d2) : 0100 [1,3]
            ({0}, 21.), #k0#10 (d2) : 0100 [2,3]
            ({0}, 22.), #k1#10 (d2) : 0010 [2,3]
            ({2}, 23.), #k0#11 (d3) : 1001 [0,1,2]
            ({1}, 24.), #k1#11 (d3) : 1111 [0,1,2]
            ({2}, 25.), #k0#12 (d3) : 1010 [0,1,3]
            ({2}, 26.), #k1#12 (d3) : 1100 [0,1,3]
            (set(), 27.), #k0#13 (d3) : 1100 [0,2,3]
            ({0,1,2}, 28.), #k1#13 (d3) : 1010 [0,2,3]
            ({0}, 29.), #k0#14 (d3) : 0000 [1,2,3]
            ({0}, 30.), #k1#14 (d3) : 0110 [1,2,3]
            ({0}, 31.), #k0#15 (d4) : 1000 [0,1,2,3]
            (set(), 32.), #k1#15 (d4) : 0001 [0,1,2,3]
        ]
        
        all_expected_search = [
            ({1}, 1.), #k0#0 (d0) : 0111
            ({0}, 2.), #k1#0 (d0) : 0001
            ({1,2}, 3.), #k0#1 (d1) : 1111
            ({1,2}, 4.), #k1#1 (d1) : 1001
            ({0,1,2}, 5.), #k0#2 (d1) : 0011
            (set(), 6.), #k1#2 (d1) : 0101
            (set(), 7.), #k0#3 (d1) : 0101
            ({0,1,2}, 8.), #k1#3 (d1) : 0011
            ({0}, 9.), #k0#4 (d1) : 0110
            ({0,2}, 10.), #k1#4 (d1) : 0000
            ({1,2}, 11.), #k0#5 (d2) : 1011
            ({1}, 12.), #k1#5 (d2) : 1101
            ({1}, 13.), #k0#6 (d2) : 1101
            ({1,2}, 14.), #k1#6 (d2) : 1011
            ({1}, 15.), #k0#7 (d2) : 1110
            ({0,2}, 16.), #k1#7 (d2) : 1000 [0,3]
            ({0}, 17.), #k0#8 (d2) : 0001 [1,2]
            ({1}, 18.), #k1#8 (d2) : 0111 [1,2]
            ({0}, 19.), #k0#9 (d2) : 0010 [1,3]
            ({0}, 20.), #k1#9 (d2) : 0100 [1,3]
            ({0}, 21.), #k0#10 (d2) : 0100 [2,3]
            ({0}, 22.), #k1#10 (d2) : 0010 [2,3]
            ({1,2}, 23.), #k0#11 (d3) : 1001 [0,1,2]
            ({1,2}, 24.), #k1#11 (d3) : 1111 [0,1,2]
            ({0,1,2}, 25.), #k0#12 (d3) : 1010 [0,1,3]
            ({2}, 26.), #k1#12 (d3) : 1100 [0,1,3]
            ({2}, 27.), #k0#13 (d3) : 1100 [0,2,3]
            ({0,1,2}, 28.), #k1#13 (d3) : 1010 [0,2,3]
            ({0,2}, 29.), #k0#14 (d3) : 0000 [1,2,3]
            ({0}, 30.), #k1#14 (d3) : 0110 [1,2,3]
            ({0,2}, 31.), #k0#15 (d4) : 1000 [0,1,2,3]
            ({1}, 32.), #k1#15 (d4) : 1110 [0,1,2,3]
        ]
        
        bmp = HammingMultiProbing(insert_radius=1, probing='align')
        bmp.batch_insert(documents, range(3))
        search = list(bmp.itersearch(query, yield_cost=True, yield_empty=True, yield_duplicates=True))
        self.assertEqual(search, align_expected_search)
        
        bmp = HammingMultiProbing(insert_radius=1, probing='all')
        bmp.batch_insert(documents, range(3))
        search = list(bmp.itersearch(query, yield_cost=True, yield_empty=True, yield_duplicates=True))
        self.assertEqual(search, all_expected_search)