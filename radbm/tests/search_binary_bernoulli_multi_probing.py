import unittest, torch
import numpy as np
from radbm.search.binary import BernoulliMultiProbing

class TestBernoulliMultiProbing(unittest.TestCase):
    def test_bernoulli_multi_probing(self):
        docs_probs = torch.tensor([
            [.3, .1, .8], #[0, 0, 1], [1, 0, 1], [0, 0, 0], ...
            [.7, .9, .2], #[1, 1, 0], [0, 1, 0], [1, 1, 1], ...
            [.3, .2, .1], #[0, 0, 0], [1, 0, 0], [0, 1, 0], ...
        ])
        qurs_probs = torch.tensor([
            [.1, .4, .8], #[0, 0, 1], [0, 1, 1], 000, 010, 101, 111, 100, 110
            [.1, .6, .2], #[0, 1, 0], [0, 0, 0], 011, 001, 110, 100, 111, 101
        ])
        docs_logits = torch.logit(docs_probs)
        qurs_logits = torch.logit(qurs_probs)
        
        bmp = BernoulliMultiProbing()
        bmp.insert(docs_logits[0], 0, number=1) #just run
        retrieved = bmp.search(qurs_logits[0], number=1) #just run
        
        bmp = BernoulliMultiProbing()
        bmp.batch_insert(docs_logits, range(3), number=1) #1
        retrieved = bmp.batch_search(qurs_logits, number=1)
        expected_retrieved = [{0}, set()]
        self.assertEqual(retrieved, expected_retrieved)
        retrieved = bmp.batch_search(qurs_logits, number=2)
        expected_retrieved = [{0}, {2}]
        self.assertEqual(retrieved, expected_retrieved)
        
        bmp = BernoulliMultiProbing()
        bmp.batch_insert(docs_logits, range(3), number=2) #2
        retrieved = bmp.batch_search(qurs_logits, number=1)
        expected_retrieved = [{0}, {1}]
        self.assertEqual(retrieved, expected_retrieved)
        retrieved = bmp.batch_search(qurs_logits, number=2)
        expected_retrieved = [{0}, {1,2}]
        self.assertEqual(retrieved, expected_retrieved)
        
        bmp = BernoulliMultiProbing()
        bmp.batch_insert(docs_logits, range(3), number=3) #3
        retrieved = bmp.batch_search(qurs_logits, number=1)
        expected_retrieved = [{0}, {1,2}]
        self.assertEqual(retrieved, expected_retrieved)
        retrieved = bmp.batch_search(qurs_logits, number=2)
        expected_retrieved = [{0}, {0,1,2}]
        self.assertEqual(retrieved, expected_retrieved)
            
        #same but with __init__ parameters
        bmp = BernoulliMultiProbing(insert_number=3, search_number=2)
        bmp.batch_insert(docs_logits, range(3))
        retrieved = bmp.batch_search(qurs_logits)
        expected_retrieved = [{0}, {0,1,2}]
        self.assertEqual(retrieved, expected_retrieved)
        
        #test with numpy
        bmp = BernoulliMultiProbing(insert_number=3, search_number=2)
        bmp.batch_insert(docs_logits.numpy(), range(3))
        retrieved = bmp.batch_search(qurs_logits.numpy())
        expected_retrieved = [{0}, {0,1,2}]
        self.assertEqual(retrieved, expected_retrieved)
        
        #test itersearch
        bmp = BernoulliMultiProbing(insert_number=3)
        bmp.batch_insert(docs_logits, range(3))
        #doc0: 001, 101, 000
        #doc1: 110, 010, 111
        #doc2: 000, 100, 010
        expected_search = [
            [ #first query's itersearch
                {0},   #001
                set(), #011
                {0,2}, #000
                {1,2}, #010
                {0},   #101
                {1},   #111
                {2},   #100
                {1},   #110
            ],
            [ #second query's itersearch
                {1,2}, #010
                {0,2}, #000
                set(), #011
                {0},   #001
                {1},   #110
                {2},   #100
                {1},   #111
                {0},   #101
            ]   
        ]
        search = list(map(list, bmp.batch_itersearch(qurs_logits, yield_empty=True, yield_duplicates=True)))
        self.assertEqual(search, expected_search)
        
        #no duplicates
        no_dup_expected_search = [
            [ #first query's itersearch
                {0},   #001
                set(), #011
                {2},   #000
                {1},   #010
                set(), #101
                set(), #111
                set(), #100
                set(), #110
            ],
            [ #second query's itersearch
                {1,2}, #010
                {0},   #000
                set(), #011
                set(), #001
                set(), #110
                set(), #100
                set(), #111
                set(), #101
            ]   
        ]
        search = list(map(list, bmp.batch_itersearch(qurs_logits, yield_empty=True)))
        self.assertEqual(search, no_dup_expected_search)
        
        #no empty
        search = list(bmp.itersearch(qurs_logits[0], yield_duplicates=True))
        no_empty_expected_search = [cand for cand in expected_search[0] if cand]
        self.assertEqual(search, no_empty_expected_search)
        
        #no dup and no empty
        search = list(bmp.itersearch(qurs_logits[0]))
        no_dup_no_empty_expected_search = [cand for cand in no_dup_expected_search[0] if cand]
        self.assertEqual(search, no_dup_no_empty_expected_search)
        
        #try with cost (make sure it runs)
        search = list(bmp.itersearch(qurs_logits[0], yield_cost=True, halt_cost=10))
            
        #update state
        bmp = BernoulliMultiProbing(insert_number=1, search_number=1)
        bmp.batch_insert(docs_logits, range(3))
        bmp = BernoulliMultiProbing().set_state(bmp.get_state())
        retrieved = bmp.batch_search(qurs_logits)
        self.assertEqual(retrieved, [{0}, set()])
        
        bmp.clear() #make sure it runs
        repr(bmp) #make sure it runs
        
    def assert_expected(self, d, q, expected, alternate_tags, **kwargs):
        #only meant for test_multiple_multi_bernoulli
        bmp = BernoulliMultiProbing(**kwargs)
        bmp.batch_insert(d, range(len(d)), alternate_tags=alternate_tags)
        out = bmp.search(q, alternate_tags=alternate_tags)
        self.assertEqual(out, expected)
        
    def test_multiple_multi_bernoulli(self):
        #that is a lot of bernoulli !
        
        #comments = <code>(probability*1000)
        docs_probs = torch.tensor([
            [
                [.1, .2, .3], #000(504), 001(216), 010(126), 100(56), 011(54), 101(24), 110(14), 111(6)
                [.1, .4, .8], #001(432), 011(288), 000(108), 010(72), 101(48), 111(32), 100(12), 110(8)
                [.6, .2, .5], #100(240), 101(240), 000(160), 001(160), 110(60), 111(60), 010(40), 011(40)
            ],
            [
                [.1, .3, .4], #000(378), 001(252), 010(162), 011(108), 100(42), 101(28), 110(18), 111(12)
                [.7, .1, .5], #100(315), 101(315), 000(135), 001(135), 110(35), 111(35), 010(15), 011(15)
                [.4, .9, .5], #010(270), 011(270), 110(180), 111(180), 000(30), 001(30), 100(20), 101(20)
            ],
        ])
        qurs_probs = torch.tensor([
            [.3, .8, .4], #010(336), 011(224), 110(144), 111(96), 000(84), 001(56), 100(36), 101(24)
            [.8, .7, .5], #110(280), 111(280), 100(120), 101(120), 010(70), 011(70), 000(30), 001(30)
        ])
        docs_logits = torch.logit(docs_probs)
        qurs_logits = torch.logit(qurs_probs)
        
        #alternate_tags=True, insert_number=3, search_number=2
        #inserted codes for 0 = 000, 001, 100
        #inserted codes for 1 = 000, 100, 010
        #searched codes = 010, 110
        expected_search = {1}
        self.assert_expected(docs_logits, qurs_logits, expected_search, alternate_tags=True, insert_number=3, search_number=2)
        
        #alternate_tags=True, insert_number=6, search_number=4
        #inserted codes for 0 = 000, 001, 100, 001, 011, 101
        #inserted codes for 1 = 000, 100, 010, 001, 101, 011
        #searched codes = 010, 110, 011, 111
        expected_search = {0, 1}
        self.assert_expected(docs_logits, qurs_logits, expected_search, alternate_tags=True, insert_number=6, search_number=4)
        
        #alternate_tags=False, insert_number=3, search_number=2
        #inserted codes for 0 = 000, 001, 011
        #inserted codes for 1 = 000, 100, 101
        #searched codes = 010, 110
        expected_search = set()
        self.assert_expected(docs_logits, qurs_logits, expected_search, alternate_tags=False, insert_number=3, search_number=2)
        
        #alternate_tags=False, insert_number=3, search_number=4
        #inserted codes for 0 = 000, 001, 011
        #inserted codes for 1 = 000, 100, 101
        #searched codes = 010, 110, 111, 011
        expected_search = {0}
        self.assert_expected(docs_logits, qurs_logits, expected_search, alternate_tags=False, insert_number=3, search_number=4)
        
        #alternate_tags=True, probing=align, insert_number=4 (removing last tag), search_number=4
        #inserted codes for 0 = 000, 001
        #                       001, 011
        #inserted codes for 1 = 000, 001
        #                       100, 101
        #searched codes = 010, 011
        #                 110, 111
        expected_search = set()
        self.assert_expected(docs_logits, qurs_logits, expected_search, alternate_tags=True, probing='align', insert_number=6, search_number=4)
        
    def test_bernoulli_multi_probing_parsing_errors(self):
        probs = .9*torch.rand(32, 64)+.05 #in [.05, .95] for numerical stability
        logits = torch.logit(probs)
        with self.assertRaises(ValueError):
            bmp = BernoulliMultiProbing(probing='no all or align')
        bmp = BernoulliMultiProbing()
        with self.assertRaises(ValueError): bmp.batch_insert(logits[:,None,None], range(32)) #to much ndim
        with self.assertRaises(TypeError): bmp.batch_insert(logits<.5, range(32)) #Tensor must be float
        with self.assertRaises(TypeError): bmp.batch_insert(logits.numpy()<.5, range(32)) #ndarray must be float
        with self.assertRaises(TypeError): bmp.batch_insert(tuple(logits), range(32)) #ndarray must be float
        with self.assertRaises(ValueError): bmp.batch_insert(logits, range(31)) #not same length
        bmp.batch_insert(logits, range(32)) #works