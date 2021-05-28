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
        docs_lp0 = (1-docs_probs).log()
        docs_lp1 = docs_probs.log()
        docs_logits = torch.logit(docs_probs)
        
        qurs_lp0 = (1-qurs_probs).log()
        qurs_lp1 = qurs_probs.log()
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
        
        #same but with log probs
        bmp = BernoulliMultiProbing(insert_number=3, search_number=2)
        bmp.batch_insert((docs_lp0, docs_lp1), range(3))
        retrieved = bmp.batch_search((qurs_lp0, qurs_lp1))
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
        
    def test_bernoulli_multi_probing_parsing_errors(self):
        probs = .9*torch.rand(32, 64)+.05 #in [.05, .95] for numerical stability
        lp0, lp1 = (1-probs).log(), probs.log()
        bmp = BernoulliMultiProbing()
        with self.assertRaises(TypeError): bmp.batch_insert(lp1<.5, range(32)) #Tensor must be float
        with self.assertRaises(ValueError): bmp.batch_insert(lp1[0], range(32)) #dim not 2
        with self.assertRaises(ValueError): bmp.batch_insert((lp0, lp1, lp1), range(32)) #length 3 must be 2
        with self.assertRaises(TypeError): bmp.batch_insert(('not a tensor', lp1), range(32)) #first item not a tensor 
        with self.assertRaises(TypeError): bmp.batch_insert((lp0, 'not a tensor'), range(32)) #second item not a tensor 
        with self.assertRaises(ValueError): bmp.batch_insert((lp0[0], lp1), range(32)) #bad dim first tensor
        with self.assertRaises(ValueError): bmp.batch_insert((lp0, lp1[0]), range(32)) #bad dim second tensor
        with self.assertRaises(TypeError): bmp.batch_insert((lp0<.5, lp1), range(32)) #first Tensor must be float
        with self.assertRaises(TypeError): bmp.batch_insert((lp0, lp1<.5), range(32)) #second Tensor must be float
        with self.assertRaises(TypeError): bmp.batch_insert(((lp0.exp()+1).log(), lp1), range(32)) #first tensor not negative
        with self.assertRaises(TypeError): bmp.batch_insert((lp0, (lp1.exp()+1).log()), range(32)) #first tensor not negative
        with self.assertRaises(TypeError): bmp.batch_insert('not a tensor nor a tuple', range(32)) #not a tensor or a tuple
        with self.assertRaises(ValueError): bmp.batch_insert((lp0, lp1), range(31)) #not same length
        bmp.batch_insert(torch.logit(probs), range(32)) #works
        bmp.batch_insert((lp0, lp1), range(32)) #works