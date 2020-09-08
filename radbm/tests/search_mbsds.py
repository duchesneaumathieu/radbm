import unittest, torch
import numpy as np
from radbm.search.mbsds import HashingMultiBernoulliSDS

class TestHashingMultiBernoulliSDS(unittest.TestCase):
    def setUp(self):
        self.indexes = [0, 1, 2]
        self.docs_log_probs = np.log([
            [.3, .1, .8], #[0, 0, 1], [1, 0, 1], [0, 0, 0], ...
            [.7, .9, .2], #[1, 1, 0], [0, 1, 0], [1, 1, 1], ...
            [.3, .2, .1], #[0, 0, 0], [1, 0, 0], [0, 1, 0], ...
        ])
        self.qurs_log_probs = np.log([
            [.1, .4, .8], #[0, 0, 1], [0, 1, 1]
            [.1, .6, .2], #[0, 1, 0], [0, 0, 0]
        ])
        
    def test_numpy_cast(self):
        mbht = HashingMultiBernoulliSDS(3, 2)
        mbht.batch_insert(torch.tensor(self.docs_log_probs), self.indexes)
        
    def test_batch_insert(self):
        mbht = HashingMultiBernoulliSDS(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        
    def test_bucket_stats(self):
        mbht = HashingMultiBernoulliSDS(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        repr(mbht) #make sure it runs
        self.assertEqual(mbht.get_buckets_avg_size(), [1,1,1])
        self.assertEqual(mbht.get_buckets_max_size(), [1,1,1])
        
    def test_get_generator_error(self):
        ndim3 = np.zeros((1,2,3))
        mbht = HashingMultiBernoulliSDS(3, 2)
        with self.assertRaises(ValueError):
            mbht.insert(ndim3, 1)
        
    def test_itersearch(self):
        expected0 = [
            {0}, #with [0, 0, 1] -> {0}, {}, {},
                 #with [0, 1, 1] -> {}, {}, {},
        ]

        expected1 = [
            {1}, {2}, #with [0, 1, 0] -> {}, {1}, {2},
            {0}, #with [0, 0, 0] -> {2}, {}, {0}, (but 2 duplicate)
        ]
        
        mbht = HashingMultiBernoulliSDS(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        result0 = list(mbht.itersearch(self.qurs_log_probs[0], nlookups=2))
        result1 = list(mbht.itersearch(self.qurs_log_probs[1], nlookups=2))
        self.assertEqual(result0, expected0)
        self.assertEqual(result1, expected1)
        
        #given log_probs1
        log_probs1 = np.log(1-np.exp(self.qurs_log_probs))
        log_probs = np.stack([self.qurs_log_probs, log_probs1], axis=1)
        result0 = list(mbht.itersearch(self.qurs_log_probs[0], nlookups=2))
        result1 = list(mbht.itersearch(self.qurs_log_probs[1], nlookups=2))
        self.assertEqual(result0, expected0)
        self.assertEqual(result1, expected1)
        
    def test_empty_itersearch(self):
        expected0 = [
            {0}, set(), set(), #with [0, 0, 1] -> {0}, {}, {},
            set(), set(), set(),  #with [0, 1, 1] -> {}, {}, {},
        ]

        expected1 = [
            set(), {1}, {2}, #with [0, 1, 0] -> {}, {1}, {2},
            set(), set(), {0}, #with [0, 0, 0] -> {2}, {}, {0}, (but 2 duplicate -> empty)
        ]
        
        mbht = HashingMultiBernoulliSDS(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        result0 = list(mbht.itersearch(self.qurs_log_probs[0], nlookups=2, yield_empty=True))
        result1 = list(mbht.itersearch(self.qurs_log_probs[1], nlookups=2, yield_empty=True))
        self.assertEqual(result0, expected0)
        self.assertEqual(result1, expected1)
        
        #given log_probs1
        log_probs1 = np.log(1-np.exp(self.qurs_log_probs))
        log_probs = np.stack([self.qurs_log_probs, log_probs1], axis=1)
        result0 = list(mbht.itersearch(self.qurs_log_probs[0], nlookups=2, yield_empty=True))
        result1 = list(mbht.itersearch(self.qurs_log_probs[1], nlookups=2, yield_empty=True))
        self.assertEqual(result0, expected0)
        self.assertEqual(result1, expected1)
        
    def test_batch_search(self):
        expected = [
            {0}, #for query 0
            {0, 1, 2}, #for query 1
        ]
        mbht = HashingMultiBernoulliSDS(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        result = mbht.batch_search(self.qurs_log_probs, nlookups=2)
        self.assertEqual(result, expected)
        
        #given log_probs1
        log_probs1 = np.log(1-np.exp(self.qurs_log_probs))
        log_probs = np.stack([self.qurs_log_probs, log_probs1], axis=1)
        result = list(mbht.batch_search(log_probs, nlookups=2))
        self.assertEqual(result, expected)
        
    def test_state(self):
        mbht = HashingMultiBernoulliSDS(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        mbht_state = mbht.get_state()
        
        mbht_copy = HashingMultiBernoulliSDS(3, 2).set_state(mbht_state)
        self.assertEqual(mbht.tables, mbht_copy.tables)