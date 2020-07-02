import unittest
import numpy as np
from radbm.retrieval.hashing import MultiBernoulliHashTables

class TestMultiBernoulliHashTables(unittest.TestCase):
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
        
    def test_batch_insert(self):
        mbht = MultiBernoulliHashTables(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        
    def test_bucket_stats(self):
        mbht = MultiBernoulliHashTables(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        repr(mbht) #make sure it runs
        self.assertEqual(mbht.get_buckets_avg_size(), [1,1,1])
        self.assertEqual(mbht.get_buckets_max_size(), [1,1,1])
        
    def test_batch_itersearch(self):
        expected0 = [
            {0}, #with [0, 0, 1] -> {0}, {}, {},
                 #with [0, 1, 1] -> {}, {}, {},
        ]

        expected1 = [
            {1}, {2}, #with [0, 1, 0] -> {}, {1}, {2},
            {2}, {0}, #with [0, 0, 0] -> {2}, {}, {0},
        ]
        
        mbht = MultiBernoulliHashTables(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        result0 = list(mbht.batch_itersearch(self.qurs_log_probs[0], nlookups=2))
        result1 = list(mbht.batch_itersearch(self.qurs_log_probs[1], nlookups=2))
        self.assertEqual(result0, expected0)
        self.assertEqual(result1, expected1)
        
    def test_batch_search(self):
        expected = [
            {0}, #for query 0
            {0, 1, 2}, #for query 1
        ]
        mbht = MultiBernoulliHashTables(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        result = mbht.batch_search(self.qurs_log_probs, nlookups=2)
        self.assertEqual(result, expected)
        
    def test_state(self):
        mbht = MultiBernoulliHashTables(3, 2)
        mbht.batch_insert(self.docs_log_probs, self.indexes)
        mbht_state = mbht.get_state()
        
        mbht_copy = MultiBernoulliHashTables(3, 2).set_state(mbht_state)
        self.assertEqual(mbht.tables, mbht_copy.tables)