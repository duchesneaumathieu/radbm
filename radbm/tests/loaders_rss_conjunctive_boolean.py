import numpy as np
import unittest, torch
from scipy.special import comb
from radbm.loaders.rss import ConjunctiveBooleanRSS
from radbm.utils.numpy.logical import issubset, issubset_product_with_trie

ConjunctiveBooleanRSS.ENUMERATE_MAX = 1000 #overwriting to accelerate tests. 

class TestConjunctiveBooleanRSS(unittest.TestCase):
    def test_init_errors(self):
        #k > l
        k, l, m, n = 10, 10, 1000, 100
        ConjunctiveBooleanRSS(k, l, m, n) #ok!
        with self.assertRaises(ValueError):
            ConjunctiveBooleanRSS(k+1, l, m, n) #not ok!
            
        #n > comb(m, l)
        k, l, m = 1, 2, 10
        ConjunctiveBooleanRSS(k, l, m, comb(m, l)) #ok!
        with self.assertRaises(ValueError):
            ConjunctiveBooleanRSS(k, l, m, comb(m, l)+1) #not ok!
        
        #comb(m, l) >= ConjunctiveBooleanRSS.ENUMERATE_MAX and n >= comb(m, l)/2
        k, l, m = 2, 3, 15
        n = comb(m, l+1)//2 #682
        ConjunctiveBooleanRSS(k, l, m, comb(m, l)) #ok since comb(15, 3) == 455 < int(1e3)
        ConjunctiveBooleanRSS(k, l+1, m, n) #ok since n == 682 < 682.5 == comb(m, l+1)/2
        with self.assertRaises(ValueError):
            #comb(15, 4) = 1365 > ConjunctiveBooleanRSS.ENUMERATE_MAX==1000
            #n+1 == 683 >= 682.5 == comb(m, l+1)/2
            ConjunctiveBooleanRSS(k, l+1, m, n+1) # not ok
            
    def assert_types_and_shapes(self, q, d, bs, k, l, data_type):
        self.assertIsInstance(q, data_type)
        self.assertIsInstance(d, data_type)
        if data_type==np.ndarray:
            self.assertTrue(q.dtype == np.int64)
            self.assertTrue(d.dtype == np.int64)
        else:
            self.assertTrue(q.dtype == torch.int64)
            self.assertTrue(d.dtype == torch.int64)
        self.assertEqual(q.shape, (bs, k))
        self.assertEqual(d.shape, (bs, l))
    
    def assert_r_list(self, r, bs):
        #comes from rss.iter_queries
        self.assertIsInstance(r, list)
        self.assertEqual(len(r), bs)
        
    def assert_r_array(self, r, bs, block, data_type):
        #comes from rss.batch
        self.assertIsInstance(r, data_type)
        if block:
            self.assertEqual(r.shape, (bs, bs))
        else:
            self.assertEqual(r.shape, (bs,))
            
    def assert_batch_iter_types_and_shapes(self, rss, bs, k, l, block, data_type):
        q, d, r = rss.batch(bs)
        self.assert_types_and_shapes(q, d, bs, k, l, data_type)
        self.assert_r_array(r, bs, block, data_type)
        
        d, _ = next(iter(rss.iter_documents(bs)))
        q, r = next(iter(rss.iter_queries(bs)))
        self.assert_types_and_shapes(q, d, bs, k, l, data_type)
        self.assert_r_list(r, bs)
        
    def test_modes_and_backends(self):
        k, l, m, n = 4, 8, 10000, 100
        #balanced, numpy (default)
        rss = ConjunctiveBooleanRSS(k, l, m, n)
        self.assert_batch_iter_types_and_shapes(rss, 32, k, l, False, np.ndarray)
        
        #block, numpy
        rss = ConjunctiveBooleanRSS(k, l, m, n, mode='block')
        self.assert_batch_iter_types_and_shapes(rss, 17, k, l, True, np.ndarray)
        
        #balanced, torch
        rss = ConjunctiveBooleanRSS(k, l, m, n, backend='torch')
        self.assert_batch_iter_types_and_shapes(rss, 39, k, l, False, torch.Tensor)
        
        #block, torch
        rss = ConjunctiveBooleanRSS(k, l, m, n, mode='block', backend='torch')
        self.assert_batch_iter_types_and_shapes(rss, 42, k, l, True, torch.Tensor)
    
    def test_iterators(self):
        bs = 23
        k, l, m, n = 4, 8, 10000, 123
        rss = ConjunctiveBooleanRSS(k, l, m, n)
        expected_shapes = 5*[bs] + [8]
        qshapes = [len(q) for q, r in rss.iter_queries(bs)]
        dshapes = [len(d) for d, i in rss.iter_documents(bs)]
        self.assertEqual(expected_shapes, qshapes)
        self.assertEqual(expected_shapes, dshapes)
        
        queries, relevants = next(iter(rss.iter_queries(bs)))
        documents = np.concatenate([d for d, i in rss.iter_documents(bs)])
        sub = issubset_product_with_trie(queries, documents)
        self.assertEqual(sub, relevants)
        
        rss.train()
        with self.assertRaises(RuntimeError):
            for documents, indexes in rss.iter_documents(bs):
                rss.valid()
                
        rss.train()
        with self.assertRaises(RuntimeError):
            for queries, relevants in rss.iter_queries(bs):
                rss.valid()
                
    def test_probs(self):
        rng = np.random.RandomState(0xcafe)
        k, l, m, n = 2, 64, 100, 1000
        rss = ConjunctiveBooleanRSS(k, l, m, n, rng=rng)
        prob_estimate = np.mean([len(r)/n for r in rss.relevants])
        self.assertAlmostEqual(prob_estimate, rss.get_relation_prob(), places=3)
        
        #test log prob
        klmn_list = [
            (12, 50, 100, 123),
            (4, 8, 10000, 10000),
            (2, 128, 10000, 1000),
            (19, 20, 21, 10),
        ]
        for k, l, m, n in klmn_list:
            rss = ConjunctiveBooleanRSS(k, l, m, n)
            self.assertAlmostEqual(rss.get_relation_prob(), np.exp(rss.get_relation_log_prob()))
    
    def assert_reproducible_batch(self, mode):
        k, l, m, n = 4, 8, 10000, 100
        
        rng1 = np.random.RandomState(0xcafe)
        rss1 = ConjunctiveBooleanRSS(k, l, m, n, mode=mode, rng=rng1)
        
        rng2 = np.random.RandomState(0xcafe)
        rss2 = ConjunctiveBooleanRSS(k, l, m, n, mode=mode, rng=rng2)
        
        for i in range(100):
            q1, d1, r1 = rss1.batch(32)
            q2, d2, r2 = rss2.batch(32)

            np.testing.assert_allclose(q1, q2)
            np.testing.assert_allclose(d1, d2)
            np.testing.assert_allclose(r1, r2)
            
    def test_unbalanced_batch(self):
        bs = 256
        k, l, m, n = 4, 8, 10000, 123
        rss = ConjunctiveBooleanRSS(k, l, m, n, mode='unbalanced')
        q, d, r = rss.batch(bs, n_positives=16)
        self.assertEqual(r.sum(), 16)
        self.assertEqual(len(r), 256)
        r_ = issubset(q, d)
        self.assertTrue(np.allclose(r, r_))
            
    def test_unbalanced_missing_kwarg_error(self):
        bs = 23
        k, l, m, n = 4, 8, 10000, 123
        rss = ConjunctiveBooleanRSS(k, l, m, n, mode='unbalanced')
        with self.assertRaises(ValueError):
            rss.batch(bs)
    
    def test_reproducibility(self):
        self.assert_reproducible_batch(mode='balanced')
        self.assert_reproducible_batch(mode='block')
        
        k, l, m, n = 4, 8, 10000, 100
        rng1 = np.random.RandomState(0xcafe)
        rss1 = ConjunctiveBooleanRSS(k, l, m, n, rng=rng1)
        
        rng2 = np.random.RandomState(0xcafe)
        rss2 = ConjunctiveBooleanRSS(k, l, m, n, rng=rng2)
        
        np.testing.assert_allclose(rss1.qterms.data, rss2.qterms.data)
        np.testing.assert_allclose(rss1.dterms.data, rss2.dterms.data)
        self.assertEqual(rss1.relevants, rss2.relevants)