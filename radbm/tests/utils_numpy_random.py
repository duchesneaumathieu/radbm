import unittest
import numpy as np
from radbm.utils.numpy.logical import isrepeat, issubset
from radbm.utils.numpy.random import (
    unique_randint_with_permutation,
    unique_randint_with_choice,
    unique_randint_with_shuffle,
    unique_randint_with_randint,
    fast_unique_randint,
    unique_randint,
    no_subset_unique_randint,
    uniform_n_choose_k_by_enumeration,
    uniform_n_choose_k_by_rejection,
)

class TestUniqueRandint(unittest.TestCase):
    def test_enough_int(self):
        low, high = 5, 12 #high-low == 7
        unique_randint(low, high, 1, 7) #should work
        with self.assertRaises(ValueError):
            unique_randint(low, high, 1, 8)
            
    def test_unique_randint_sanity(self):
        low, high = 123, 234
        n, k = 10000, 12
        
        for sampler in [
            unique_randint_with_permutation,
            unique_randint_with_choice,
            unique_randint_with_shuffle,
            unique_randint_with_randint,
            fast_unique_randint,
        ]:
            samples = sampler(low, high, n, k)
            self.assertEqual(samples.shape, (n, k))
            self.assertTrue((low<=samples).all())
            self.assertTrue((samples<high).all())
            self.assertFalse(isrepeat(samples).any())
            
    def test_unique_randint_branches(self):
        #fastbranch
        unique_randint(123, 234, 32, 10)
        
        #permutation branch
        unique_randint(123, 234, 32, 100)
        
        #fast_unique_randint branch
        unique_randint(0, 500_000, 32, 2000)
        
class TestNoSubsetUniqueRandint(unittest.TestCase):
    def test_no_subset_unique_randint(self):
        x = unique_randint(123, 234, 10000, 80)
        y = no_subset_unique_randint(123, 234, 10000, 4, x)
        self.assertFalse(issubset(x, y).any())
        
class TestUniformNChooseK(unittest.TestCase):
    def assertUnique(self, x):
        h = set()
        for v in x:
            tv = tuple(v)
            self.assertTrue(tv not in h)
            h.add(tv)
    
    def test_uniform_n_choose_k(self):
        n, k = 10, 5
        t = 252 #comb(10, 5)
        samples = uniform_n_choose_k_by_enumeration(n, k, t)
        self.assertEqual(samples.shape, (t, k))
        self.assertUnique(samples)
        
        samples = uniform_n_choose_k_by_rejection(n, k, t)
        self.assertEqual(samples.shape, (t, k))
        self.assertUnique(samples)
    
        with self.assertRaises(ValueError):
            uniform_n_choose_k_by_enumeration(n, k, t+1)
            
        with self.assertRaises(ValueError):
            uniform_n_choose_k_by_rejection(n, k, t+1)