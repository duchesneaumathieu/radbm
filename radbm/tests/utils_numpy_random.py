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