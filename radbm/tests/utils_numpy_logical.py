import unittest
import numpy as np
from radbm.utils.numpy.logical import (
    set_isrepeat,
    vec_isrepeat,
    isrepeat,
    set_issubset,
    vec_issubset,
    issubset,
)   

class TestNumpyLogical(unittest.TestCase):
    def test_isrepeat(self):
        x = np.array([
            [2,4,5,1,2], #double 2
            [2,3,4,5,6],
            [5,3,2,1,7],
            [2,5,1,3,5], #double 5
            [2,5,8,9,4],
        ])
        expected = np.array([True, False, False, True, False])
        set_results = set_isrepeat(x)
        vec_results = vec_isrepeat(x)
        self.assertTrue(np.array_equal(expected, set_results))
        self.assertTrue(np.array_equal(expected, vec_results))
        
        #taking 12 uniform elements out of 100 has a 50% chance of having duplicate
        x = np.random.randint(0, 100, (1000, 12))
        set_results = set_isrepeat(x)
        vec_results = vec_isrepeat(x)
        self.assertTrue(np.array_equal(set_results, vec_results))
        
        isrepeat(np.random.randint(0, 100, (100, 65)))
        isrepeat(np.random.randint(0, 100, (100, 63)))
        
    def test_issubset(self):
        x = np.array([
            [2,4,5,1,2], #double 2
            [2,3,4,5,6],
            [5,3,2,1,7],
            [2,5,1,3,5], #double 5
            [2,5,8,9,4],
        ])
        y = np.array([
            [5,2,4,1,6,3], #True
            [5,1,2,3,5,6], #False missing 4
            [2,7,1,6,5,3], #True
            [2,4,1,5,6,8], #False missing 3
            [6,3,1,6,7,0], #False missing all
        ])
        expected = np.array([True, False, True, False, False])
        set_results = set_issubset(x, y)
        vec_results = vec_issubset(x, y)
        self.assertTrue(np.array_equal(expected, set_results))
        self.assertTrue(np.array_equal(expected, vec_results))
        
        n, bs = 100, 1000
        l, k = 4, 84
        x = np.array([np.random.permutation(n)[:l] for _ in range(bs)])
        y = np.array([np.random.permutation(n)[:k] for _ in range(bs)])
        set_results = set_issubset(x, y)
        vec_results = vec_issubset(x, y)
        self.assertTrue(np.array_equal(set_results, vec_results))
        
        y = np.random.randint(0, 200, (100, 190))
        issubset(np.random.randint(0, 200, (100, 129)), y)
        issubset(np.random.randint(0, 200, (100, 127)), y)