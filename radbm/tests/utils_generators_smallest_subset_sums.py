import unittest
import numpy as np
from itertools import islice
from radbm.utils.generators import smallest_subset_sums

class TestSmallestSubsetSums(unittest.TestCase):
    def test_smallest_subset_sums(self):
        rng = np.random.RandomState(0xcafe)
        values = rng.uniform(0, 100, (1000,))
        subsets = list(islice(smallest_subset_sums(values), 10000))
        sums = [sum(values[i] for i in subset) for subset in subsets]
        
        #assert unique
        subsets_tuple = [tuple(subset) for subset in subsets]
        self.assertEqual(len(subsets_tuple), len(set(subsets_tuple)))
        
        #assert increasing order
        self.assertTrue(np.all(0<=np.diff(sums)))
        self.assertEqual(len(sums), 10000)
        
        #test yield_stats
        self.assertEqual(4, len(next(smallest_subset_sums(values, yield_stats=True))))
        
        #test negative value error
        values[42] = -1/12
        with self.assertRaises(ValueError):
            next(smallest_subset_sums(values))