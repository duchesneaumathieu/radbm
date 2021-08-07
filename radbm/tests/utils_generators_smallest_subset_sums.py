import unittest
import numpy as np
from itertools import islice
from radbm.utils.generators import smallest_subset_sums
ISLICE_MAX = 10000

class TestSmallestSubsetSums(unittest.TestCase):
    def test_smallest_subset_sums(self):
        rng = np.random.RandomState(0xcafe)
        values = rng.uniform(0, 100, (1000,))
        subsets = list(islice(smallest_subset_sums(values), ISLICE_MAX))
        sums = [sum(values[i] for i in subset) for subset in subsets]
        
        #assert unique
        subsets_tuple = [tuple(subset) for subset in subsets]
        self.assertEqual(len(subsets_tuple), len(set(subsets_tuple)))
        
        #assert increasing order
        self.assertTrue(np.all(0<=np.diff(sums)))
        self.assertEqual(len(sums), ISLICE_MAX)
        
        #test yield_sums
        subsets, yielded_sums = zip(*islice(smallest_subset_sums(values, yield_sums=True), ISLICE_MAX))
        self.assertEqual(sums, list(yielded_sums))
        
        #test yield_stats
        self.assertEqual(4, len(next(smallest_subset_sums(values, yield_stats=True))))
        self.assertEqual(5, len(next(smallest_subset_sums(values, yield_sums=True, yield_stats=True))))
        
        #test negative value error
        values[42] = -1/12
        with self.assertRaises(ValueError):
            next(smallest_subset_sums(values))