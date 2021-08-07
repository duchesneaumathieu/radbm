import unittest
from radbm.utils.generators import sorted_merge

class TestSortedMerge(unittest.TestCase):
    def assert_sorted_merge(self, *gens):
        data = [[v for v in g] for g in gens]
        rebuild = [[] for _ in range(len(gens))]
        sorted_data = []
        for k, v in sorted_merge(*gens):
            sorted_data.append(v)
            rebuild[k].append(v)
        self.assertEqual(sorted_data, sorted(sorted_data)) #make sure it is actually sorted
        self.assertEqual(rebuild, data)
        
        gens = [list(reversed(g)) for g in gens]
        data = [[v for v in g] for g in gens]
        rebuild = [[] for _ in range(len(gens))]
        sorted_data = []
        for k, v in sorted_merge(*gens, key=lambda x: -x):
            sorted_data.append(v)
            rebuild[k].append(v)
        self.assertEqual(sorted_data, sorted(sorted_data, reverse=True))
        self.assertEqual(rebuild, data)
    
    def test_sorted_merge(self):
        cases = [
            [range(2, 10), range(-4, 8), range(6, 9)],
            [range(6, 10), [], range(3, 7)],
            [[], [3,3,3,3], range(3, 7), range(2, 8)],
            [[], [], []]
        ]
        for case in cases:
            self.assert_sorted_merge(*case)