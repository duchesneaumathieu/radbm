import unittest, torch
from radbm.metrics import subset_distance

class TestSubsetDistance(unittest.TestCase):
    def test_subset_distance(self):
        d = int(subset_distance(
            torch.tensor([True, False, True, True]),
            torch.tensor([False, False, False, False]),
            dim=-1
        ))
        expected = 3
        self.assertEqual(expected, d)
        
        x = torch.randint(0, 2, (5,6,10), dtype=bool)
        y = torch.randint(0, 2, (5,6,10), dtype=bool)
        d = subset_distance(x, y, dim=-1)
        expected = torch.zeros((5,6), dtype=int)
        for i in range(5):
            for j in range(6):
                x_set = set(torch.where(x[i, j])[0].numpy().tolist())
                y_set = set(torch.where(y[i, j])[0].numpy().tolist())
                len(x_set - (x_set.intersection(y_set)))
                expected[i, j] = len(x_set - (x_set.intersection(y_set)))
        self.assertTrue(torch.equal(expected, d))
