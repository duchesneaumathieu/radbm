import unittest, torch
from radbm.metrics.spatial import superset_cost

class TestSupersetCost(unittest.TestCase):
    def test_superset_cost(self):
        d = int(superset_cost(
            torch.tensor([True, False, True, True]),
            torch.tensor([False, False, False, False]),
            dim=-1
        ))
        expected = 3
        self.assertEqual(expected, d)
        
        x = torch.randint(0, 2, (5,6,10), dtype=bool)
        y = torch.randint(0, 2, (5,6,10), dtype=bool)
        d = superset_cost(x, y, dim=-1)
        expected = torch.zeros((5,6), dtype=int)
        for i in range(5):
            for j in range(6):
                x_set = set(torch.where(x[i, j])[0].numpy().tolist())
                y_set = set(torch.where(y[i, j])[0].numpy().tolist())
                len(x_set - (x_set.intersection(y_set)))
                expected[i, j] = len(x_set - (x_set.intersection(y_set)))
        self.assertTrue(torch.equal(expected, d))