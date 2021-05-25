import unittest, torch
from radbm.metrics import (
    hamming_distance,
    pre_average_precision,
    pre_mean_average_precision,
    batch_pre_average_precision,
)

def _ap(scores, relevants):
    assert relevants.dtype is torch.int64
    rank = torch.argsort(scores)
    relevants_rank = torch.tensor(sorted([torch.where(rank==i)[0][0] for i in relevants]))+1 #slow
    ks = torch.arange(1, len(relevants)+1)
    ap = (ks/relevants_rank).mean()
    return ap

class TestPreAP(unittest.TestCase):
    def test_pre_average_precision_small1(self):
        #example 1 (equivalence with the AP when unique scores)
        scores = torch.tensor([20, 30, 10, 50, 60, 80, 90, 70, 40, 100])
        #       documents'id:  0   1    2   3   4   5   6   7   8   9
        # total order 2 < 0 < 1 < 8 < 3 < 4 < 7 < 5 < 6 < 9
        relevants = torch.tensor([1, 2, 7])
        #we get 1/3 recall (we find 2) at rank 1 -> 1 precision
        #we get 2/3 recall (we find 1) at rank 3 -> 2/3 precision
        #we get 3/3 recall (we find 7) at rank 7 -> 3/7 precision
        #thus the AP is 1/3 + 2/3/3 + 3/7/3 = 3/9 + 2/9 + 1/7 = 5/9 + 1/7 = 35/63 + 9/63 = 44/63
        expected_ap = torch.tensor(44/63)
        
        ap = _ap(scores, relevants)
        pre_ap = pre_average_precision(scores, relevants)
        self.assertEqual(expected_ap, ap)
        self.assertEqual(expected_ap, pre_ap)
        
    def test_pre_average_precision_small2(self):
        #example 2 (taken from TestUCK)
        scores = torch.tensor([60, 20, 70, 40, 50, 50, 30, 70, 50, 40]) #the score of each document
        #       documents'id:   0,  1,  2,  3,  4,  5,  6,  7,  8,  9
        #thus the total pre-order is: 1 < 6 < 3,9 < 4,5,8 <  0 < 2,7
        #                         t = 0   1    2      3      4    5
        relevants = torch.tensor([8, 3, 5])
        #we know that the UCk is torch.tensor([7/2, 16/3, 20/3])
        #thus the pre-AP should be (1/(7/2) + 2/(16/3) + 3/(20/3))/3 = 2/21 + 2/16 + 3/20 = 311/840
        expected_pre_ap = 311/840
        pre_ap = float(pre_average_precision(scores, relevants))
        self.assertAlmostEqual(expected_pre_ap, pre_ap)
        
    def test_pre_mean_average_precision(self):
        #combining both examples
        scores = torch.tensor([
            [20, 30, 10, 50, 60, 80, 90, 70, 40, 100],
            [60, 20, 70, 40, 50, 50, 30, 70, 50, 40]
        ])
        relevants = [
            torch.tensor([1, 2, 7]),
            torch.tensor([8, 3, 5]),
        ]
        #the pre-MAP should be (44/63 + 311/840)/2 = 2693/5040
        expected_pre_map = 2693/5040
        pre_map = float(pre_mean_average_precision(scores, relevants))
        self.assertAlmostEqual(expected_pre_map, pre_map)
        
    def test_pre_average_precision_big(self):
        #using a total order example to compare with the AP
        scores = torch.rand(10000)
        relevants = torch.unique(torch.randint(0, 10000, (30,)))
        ap = _ap(scores, relevants)
        pre_ap = pre_average_precision(scores, relevants)
        self.assertAlmostEqual(ap, pre_ap)
        
    def test_bool_relevants(self):
        scores = torch.rand(10000)
        bool_relevants = torch.randint(0, 2, (10000,), dtype=torch.bool)
        int64_relevants = torch.where(bool_relevants)[0]
        
        bool_pre_ap = pre_average_precision(scores, bool_relevants)
        int64_pre_ap = pre_average_precision(scores, int64_relevants)
        self.assertAlmostEqual(bool_pre_ap, int64_pre_ap)
        
    def test_batch_pre_average_precision(self):
        queries = torch.randint(0, 2, (456, 64), dtype=torch.bool)
        documents = torch.randint(0, 2, (8889, 64), dtype=torch.bool)
        relevants = [torch.unique(torch.randint(0, 8889, (30,))) for i in range(456)]
        
        scoring_function = lambda q, d: hamming_distance(q, d, dim=-1)
        pre_aps = batch_pre_average_precision(queries, documents, relevants, scoring_function, batch_size=173)
        
        scores0 = hamming_distance(queries[[0]], documents, dim=-1)
        pre_ap0 = pre_average_precision(scores0, relevants[0])
        self.assertEqual(pre_ap0, pre_aps[0])
        
        scores172 = hamming_distance(queries[[172]], documents, dim=-1)
        pre_ap172 = pre_average_precision(scores172, relevants[172])
        self.assertEqual(pre_ap172, pre_aps[172])
        
        scores173 = hamming_distance(queries[[173]], documents, dim=-1)
        pre_ap173 = pre_average_precision(scores173, relevants[173])
        self.assertEqual(pre_ap173, pre_aps[173])
        
        scores455 = hamming_distance(queries[[455]], documents, dim=-1)
        pre_ap455 = pre_average_precision(scores455, relevants[455])
        self.assertEqual(pre_ap455, pre_aps[455])
        
        missing_relevants = relevants[:-1]
        with self.assertRaises(ValueError):
            batch_pre_average_precision(queries, documents, missing_relevants, scoring_function, batch_size=173)