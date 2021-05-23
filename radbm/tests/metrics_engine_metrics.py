import unittest, torch
from radbm.metrics.search_costs import (
    costs_at_k,
    total_cost_at_k,
    total_cost_ratio_from_costs,
    total_cost_ratio,
)

class TestTCR(unittest.TestCase):
    def test_costs_at_k(self):
        cands = [{0, 1}, {2}, {3, 4, 5}, set(), {6}, {7, 8}, {9, 10, 11}]
        costs = [    2.,  6.,        8.,    9., 11.,    15.,         25.]
        
        #test 1
        candidates = zip(cands, costs)
        relevants = {0, 4, 5, 6, 7}
        uck, eck = costs_at_k(candidates, relevants)
        expected_uck = torch.tensor([0+1*3/2, 3+1*4/3, 3+2*4/3, 6+1*1/1, 7+1*3/2])
        expected_eck = torch.tensor([2., 8., 8., 11., 15.])
        self.assertTrue(torch.allclose(uck, expected_uck))
        self.assertTrue(torch.allclose(eck, expected_eck))
        
        #test 2 (empty candidates)
        with self.assertRaises(ValueError):
            uck, eck = costs_at_k([], relevants)
        uck, eck = costs_at_k([], relevants, N=12)
        expected_uck = torch.tensor([0+i*13/6 for i in range(1, 6)])
        expected_eck = torch.tensor([0.])
        self.assertTrue(torch.allclose(uck, expected_uck))
        self.assertTrue(torch.allclose(eck, expected_eck))
        
        #test 3 (missing relevant documents)
        relevants = {0, 4, 5, 6, 7, 12, 13} #12 and 13 not in cands
        with self.assertRaises(ValueError):
            uck, eck = costs_at_k(zip(cands, costs), relevants)
        
        with self.assertRaises(RuntimeError):
            uck, eck = costs_at_k(zip(cands, costs), relevants, N=13)
        uck, eck = costs_at_k(zip(cands, costs), relevants, N=15)
        expected_uck = torch.tensor([0+1*3/2, 3+1*4/3, 3+2*4/3, 6+1*1/1, 7+1*3/2, 12+1*(15-12+1)/3, 12+2*(15-12+1)/3])
        expected_eck = torch.tensor([2., 8., 8., 11., 15., 0., 0.])
        self.assertTrue(torch.allclose(uck, expected_uck))
        self.assertTrue(torch.allclose(eck, expected_eck))
        
    def test_total_cost_at_k(self):
        cands = [{0, 1}, {2}, {3, 4, 5}, set(), {6}, {7, 8}, {9, 10, 11}]
        costs = [    2.,  6.,        8.,    9., 11.,    15.,         25.]
        relevants = {0, 4, 5, 6, 7}
        total_cost_at_k(zip(cands, costs), relevants) #just make sure it runs
        
    def test_total_cost_ratio(self):
        cands = [{0, 1}, {2}, {3, 4, 5}, set(), {6}, {7, 8}, {9, 10, 11}]
        costs = [    2.,  6.,        8.,    9., 11.,    15.,         25.]
        candidates = zip(cands, costs)
        relevants = {0, 4, 5, 6, 7, 12, 13}
        N = 15
        for N in range(14, 20):
            expected_uck = torch.tensor([0+1*3/2, 3+1*4/3, 3+2*4/3, 6+1*1/1, 7+1*3/2, 12+1*(N-12+1)/3, 12+2*(N-12+1)/3])
            expected_eck = torch.tensor([2., 8., 8., 11., 15., 0., 0.])
            expected_tck = expected_uck + expected_eck

            lr = len(relevants)
            coeffs = torch.tensor([i*(N+1)/(lr+1) for i in range(1, lr+1)])
            expected_tcrk = expected_tck/coeffs

            coeffs /= coeffs.sum()
            expected_tcr = (coeffs * expected_tcrk).sum()

            tcr = total_cost_ratio(zip(cands, costs), relevants, N)
            self.assertTrue(torch.allclose(tcr, expected_tcr))