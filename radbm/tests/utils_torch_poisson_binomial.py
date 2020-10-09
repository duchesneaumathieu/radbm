import unittest, itertools, torch
from radbm.utils.torch import log_poisson_binomial, log_hamming_binomial

def explicit_poisson_binomial(q):
    out = torch.zeros(len(q)+1)
    for k in range(len(q)+1):
        for I in map(list, itertools.combinations(range(len(q)), k)):
            nI = [i for i in range(len(q)) if i not in I]
            out[k] += q[I].prod()*(1-q)[nI].prod()
    return out

class TestLogPoissonBinomial(unittest.TestCase):
    def test_accuracy(self):
        q = torch.tensor([0.2, 0.1, 0.8, 0.7, 0.99])
        expected = explicit_poisson_binomial(q)
        log_q0, log_q1 = (1-q).log(), q.log()
        pb = log_poisson_binomial(log_q0, log_q1).exp()
        self.assertTrue(torch.allclose(pb, expected))
        
    def test_batch(self):
        torch.manual_seed(0xcafe)
        q = torch.rand(3,4,5,64)
        log_q0, log_q1 = (1-q).log(), q.log()
        log_pb = log_poisson_binomial(log_q0, log_q1)
        log_pb123 = log_poisson_binomial(log_q0[1,2,3], log_q1[1,2,3])
        self.assertTrue(torch.allclose(log_pb[1,2,3], log_pb123))
        
    def test_shape_mismatch(self):
        q = torch.tensor([0.2, 0.1, 0.8, 0.7, 0.99])
        log_q0, log_q1 = (1-q).log(), q.log()
        with self.assertRaises(ValueError):
            log_poisson_binomial(log_q0, log_q1[:-1])
        
class TestLogHammingBinomial(unittest.TestCase):
    def test_accuracy(self):
        p1 = torch.tensor([0.2, 0.1, 0.8, 0.7, 0.99])
        p2 = torch.tensor([0.3, 0.75, 0.4, 0.5, 0.9])
        q = p1*(1-p2) + (1-p1)*p2
        expected = explicit_poisson_binomial(q)
        log_p10, log_p11 = (1-p1).log(), p1.log()
        log_p20, log_p21 = (1-p2).log(), p2.log()
        hb = log_hamming_binomial(log_p10, log_p11, log_p20, log_p21).exp()
        self.assertTrue(torch.allclose(hb, expected))