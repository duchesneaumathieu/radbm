import unittest, torch
from radbm.search import BernoulliReduction

class TestBernoulliReduction(unittest.TestCase):
    def test_bernoulli_reduction(self):
        fq = torch.nn.Identity()
        fd = torch.nn.Identity()
        reduction = BernoulliReduction(fq, fd, 'struct')
        
        documents = 2*torch.rand(32, 8) - 1 #in [-1, 1]
        log_p0, log_p1 = reduction.documents_reduction(documents)
        total = log_p0.exp() + log_p1.exp()
        self.assertTrue(torch.allclose(total, torch.ones_like(total)))
        
        queries = 2*torch.rand(32, 8) - 1
        log_p0, log_p1 = reduction.queries_reduction(documents)
        total = log_p0.exp() + log_p1.exp()
        self.assertTrue(torch.allclose(total, torch.ones_like(total)))