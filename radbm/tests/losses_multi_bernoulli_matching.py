import unittest, torch
from radbm.utils.torch import HammingMatch
from radbm.losses import FbetaLoss, BCELoss
from radbm.losses import FbetaMultiBernoulliMatchingLoss, BCEMultiBernoulliMatchingLoss

class TestFbetaMBMLoss(unittest.TestCase):
    def test_no_block_no_multi(self):
        x = 10*torch.rand(2, 32)-5
        y = 10*torch.rand(2, 32)-5
        r = torch.tensor([True, False])
        
        for dist, beta, prob_y1 in [(0, 1, 1/2), (1, 1/2, 1/10), (4, 1/4, 1/100)]:
            match = HammingMatch(dist=dist)
            loss = FbetaMultiBernoulliMatchingLoss(match, l2_ratio=0, beta=beta, prob_y1=prob_y1)(x, y, r)
            pos_log_probs, neg_log_probs = match(x, y)
            tp_log_probs = pos_log_probs[[0]]
            fp_log_probs = pos_log_probs[[1]]
            expected = FbetaLoss(beta=beta, prob_y1=prob_y1)(tp_log_probs, fp_log_probs)
            self.assertTrue(torch.allclose(expected, loss))
            
    def test_no_block_multi(self):
        x = 10*torch.rand(2, 1, 32)-5
        y = 10*torch.rand(2, 1, 32)-5
        r = torch.tensor([True, False])
        
        match = HammingMatch(dist=0)
        f = FbetaMultiBernoulliMatchingLoss(match, l2_ratio=0.001, beta=1/2, prob_y1=1/10)
        v1 = f(x, y, r)
        v2 = f(x[:,0], y, r)
        v3 = f(x, y[:,0], r)
        v4 = f(x[:,0], y[:,0], r)
        for a, b in [(v1, v2), (v2, v3), (v3, v4)]:
            self.assertTrue(torch.allclose(a, b))
        
    def test_block_no_multi(self):
        x = 10*torch.rand(2, 32)-5
        y = 10*torch.rand(1, 32)-5
        r = torch.tensor([[True], [False]])
        
        match = HammingMatch(dist=0)
        f = FbetaMultiBernoulliMatchingLoss(match, l2_ratio=0.001, beta=1/2, prob_y1=1/10)
        
        v1 = f(x, y, r)
        v2 = f(x, torch.cat([y,y]), r[:,0])
        self.assertTrue(torch.allclose(v1, v2))
        
    def test_block_multi(self):
        x = 10*torch.rand(2, 3, 32)-5
        y = 10*torch.rand(2, 4, 32)-5
        r = torch.tensor([[True, False], [False, True]])
        
        match = HammingMatch(dist=0)
        f = FbetaMultiBernoulliMatchingLoss(match, l2_ratio=0.001, beta=1/2, prob_y1=1/10)
        f(x, y, r) #make sure it runs...
        
    def test_no_block_error(self):
        x = 10*torch.rand(2, 3, 32)-5
        y = 10*torch.rand(1, 4, 32)-5
        r = torch.tensor([True, False])
        
        match = HammingMatch(dist=0)
        f = FbetaMultiBernoulliMatchingLoss(match, l2_ratio=0.001, beta=1/2, prob_y1=1/10)
        with self.assertRaises(ValueError):
            f(x, y, r) #multi
            
        with self.assertRaises(ValueError):
            f(x[:,0], y[:,0], r) #no multi
            
    def test_block_error(self):
        x = 10*torch.rand(2, 3, 32)-5
        y = 10*torch.rand(1, 4, 32)-5
        r = torch.tensor([[True, False], [False, False], [True, True]])
        
        match = HammingMatch(dist=0)
        f = FbetaMultiBernoulliMatchingLoss(match, l2_ratio=0.001, beta=1/2, prob_y1=1/10)
        with self.assertRaises(ValueError):
            f(x, y, r[:,[0]]) #x.shape[0] != r.shape[0]
            
        with self.assertRaises(ValueError):
            f(x, y, r[:2]) #y.shape[0] != r.shape[1]
        
class TestBCEMBMLoss(unittest.TestCase):
    def test_no_block_no_multi(self):
        x = 10*torch.rand(2, 32)-5
        y = 10*torch.rand(2, 32)-5
        r = torch.tensor([True, False])
        
        for dist, w1 in [(0, 1), (1, 1/2), (4, 1/4)]:
            match = HammingMatch(dist=dist)
            loss = BCEMultiBernoulliMatchingLoss(match, l2_ratio=0, w1=w1)(x, y, r)
            pos_log_probs, neg_log_probs = match(x, y)
            tp_log_probs = pos_log_probs[[0]]
            tn_log_probs = neg_log_probs[[1]]
            expected = BCELoss(w1=w1)(tp_log_probs, tn_log_probs)
            self.assertTrue(torch.allclose(expected, loss))
            
    def test_no_block_multi(self):
        x = 10*torch.rand(2, 1, 32)-5
        y = 10*torch.rand(2, 1, 32)-5
        r = torch.tensor([True, False])
        
        match = HammingMatch(dist=0)
        f = BCEMultiBernoulliMatchingLoss(match, l2_ratio=0.001, w1=1)
        v1 = f(x, y, r)
        v2 = f(x[:,0], y, r)
        v3 = f(x, y[:,0], r)
        v4 = f(x[:,0], y[:,0], r)
        for a, b in [(v1, v2), (v2, v3), (v3, v4)]:
            self.assertTrue(torch.allclose(a, b))
        
    def test_block_no_multi(self):
        x = 10*torch.rand(2, 32)-5
        y = 10*torch.rand(1, 32)-5
        r = torch.tensor([[True], [False]])
        
        match = HammingMatch(dist=0)
        f = BCEMultiBernoulliMatchingLoss(match, l2_ratio=0.001, w1=1)
        
        v1 = f(x, y, r)
        v2 = f(x, torch.cat([y,y]), r[:,0])
        self.assertTrue(torch.allclose(v1, v2))
        
    def test_block_multi(self):
        x = 10*torch.rand(2, 3, 32)-5
        y = 10*torch.rand(2, 4, 32)-5
        r = torch.tensor([[True, False], [False, True]])
        
        match = HammingMatch(dist=0)
        f = BCEMultiBernoulliMatchingLoss(match, l2_ratio=0.001, w1=1)
        f(x, y, r) #make sure it runs...
        
    def test_no_block_error(self):
        x = 10*torch.rand(2, 3, 32)-5
        y = 10*torch.rand(1, 4, 32)-5
        r = torch.tensor([True, False])
        
        match = HammingMatch(dist=0)
        f = BCEMultiBernoulliMatchingLoss(match, l2_ratio=0.001, w1=1)
        with self.assertRaises(ValueError):
            f(x, y, r) #multi
            
        with self.assertRaises(ValueError):
            f(x[:,0], y[:,0], r) #no multi
            
    def test_block_error(self):
        x = 10*torch.rand(2, 3, 32)-5
        y = 10*torch.rand(1, 4, 32)-5
        r = torch.tensor([[True, False], [False, False], [True, True]])
        
        match = HammingMatch(dist=0)
        f = BCEMultiBernoulliMatchingLoss(match, l2_ratio=0.001, w1=1)
        with self.assertRaises(ValueError):
            f(x, y, r[:,[0]]) #x.shape[0] != r.shape[0]
            
        with self.assertRaises(ValueError):
            f(x, y, r[:2]) #y.shape[0] != r.shape[1]