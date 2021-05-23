import unittest, torch
import numpy as np
from radbm.utils.torch import HammingMatch
from radbm.losses import FbetaLoss, BCELoss
from radbm.losses import FbetaMultiBernoulliMatchingLoss, BCEMultiBernoulliMatchingLoss

class TestFbetaMBMLoss(unittest.TestCase):
    def test_no_block(self):
        x = 10*torch.rand(2, 32)-5
        y = 10*torch.rand(2, 32)-5
        r = torch.tensor([True, False])
        
        for dist, beta, prob_y1 in [(0, 1, 1/2), (1, 1/2, 1/10), (4, 1/4, 1/100)]:
            match = HammingMatch(dist=dist).soft_match
            loss = FbetaMultiBernoulliMatchingLoss(match, reg_alpha=0, log2_beta=np.log2(beta), prob_y1=prob_y1)(x, y, r)
            neg_log_probs, pos_log_probs = match(x, y)
            tp_log_probs = pos_log_probs[[0]]
            fp_log_probs = pos_log_probs[[1]]
            expected = FbetaLoss(log2_beta=np.log2(beta), prob_y1=prob_y1)(tp_log_probs, fp_log_probs)
            self.assertTrue(torch.allclose(expected, loss))
        
    def test_block(self):
        x = 10*torch.rand(2, 32)-5
        y = 10*torch.rand(1, 32)-5
        r = torch.tensor([[True], [False]])
        
        match = HammingMatch(dist=0).soft_match
        f = FbetaMultiBernoulliMatchingLoss(match, reg_alpha=0.001, log2_beta=-1, prob_y1=1/10)
        
        v1 = f(x, y, r)
        v2 = f(x, torch.cat([y,y]), r[:,0])
        self.assertTrue(torch.allclose(v1, v2))
        
    def test_no_block_error(self):
        x = 10*torch.rand(2, 32)-5
        y = 10*torch.rand(1, 32)-5
        r = torch.tensor([True, False])
        
        match = HammingMatch(dist=0).soft_match
        f = FbetaMultiBernoulliMatchingLoss(match, reg_alpha=0.001, log2_beta=-1, prob_y1=1/10)
        with self.assertRaises(ValueError):
            f(x, y, r)
            
    def test_block_error(self):
        x = 10*torch.rand(2, 3, 32)-5
        y = 10*torch.rand(1, 4, 32)-5
        r = torch.tensor([[True, False], [False, False], [True, True]])
        
        match = HammingMatch(dist=0).soft_match
        f = FbetaMultiBernoulliMatchingLoss(match, reg_alpha=0.001, log2_beta=-1, prob_y1=1/10)
        with self.assertRaises(ValueError):
            f(x, y, r[:,[0]]) #x.shape[0] != r.shape[0]
            
        with self.assertRaises(ValueError):
            f(x, y, r[:2]) #y.shape[0] != r.shape[1]
        
class TestBCEMBMLoss(unittest.TestCase):
    def test_no_block(self):
        x = 10*torch.rand(2, 32)-5
        y = 10*torch.rand(2, 32)-5
        r = torch.tensor([True, False])
        
        for dist, log2_lambda in [(0, -1.), (1, -2), (4, -3)]:
            match = HammingMatch(dist=dist).soft_match
            loss = BCEMultiBernoulliMatchingLoss(match, reg_alpha=0, log2_lambda=log2_lambda)(x, y, r)
            neg_log_probs, pos_log_probs = match(x, y)
            tp_log_probs = pos_log_probs[[0]]
            tn_log_probs = neg_log_probs[[1]]
            expected = BCELoss(log2_lambda=log2_lambda)(tp_log_probs, tn_log_probs)
            self.assertTrue(torch.allclose(expected, loss))
        
    def test_block(self):
        x = 10*torch.rand(2, 32)-5
        y = 10*torch.rand(1, 32)-5
        r = torch.tensor([[True], [False]])
        
        match = HammingMatch(dist=0).soft_match
        f = BCEMultiBernoulliMatchingLoss(match, reg_alpha=0.001, log2_lambda=-1.)
        
        v1 = f(x, y, r)
        v2 = f(x, torch.cat([y,y]), r[:,0])
        self.assertTrue(torch.allclose(v1, v2))
        
    def test_no_block_error(self):
        x = 10*torch.rand(2, 3, 32)-5
        y = 10*torch.rand(1, 4, 32)-5
        r = torch.tensor([True, False])
        
        match = HammingMatch(dist=0).soft_match
        f = BCEMultiBernoulliMatchingLoss(match, reg_alpha=0.001, log2_lambda=-1.)
        with self.assertRaises(ValueError):
            f(x, y, r)
            
    def test_block_error(self):
        x = 10*torch.rand(2, 3, 32)-5
        y = 10*torch.rand(1, 4, 32)-5
        r = torch.tensor([[True, False], [False, False], [True, True]])
        
        match = HammingMatch(dist=0).soft_match
        f = BCEMultiBernoulliMatchingLoss(match, reg_alpha=0.001, log2_lambda=-1.)
        with self.assertRaises(ValueError):
            f(x, y, r[:,[0]]) #x.shape[0] != r.shape[0]
            
        with self.assertRaises(ValueError):
            f(x, y, r[:2]) #y.shape[0] != r.shape[1]