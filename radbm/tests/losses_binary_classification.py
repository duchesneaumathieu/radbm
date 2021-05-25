import torch
import unittest
import numpy as np
from radbm.losses import FbetaLoss, BCELoss

class _ExpFbetaLoss(object):
    def __init__(self, beta, prob_y1, estimator_sharing=True):
        self.beta = beta
        self.estimator_sharing = estimator_sharing
        self.prob_y1 = prob_y1
        self.c1 = prob_y1*beta**2
        self.c2 = prob_y1*(1 + beta**2)

    def __call__(self, tp_probs, fp_probs, estimator_sharing=True):
        if self.estimator_sharing:
            t1_tp_probs = t2_tp_probs = tp_probs
        else:
            tp_probs = tp_probs.flatten()
            k = len(tp_probs)//2
            t1_tp_probs = tp_probs[:k]
            t2_tp_probs = tp_probs[k:]
        t1 = t1_tp_probs.mean()
        recall = t2_tp_probs.mean()
        fallout = fp_probs.mean()
        t = self.prob_y1*recall + (1-self.prob_y1)*fallout
        return self.c2*t1 / (self.c1 + t)
    
class TestFbetaLoss(unittest.TestCase):
    def test_fbeta_loss(self):
        #with sharing
        for i in range(100):
            beta = 2*torch.rand(1)
            prob_y1 = torch.rand(1)
            efl = _ExpFbetaLoss(beta, prob_y1, estimator_sharing=True)
            nfl = FbetaLoss(np.log2(beta), prob_y1, estimator_sharing=True, naive=True)
            fl = FbetaLoss(np.log2(beta), prob_y1, estimator_sharing=True, naive=False)
            pos = .4*torch.rand(19)+.4 #in [.4, .8]
            neg = .6*torch.rand(23)+.1 #in [.1, .7]
            expected = efl(pos, neg)
            self.assertTrue(torch.allclose(expected, torch.exp(-nfl(pos.log(), neg.log()))))
            #the non-naive estimator is always smaller the the naive estimator calculated
            #using the same pos and neg
            self.assertTrue(expected > torch.exp(-fl(pos.log(), neg.log())))
        for i in range(100):
            beta = 2*torch.rand(1)
            prob_y1 = torch.rand(1)
            efl = _ExpFbetaLoss(beta, prob_y1, estimator_sharing=False)
            nfl = FbetaLoss(np.log2(beta), prob_y1, estimator_sharing=False, naive=True)
            fl = FbetaLoss(np.log2(beta), prob_y1, estimator_sharing=False, naive=False)
            pos = .4*torch.rand(19)+.4 #in [.4, .8]
            neg = .6*torch.rand(23)+.1 #in [.1, .7]
            expected = efl(pos, neg)
            self.assertTrue(torch.allclose(expected, torch.exp(-nfl(pos.log(), neg.log()))))
            self.assertTrue(expected > torch.exp(-fl(pos.log(), neg.log())))
            
    def test_missing_pairs_error(self):
        x = torch.rand(10)
        y = torch.rand(0)
        f = FbetaLoss(log2_beta=-1, prob_y1=1/10)
        with self.assertRaises(ValueError):
            f(x, y)
        with self.assertRaises(ValueError):
            f(y, x)

class TestBCELoss(unittest.TestCase):
    def test_bce_loss(self):
        y = torch.randint(0, 2, (100,), dtype=bool)
        n = len(y); n1 = y.sum(); n0 = n - n1
        for log2_lambda in torch.linspace(-10, -1, 100):
            lmda = 2**log2_lambda
            w1 = n*lmda/n1
            w0 = (n-n1*w1)/n0
            weight = y*w1 + (~y)*w0
            probs = torch.rand(100)
            expected = torch.nn.BCELoss(weight=weight)(probs, y.float())
            loss = BCELoss(log2_lambda=log2_lambda)(probs[y].log(), (1-probs)[~y].log())
            self.assertTrue(torch.allclose(expected, loss))
            
    def test_bce_ramping(self):
        ramping_bce = BCELoss(log2_lambda=(10,100,-10,-1))
        bce10 = BCELoss(log2_lambda=-10)
        bce1 = BCELoss(log2_lambda=-1)
        tp_log_probs = torch.rand(123).log()
        tn_log_probs = torch.rand(321).log()
        l1 = bce1(tp_log_probs, tn_log_probs)
        l10 = bce10(tp_log_probs, tn_log_probs)
        for n in torch.linspace(-2, 10, 100):
            l = ramping_bce(tp_log_probs, tn_log_probs, step=int(n))
            self.assertTrue(torch.allclose(l10, l))
        #assuming it is fine in the in [10, 100]
        for n in torch.linspace(100, 1000, 100):
            l = ramping_bce(tp_log_probs, tn_log_probs, step=int(n))
            self.assertTrue(torch.allclose(l1, l))
            
    def test_missing_pairs_error(self):
        x = torch.rand(10)
        y = torch.rand(0)
        f = BCELoss()
        with self.assertRaises(ValueError):
            f(x, y)
        with self.assertRaises(ValueError):
            f(y, x)