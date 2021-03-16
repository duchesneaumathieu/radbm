import torch
import unittest
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
            nfl = FbetaLoss(beta, prob_y1, estimator_sharing=True, naive=True)
            fl = FbetaLoss(beta, prob_y1, estimator_sharing=True, naive=False)
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
            nfl = FbetaLoss(beta, prob_y1, estimator_sharing=False, naive=True)
            fl = FbetaLoss(beta, prob_y1, estimator_sharing=False, naive=False)
            pos = .4*torch.rand(19)+.4 #in [.4, .8]
            neg = .6*torch.rand(23)+.1 #in [.1, .7]
            expected = efl(pos, neg)
            self.assertTrue(torch.allclose(expected, torch.exp(-nfl(pos.log(), neg.log()))))
            self.assertTrue(expected > torch.exp(-fl(pos.log(), neg.log())))

class TestBCELoss(unittest.TestCase):
    def test_bce_loss(self):
        y = torch.randint(0, 2, (100,), dtype=bool)
        for w1 in 2**-torch.linspace(0, 12, 6):
            w0 = 2 - w1
            weight = y*w1 + (~y)*w0
            probs = torch.rand(100)
            expected = torch.nn.BCELoss(weight=weight)(probs, y.float())
            loss = BCELoss(w1)(probs[y].log(), (1-probs)[~y].log())
            self.assertTrue(torch.allclose(expected, loss))