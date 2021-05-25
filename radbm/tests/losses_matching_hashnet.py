import unittest, torch
from radbm.losses.matching import HashNetMatchingLoss

class TestHashNetMatchingLoss(unittest.TestCase):
    def test_hashnet_matching_loss(self):
        n=64
        log2_lambda = -3
        alpha, beta0 = 4/n, 1
        loss = HashNetMatchingLoss(log2_lambda=log2_lambda, alpha=alpha, beta0=beta0)

        queries = 6*torch.rand((32, n)) - 3
        documents = 6*torch.rand((32, n)) - 3
        r = relevants = torch.randint(0, 2, (32,), dtype=bool)
        l1 = loss(queries, documents, relevants)

        qbits = torch.tanh(queries)
        dbits = torch.tanh(documents)
        dist = (qbits * dbits).sum(dim=1)
        prob = torch.sigmoid(alpha*dist)
        tp_prob = torch.log(prob[r])
        tn_prob = torch.log(1-prob[~r])
        lmbd = 2**log2_lambda
        l2 = -lmbd*tp_prob.mean() - (1-lmbd)*tn_prob.mean()
        self.assertAlmostEqual(float(l1), float(l2), places=5)
        
        with self.assertRaises(ValueError):
            loss(queries[None], documents, relevants)
        with self.assertRaises(ValueError):
            loss(queries, documents[None], relevants)
        with self.assertRaises(ValueError):
            loss(queries, documents, relevants[None,None])
        with self.assertRaises(TypeError):
            loss(queries, documents, relevants.float())
            
        #try block
        relevants = torch.randint(0, 2, (32,32), dtype=bool)
        loss(queries, documents, relevants)