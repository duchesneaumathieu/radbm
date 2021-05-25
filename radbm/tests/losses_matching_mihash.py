import unittest, torch
from radbm.losses.matching import MIHashMatchingLoss

class TestMIHashMatchingLoss(unittest.TestCase):
    def test_mihash_matching_loss(self):
        n=64
        loss = MIHashMatchingLoss(nbits=n, match_prob=1/100)

        queries = 6*torch.rand((32, n)) - 3
        documents = 6*torch.rand((32, n)) - 3
        r = relevants = torch.randint(0, 2, (32,32), dtype=bool)
        loss(queries, documents, relevants)
        
        with self.assertRaises(ValueError):
            loss(queries[None], documents, relevants)
        with self.assertRaises(ValueError):
            loss(queries, documents[None], relevants)
        with self.assertRaises(ValueError):
            loss(queries, documents, relevants[None,None])
        with self.assertRaises(TypeError):
            loss(queries, documents, relevants.float())