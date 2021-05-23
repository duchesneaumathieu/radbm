import torch, unittest, itertools
from radbm.utils.torch.multi_bernoulli.match import MultiIndexingMatch, HammingMatch

def hamming_binomial(x, y):
    sh = (x+y).shape #broadcasting shape
    n = sh[-1]
    hb = torch.zeros(sh[:-1] + (n+1,))
    for outcome in itertools.product(*2*n*[(False, True)]):
        x_outcome = torch.tensor(outcome[:n])
        y_outcome = torch.tensor(outcome[n:])
        hamming_dist = (x_outcome^y_outcome).sum()
        x_probs = x[...,x_outcome].prod(dim=-1)*(1-x[...,~x_outcome]).prod(dim=-1)
        y_probs = y[...,y_outcome].prod(dim=-1)*(1-y[...,~y_outcome]).prod(dim=-1)
        probs = x_probs*y_probs
        hb[..., hamming_dist] += probs
    return hb


class TestHammingMatch(unittest.TestCase):
    def test_hamming_match(self):
        gen = torch.Generator().manual_seed(0xcafe)
        x_logits = torch.zeros((2,1,4,1,6), dtype=torch.float32)
        y_logits = torch.zeros((1,3,1,5,6), dtype=torch.float32)
        x_logits.uniform_(-5, 5, generator=gen)
        y_logits.uniform_(-5, 5, generator=gen)
        x_probs = torch.sigmoid(x_logits)
        y_probs = torch.sigmoid(y_logits)
        
        expected_hb = hamming_binomial(x_probs, y_probs)
        
        for dist in range(6):
            expected_match_probs = expected_hb[..., :dist+1].sum(dim=-1)
            log_probs_not_match, log_probs_match = HammingMatch(dist).soft_match(x_logits, y_logits)
            probs_not_match = log_probs_not_match.exp()
            probs_match = log_probs_match.exp()
            self.assertTrue(torch.allclose(torch.ones_like(probs_match), probs_match+probs_not_match))
            self.assertTrue(torch.allclose(expected_match_probs, probs_match))
            

def soft_multi_indexing_equal_probs(x, y): #equal == HammingMatch(dist=0)
    x = x[:, :, None, :]
    y = y[:, None, :, :]
    bitwise_equal_probs = x*y + (1-x)*(1-y)
    index_wise_equal_probs = bitwise_equal_probs.prod(dim=3)
    not_equal_probs = (1-index_wise_equal_probs).prod(dim=1).prod(dim=1)
    return 1 - not_equal_probs
    
def hard_multi_indexing_equal_probs(x, y):
    return (x[:, :, None, :] == y[:, None, :, :]).all(dim=3).any(dim=2).any(dim=1)

class TestMultiIndexingMatch(unittest.TestCase):
    def test_error(self):
        match = MultiIndexingMatch()
        with self.assertRaises(NotImplementedError):
            match.soft_match(None, None)
        with self.assertRaises(NotImplementedError):
            match.hard_match(None, None)
    
    def test_soft_match(self):
        bs, nx, ny, k = 256, 3, 4, 5
        x = torch.rand((bs, nx, k))
        y = torch.rand((bs, ny, k))
        log_prob_not_match, log_prob_match = HammingMatch(dist=0).soft_multi_indexing_match( #soft equality
            torch.logit(x),
            torch.logit(y),
        )
    
        expected_equal_probs = soft_multi_indexing_equal_probs(x, y)
        self.assertTrue(torch.allclose(log_prob_match.exp(), expected_equal_probs))
        self.assertTrue(torch.allclose(log_prob_not_match.exp(), 1-expected_equal_probs))
        
    def test_hard_match(self):
        bs, nx, ny, k = 256, 3, 4, 5
        x = torch.randint(0, 2, (bs, nx, k), dtype=bool)
        y = torch.randint(0, 2, (bs, ny, k), dtype=bool)
        match = HammingMatch(dist=0).hard_multi_indexing_match(x, y)
        expected_match = hard_multi_indexing_equal_probs(x, y)
        torch.equal(match, expected_match)