import torch, unittest, itertools
from radbm.utils.torch import HammingMatch

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
            log_probs_not_match, log_probs_match = HammingMatch(dist)(x_logits, y_logits)
            probs_not_match = log_probs_not_match.exp()
            probs_match = log_probs_match.exp()
            self.assertTrue(torch.allclose(torch.ones_like(probs_match), probs_match+probs_not_match))
            self.assertTrue(torch.allclose(expected_match_probs, probs_match))