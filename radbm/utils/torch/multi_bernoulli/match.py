import torch
from .poisson_binomial import log_hamming_binomial
from radbm.utils.torch import torch_lse
logsigmoid = torch.nn.LogSigmoid()

class HammingMatch(object):
    """
    Functional to compute the probability that two Multi-Bernoulli's
    Hamming distance is below or equal to `dist`.
    
    Parameters
    ----------
    dist : int (positive)
        The radius at which two binary vectors are considered to be 
        "matching". `dist` should be at most the number of bits.
    """
    def __init__(self, dist):
        self.dist = dist
        
    def __call__(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor (dtype: float)
            The logits (pre-sigmoid) of the bits of the first set of
            Multi-Bernoulli. x.shape = (a1, a2, ..., am, n) where n is
            the number of bits. 
        y : torch.tensor (dtype: float)
            The logits (pre-sigmoid) of the bits of the first set of
            Multi-Bernoulli. y.shape = (b1, b2, ..., bm, n) where n is
            the number of bits. 
            
        Returns
        -------
        log_probs_not_match : torch.tensor (dtype: float)
            The log probability that the two Multi-Bernoulli's Hamming distance
            is above `dist`, shape = (c1, c2, ..., cm, n+1).
        log_probs_match : torch.tensor (dtype: float)
            The log probability that the two Multi-Bernoulli's Hamming distance
            is below or equal to `dist`, shape = (c1, c2, ..., cm, n+1).
            
        Notes
        -----
            (a1, a2, ..., am, n) and (b1, b2, ..., bm, n) should be broadcastable,
            where (c1, c2, ..., cm, n) is their broadcasted shape.
            
        """
        xp, yp, xn, yn = map(logsigmoid, (x, y, -x, -y))
        log_hb = log_hamming_binomial(xn, xp, yn, yp)
        log_probs_match = torch_lse(log_hb[..., :self.dist+1], dim=-1)
        log_probs_not_match = torch_lse(log_hb[..., self.dist+1:], dim=-1)
        return log_probs_not_match, log_probs_match