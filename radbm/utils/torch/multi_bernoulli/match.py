import torch
from .poisson_binomial import log_hamming_binomial
from .log_arithmetic import torch_log_prob_any
from radbm.utils.torch import torch_lse
from radbm.metrics import hamming_distance
logsigmoid = torch.nn.LogSigmoid()

class MultiIndexingMatch(object):
    def soft_match(self, x, y):
        raise NotImplementedError('MultiIndexingMatch should be subclassed and soft_match should be implemented.')
        
    def hard_match(self, x, y):
        raise NotImplementedError('MultiIndexingMatch should be subclassed and hard_match should be implemented.')
    
    def soft_multi_indexing_match(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor (dtype: float)
            The logits (pre-sigmoid) of the bits of the first set of
            Multi-Bernoulli. x.shape = (a1, a2, ..., am, nx, kx) where nx is the number
            of index of x and kx is the indexes size.
        y : torch.tensor (dtype: float)
            The logits (pre-sigmoid) of the bits of the first set of
            Multi-Bernoulli. y.shape = (b1, b2, ..., bm, ny, ky) where ny is the number
            of index of y and ky is the indexes size.
            
        Returns
        -------
        log_probs_not_match : torch.tensor (dtype: float)
            The log probability that all of the nx index of x does not match with 
            all of the ny index of y, shape = (c1, c2, ..., cm).
        log_probs_match : torch.tensor (dtype: float)
            The log probability that any of the nx index of x matches with any
            of the ny index of y, shape = (c1, c2, ..., cm).
            
        Notes
        -----
            (a1, a2, ..., am) and (b1, b2, ..., bm) should be compatible,
            i.e. they must by broadcastable where the broadcast gives
            (c1, c2, ..., cm)
        """
        x = x.unsqueeze(-2)
        y = y.unsqueeze(-3)
        index_wise_log_probs_not_match, index_wise_log_probs_match = self.soft_match(x, y)
        log_probs_not_match, log_probs_match = torch_log_prob_any(
            index_wise_log_probs_not_match.flatten(start_dim=-2),
            index_wise_log_probs_match.flatten(start_dim=-2),
        )
        return log_probs_not_match, log_probs_match
    
    def hard_multi_indexing_match(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor
            x.shape = (a1, a2, ..., am, nx, k) where k is
            the number of bits and nx is the number of index of x.
        y : torch.tensor
            y.shape = (b1, b2, ..., bm, ny, k) where k is
            the number of bits and ny is the number of index of y.
            
        Returns
        -------
        match : torch.tensor (dtype: bool)
            Whether if any of the nx index of x matches with any
            of the ny index of y, shape = (c1, c2, ..., cm).
            
        Notes
        -----
            (a1, a2, ..., am, k) and (b1, b2, ..., bm, k) should be compatible,
            i.e. they must by broadcastable where the broadcast gives
            (c1, c2, ..., cm, k)
        """
        x = x.unsqueeze(-2)
        y = y.unsqueeze(-3)
        index_wise_match = self.hard_match(x, y)
        match = index_wise_match.flatten(start_dim=-2).any(dim=-1)
        return match
    


class HammingMatch(MultiIndexingMatch):
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
        
    def soft_match(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor (dtype: float)
            The logits (pre-sigmoid) of the bits of the first set of
            Multi-Bernoulli. x.shape = (a1, a2, ..., am, k) where k is
            the number of bits. 
        y : torch.tensor (dtype: float)
            The logits (pre-sigmoid) of the bits of the first set of
            Multi-Bernoulli. y.shape = (b1, b2, ..., bm, k) where k is
            the number of bits. 
            
        Returns
        -------
        log_probs_not_match : torch.tensor (dtype: float)
            The log probability that the two Multi-Bernoulli's Hamming distance
            is above `dist`, shape = (c1, c2, ..., cm).
        log_probs_match : torch.tensor (dtype: float)
            The log probability that the two Multi-Bernoulli's Hamming distance
            is below or equal to `dist`, shape = (c1, c2, ..., cm).
            
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
    
    def hard_match(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor (dtype: bool)
            The bits of the first set of Multi-Bernoulli.
            x.shape = (a1, a2, ..., am, k) where k is the number of bits. 
        y : torch.tensor (dtype: bool)
            The bits of the first set of Multi-Bernoulli.
            y.shape = (b1, b2, ..., bm, k) where k is the number of bits. 
            
        Returns
        -------
        match : torch.tensor (dtype: float)
            Whether the two Multi-Bernoulli's Hamming distance
            is below or equal to `dist`, shape = (c1, c2, ..., cm).
            
        Notes
        -----
            (a1, a2, ..., am, n) and (b1, b2, ..., bm, n) should be broadcastable,
            where (c1, c2, ..., cm, n) is their broadcasted shape.
            
        """
        dists = hamming_distance(x, y, dim=-1)
        match = dists <= self.dist
        return match
