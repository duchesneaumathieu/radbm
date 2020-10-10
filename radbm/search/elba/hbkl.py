import torch
import numpy as np
from scipy.stats import binom
from torch.nn.utils import clip_grad_value_
from radbm.search.elba import EfficientLearnableBinaryAccess
from radbm.utils.torch import log_hamming_binomial
log_sigmoid = torch.nn.LogSigmoid()

def kl_div(log_p, log_q, *args, **kwargs):
    return (log_p.exp()*(log_p - log_q)).sum(*args, **kwargs)

class HBKL(EfficientLearnableBinaryAccess):
    """
    Hamming Binomial with KL-divergence

    Parameters
    ----------
    fq : torch.nn.Module
        The query Multi-Bernoulli encoder.
    fd : torch.nn.Module
        The document Multi-Bernoulli encoder.
    struct : BaseSDS subclass
        The structure used in ELBA.
    nbits : int
        The number of bits
    positive_binomial_prob : float [0,1]
        The target binomial probability for matching query-document pairs
    negative_binomial_prob : float [0,1]
        The target binomial probability for non-matching query-document pairs
    """
    def __init__(self, fq, fd, struct, nbits, positive_binomial_prob, negative_binomial_prob, *args, **kwargs):
        super().__init__(fq, fd, struct, *args, **kwargs)
        self.register_buffer('pos_binom_logpmf', torch.tensor(binom(nbits, positive_binomial_prob).logpmf(range(nbits+1))))
        self.register_buffer('neg_binom_logpmf', torch.tensor(binom(nbits, negative_binomial_prob).logpmf(range(nbits+1))))
        
    def log_hamming_binomial(self, q, d):
        zq = self.fq(q)[:,None]
        zd = self.fd(d)[None,:]
        return log_hamming_binomial(*map(log_sigmoid, [-zq, zq, -zd, zd]))
    
    def loss(self, q, d, r):
        log_hb = self.log_hamming_binomial(q, d)
        pos_losses = kl_div(self.pos_binom_logpmf, log_hb[r], dim=1)
        neg_losses = kl_div(log_hb[~r], self.neg_binom_logpmf, dim=1)
        return pos_losses.mean() + neg_losses.mean() #could potentially weight those terms
        
    def step(self, q, d, r):
        """
        Do a training step.
        
        Parameters
        ----------
        q : torch.Tensor
            A batch of queries.
        d : torch.Tensor
            A batch of documents.
        r : torch.Tensor (dtype=torch.bool)
            A matrix (2D tensor) with r[i,j] indicating if q[i] match with d[j]
            
        Returns
        -------
        loss : torch.Tensor (size 1)
            The loss of HBKL.
        """
        self.zero_grad()
        loss = self.loss(q, d, r)
        loss.backward()
        clip_grad_value_(self.parameters(), 5)
        self.optim.step()
        return loss