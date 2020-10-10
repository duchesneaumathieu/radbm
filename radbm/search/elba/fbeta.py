import torch
from torch.nn.utils import clip_grad_value_
import numpy as np

from radbm.utils import Ramp
from radbm.search.elba import EfficientLearnableBinaryAccess
from radbm.utils.torch import (
    torch_lme,
    positive_loss_adaptative_l2_reg,
    multi_bernoulli_equality,
    log_poisson_binomial,
)

def _importance(match, match_prob):
    batch_prob = match.float().mean()
    r1 = match_prob/batch_prob
    r0 = (1-match_prob)/(1-batch_prob)
    return torch.log(r1*match + r0*~match)

def _gamma(beta, log_match_prob, log_m):
    b2 = beta**2
    log_b2 = np.log(b2)
    c1 = np.log(1+b2) - log_b2
    c2 = log_match_prob + log_b2
    return c1 - torch.nn.Softplus()(log_m-c2)

def fbeta_loss(match, log_match, beta, log_match_prob):
    """
    Compute the Fbeta Loss
    
    Parameters
    ----------
    match : torch.Tensor (dtype=torch.bool)
        Indicate if there is a match or not
    log_match : torch.Tensor (dtype=torch.float)
        The log probability of matching given by an estimator
    beta : float
        The beta in the Fbeta Loss
    log_match_prob : float
        The log probability that there is a match given a random query
        and a random document.
        
    Returns
    -------
    fbeta_loss : torch.Tensor (dtype=torch.float)
        Containing 1 element, this is the Fbeta Loss of the estimator w.r.t match.
    """
    log_r_hats = log_match[match]
    log_m_hats = log_match + _importance(match, np.exp(log_match_prob))
    log_rho = log_r_hats.mean()
    log_m_hat = torch_lme(log_m_hats)
    log_g = _gamma(beta, log_match_prob, log_m_hat)
    log_fbeta = log_rho + log_g
    return -log_fbeta

class Fbeta(EfficientLearnableBinaryAccess):
    """
    The Fbeta model is an ELBA that use the Fbeta Loss to train the Multi-Bernoulli encoders.
    
    Parameters
    ----------
    fq : torch.nn.Module
        The query Multi-Bernoulli encoder.
    fd : torch.nn.Module
        The document Multi-Bernoulli encoder.
    struct : BaseSDS subclass
        The structure used in ELBA.
    log_match_prob : float
        The log probability that there is a match given a random query
        and a random document.
    sim : function (torch.Tensor \times torch.Tensor -> torch.Tensor)
        A function taking the query's Multi-Bernoulli code and the document's
        Multi-Bernoulli code and return the bitwise log probability that each bit match.
    match_dist : int
        The maximum number of dissimilar bits for which we consider that two binary
        vectors are matching.
    ramp : function
        A ramping function used for ramping the log2(beta) at each steps.
    """
    def __init__(self, fq, fd, struct, log_match_prob, sim=multi_bernoulli_equality, match_dist=1, ramp=Ramp(0,1024,-64,-8)):
        super().__init__(fq, fd, struct)
        self.sim = sim
        self.ramp = (lambda x: ramp) if isinstance(ramp, (int, float)) else ramp
        self.log_match_prob = log_match_prob
        self.match_dist = match_dist
        
    def loss(self, match, log_match, nbatch=None):
        log2_beta = self.ramp(nbatch)
        loss = fbeta_loss(match, log_match, 2**log2_beta, self.log_match_prob)
        return loss
        
    def step(self, q, d, match, l2_ratio=0.01, nbatch=None):
        """
        Do a training step.
        
        Parameters
        ----------
        q : torch.Tensor
            A batch of queries.
        d : torch.Tensor
            A batch of documents.
        match : torch.Tensor (dtype=torch.bool)
            A matrix (2D tensor) with match[i,j] indicating if q[i] match with d[j]
        l2_ratio : float (optional)
            The wanted ratio between the Fbeta Loss and L2 regularization. (default 0.01)
        nbatch : int or None (optional)
            Give the number of batch, this is uses for ramping. If None, this uses the
            final ramping value.
            
        Returns
        -------
        loss : torch.Tensor (size 1)
            The loss of the current batch.
        """
        self.zero_grad()
        zq = self.fq(q)
        zd = self.fd(d)
        log_p0, log_p1 = self.sim(zq[:,None], zd[None,:])
        log_pb = log_poisson_binomial(log_p1, log_p0) #inverting log_p0 and log_p1 to count the zeros
        log_match = log_pb[:,:,:self.match_dist+1].sum(dim=2)
        loss = self.loss(match, log_match, nbatch)
        loss = positive_loss_adaptative_l2_reg(loss, l2_ratio, [zq,zd])
        loss.backward()
        clip_grad_value_(self.parameters(), 5)
        self.optim.step()
        return loss