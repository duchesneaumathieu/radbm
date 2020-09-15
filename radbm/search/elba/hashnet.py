import torch
import numpy as np
from torch.nn.utils import clip_grad_value_
from radbm.search.elba import EfficientLearnableBinaryAccess
softplus = torch.nn.Softplus()

class HashNet(EfficientLearnableBinaryAccess):
    """
    HashNet as in "HashNet: Deep Learning to Hash by Continuation"
    by Zhangjie Cao, Mingsheng Long, Jianmin Wang and Philip S. Yu.

    Parameters
    ----------
    fq : torch.nn.Module
        The query Multi-Bernoulli encoder.
    fd : torch.nn.Module
        The document Multi-Bernoulli encoder.
    struct : BaseSDS subclass
        The structure used in ELBA.
    match_prob : float (in [0,1])
        The probability that there is a match given a random query
        and a random document.
    alpha : float (optional)
        The HashNet's alpha used in the adaptative sigmoid (default 1)
    beta : float (optional)
        The HashNet's beta used for ramping the stage in the tanh(stage*beta*x).
        (default 1)
    """
    def __init__(self, fq, fd, struct, match_prob, alpha=1, beta=1, *args, **kwargs):
        super().__init__(fq, fd, struct, *args, **kwargs)
        self.match_prob = match_prob
        self.alpha = alpha
        self.beta = beta
        
    def step(self, q, d, r, stage=1):
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
        stage : float (optional)
            The stage for computing the tanh(stage*beta*x). (default=1)
            
        Returns
        -------
        loss : torch.Tensor (size 1)
            The loss of HashNet.
        """
        self.zero_grad()
        zq = self.fq(q)
        zd = self.fd(d)
        bq = torch.tanh(stage*self.beta*zq)
        bd = torch.tanh(stage*self.beta*zd)
        sh = (bq[:,None]*bd[None,:]).sum(dim=2)
        ash = self.alpha*sh
        p = self.match_prob
        w = (r/p + ~r/(1-p))
        losses = w*(softplus(ash) - r*ash)
        loss = losses.mean()
        loss.backward()
        clip_grad_value_(self.parameters(), 5)
        self.optim.step()
        return loss