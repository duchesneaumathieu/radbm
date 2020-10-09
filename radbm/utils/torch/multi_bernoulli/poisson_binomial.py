import torch
import numpy as np
from radbm.utils.torch import torch_logsumexp

def _log_poisson_binomial(log_q0, log_q1):
    bs, n = log_q0.shape
    dtype, device = log_q0.dtype, log_q0.device
    ninf = torch.tensor(-np.inf, dtype=dtype, device=device).expand(1, bs)
    z = torch.zeros((1, bs), dtype=dtype, device=device)
    for n,(a,b) in enumerate(zip(log_q0.permute(1,0), log_q1.permute(1,0))):
        w0 = torch.cat([a + z, ninf], dim=0)
        w1 = torch.cat([ninf, b + z], dim=0)
        z = torch_logsumexp(w0, w1)
    return z.permute(1,0)

def log_poisson_binomial(log_q0, log_q1):
    """
    Computes the events log probabilities w.r.t. a batch of Poisson Binomial R.V.
    The computation is numerically stable.
    
    Parameters
    ----------
    log_q0 : torch.tensor (dtype=torch.float)
        The log probability of each bits to be zero. The Poisson Binomial is considered to
        be on the last dim. shape=(a1,a2,a3,...,am,n) where n is the number of bits for each 
        Poisson Binomial. a1,a2,a3,...,am are arbitrary but should match with log_q1.
    
    log_q1 : torch.tensor (dtype=torch.float)
        The log probability of each bits to be one. The Poisson Binomial is considered to
        be on the last dim. shape=(a1,a2,a3,...,am,n) where n is the number of bits for each 
        Poisson Binomial. a1,a2,a3,...,am are arbitrary but should match with log_q0.
        
    Returns
    -------
    log_pb : torch.tensor(dtype=torch.float)
        The log probability of each Poisson Binomial events. shape=(a1,a2,a3,...,am,n+1).
        log_pb[i1,i2,i3,...,am,k] is the log probability that the sum of the n Bernoulli
        with parameters log_q1.exp()[i1,i2,i3,...,am] gives k.
        
    Notes
    -----
    We should have (1-log_q0.exp()).log() = log_q1 in theory, hence the input of this function
    is over specified. But for numerical stability we need both log_q0 and log_q1 and it is not
    possible to compute log_q0 from log_q1 with numerical stability (and vice versa). This is why
    they are both required as input. Since in some cases, it is possible to compute log_q0 and
    log_q1 with numerical stability, i.e. log_q0 = log_sigmoid(-logits) and log_q1 = log_sigmoid(logits)
        
    """
    shape = log_q0.shape
    if shape != log_q1.shape:
        msg = 'log_q0.shape and log_q1.shape must be the same, got {} and {} respectively'
        raise ValueError(msg.format(shape, log_q1.shape))
    n = shape[-1]
    log_pb = _log_poisson_binomial(log_q0.view(-1,n), log_q1.view(-1,n))
    return log_pb.view(*shape[:-1], n+1)

def log_hamming_binomial(log_p10, log_p11, log_p20, log_p21):
    """
    Computes the log probabilities of each Hamming Binomial events, parameterized
    with p1 and p2.
    
    Parameters
    ----------
    log_p10 : torch.tensor (dtype=torch.float)
        The log probability of each bits to be zero of the first random vector.
        The Hamming Binomial is considered to be on the last dim. shape=(a1,a2,a3,...,am,n)
        where n is the number of bits for each vectors. a1,a2,a3,...,am are arbitrary but
        should be broadcastable with the other inputs.
    
    log_p11 : torch.tensor (dtype=torch.float)
        The log probability of each bits to be one of the first random vector.
        The Hamming Binomial is considered to be on the last dim. shape=(a1,a2,a3,...,am,n)
        where n is the number of bits for each vectors. a1,a2,a3,...,am are arbitrary but
        should be broadcastable with the other inputs.
        
    log_p20 : torch.tensor (dtype=torch.float)
        The log probability of each bits to be zero of the second random vector.
        The Hamming Binomial is considered to be on the last dim. shape=(a1,a2,a3,...,am,n)
        where n is the number of bits for each vectors. a1,a2,a3,...,am are arbitrary but
        should be broadcastable with the other inputs.
    
    log_p21 : torch.tensor (dtype=torch.float)
        The log probability of each bits to be one of the second random vector.
        The Hamming Binomial is considered to be on the last dim. shape=(a1,a2,a3,...,am,n)
        where n is the number of bits for each vectors. a1,a2,a3,...,am are arbitrary but
        should be broadcastable with the other inputs.
        
    Returns
    -------
    log_hb : torch.tensor (dtype=torch.float)
        The log probability of each Hamming Binomial events. shape=(a1,a2,a3,...,am,n+1).
        log_pb[i1,i2,i3,...,am,k] is the log probability that the hamming distance between
        the two random Multi-Bernoulli parameterized by log_p11.exp()[i1,i2,i3,...,am] and
        log_p21.exp()[i1,i2,i3,...,am] be k.
        
    Notes
    -----
    see log_poisson_binomial's notes.
    """
    log_q0 = torch_logsumexp(log_p11 + log_p21, log_p10 + log_p20)
    log_q1 = torch_logsumexp(log_p11 + log_p20, log_p10 + log_p21)
    return log_poisson_binomial(log_q0, log_q1)