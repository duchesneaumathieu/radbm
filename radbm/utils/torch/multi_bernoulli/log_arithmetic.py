import torch
from radbm.utils.torch import torch_logsumexp
logsigmoid = torch.nn.LogSigmoid()

def multi_bernoulli_equality(xz, yz):
    """
    Compute the bitwise log probability that two Multi-Bernoulli are equal.
    
    Parameters
    ----------
    xz : torch.tensor
        the logits (before sigmoid) of the first Multi-Bernoulli
    yz : torch.tensor
        the logits (before sigmoid) of the second Multi-Bernoulli
        
    Returns
    -------
    log_p0 : torch.tensor
        the bitwise log probability that the two Multi-Bernoulli are not equal
    log_p1 : torch.tensor
        the bitwise log probability that the two Multi-Bernoulli are equal
        
    Notes
    -----
    xz and yz need not to have the same shape, but they should
    be broadcastable.
    """
    xp, yp, xn, yn = map(logsigmoid, (xz, yz, -xz, -yz))
    log_p0 = torch_logsumexp(xp + yn, xn + yp)
    log_p1 = torch_logsumexp(xp + yp, xn + yn)
    return log_p0, log_p1
    
def multi_bernoulli_subset(xz, yz):
    """
    Compute the bitwise log probability that the first Multi-Bernoulli
    is lower are equal to the second.
    
    Parameters
    ----------
    xz : torch.tensor
        the logits (before sigmoid) of the first Multi-Bernoulli
    yz : torch.tensor
        the logits (before sigmoid) of the second Multi-Bernoulli
        
    Returns
    -------
    log_p0 : torch.tensor
        the bitwise log probability of not subset
    log_p1 : torch.tensor
        the bitwise log probability of subset
        
    Notes
    -----
    xz and yz need not to have the same shape, but they should
    be broadcastable.
    """
    xp, yp, xn, yn = map(logsigmoid, (xz, yz, -xz, -yz))
    log_p0 = xp + yn
    log_p1 = torch_logsumexp(xp + yp, xn + yn, xn + yp)
    return log_p0, log_p1

def multi_bernoulli_activated_equality(xz, yz, az):
    """
    Compute the bitwise log probability that two Multi-Bernoulli are equal
    or that a third Multi-Bernoulli is one.
    
    Parameters
    ----------
    xz : torch.tensor
        the logits (before sigmoid) of the first Multi-Bernoulli
    yz : torch.tensor
        the logits (before sigmoid) of the second Multi-Bernoulli
    az : torch.tensor
        the logits of the third Multi-Bernoulli which act as an activation
        of the equality.
        
    Returns
    -------
    log_p0 : torch.tensor
        the bitwise log probability that the two Multi-Bernoulli are not equal
        and the third is zero.
    log_p1 : torch.tensor
        the bitwise log probability that the two Multi-Bernoulli are equal
        or the third is one.
        
    Notes
    -----
    xz and yz need not to have the same shape, but they should
    be broadcastable.
    """
    xp, yp, ap, xn, yn, an = map(logsigmoid, (xz, yz, az, -xz, -yz, -az))
    log_p0 = torch_logsumexp(an + xp + yn, an + xn + yp)
    log_p1 = torch_logsumexp(ap, an + xp + yp, an + xn + yn)
    return log_p0, log_p1

def multi_bernoulli_activated_subset(xz, yz, az):
    """
    Compute the bitwise log probability that the first Multi-Bernoulli
    is lower are equal to the second or that a third Multi-Bernoulli is one.
    
    Parameters
    ----------
    xz : torch.tensor
        the logits (before sigmoid) of the first Multi-Bernoulli
    yz : torch.tensor
        the logits (before sigmoid) of the second Multi-Bernoulli
    az : torch.tensor
        the logits of the third Multi-Bernoulli which act as an activation
        of the "subset".
        
    Returns
    -------
    log_p0 : torch.tensor
    log_p1 : torch.tensor
        
    Notes
    -----
    xz and yz need not to have the same shape, but they should
    be broadcastable.
    """
    xp, yp, ap, xn, yn, an = map(logsigmoid, (xz, yz, az, -xz, -yz, -az))
    log_p0 = an + xp + yn
    log_p1 = torch_logsumexp(ap, an + xp + yp, an + xn + yn, an + xn + yp)
    return log_p0, log_p1