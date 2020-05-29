import torch
from radbm.utils.torch import torch_logsumexp
logsigmoid = torch.nn.LogSigmoid()

def poisson_binomial_equality(xz, yz):
    xp, yp, xn, yn = map(logsigmoid, (xz, yz, -xz, -yz))
    return torch_logsumexp(xp + yp, xn + yn)
    
def poisson_binomial_subset(xz, yz):
    xp, yp, xn, yn = map(logsigmoid, (xz, yz, -xz, -yz))
    return torch_logsumexp(xp + yp, xn + yn, xn + yp)

def poisson_binomial_activated_equality(xz, yz, az):
    xp, yp, ap, xn, yn, an = map(logsigmoid, (xz, yz, az, -xz, -yz, -az))
    return torch_logsumexp(ap, an + xp + yp, an + xn + yn)

def poisson_binomial_activated_subset(xz, yz, az):
    xp, yp, ap, xn, yn, an = map(logsigmoid, (xz, yz, az, -xz, -yz, -az))
    return torch_logsumexp(ap, an + xp + yp, an + xn + yn, an + xn + yp)