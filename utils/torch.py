import torch
import numpy as np
from functools import reduce
from collections import OrderedDict

def torch_maximum(*tensors):
    return reduce(torch.max, tensors)

def torch_logsumexp(*tensors):
    maxes = torch_maximum(*tensors)
    return sum((tensor - maxes).exp() for tensor in tensors).log() + maxes

def torch_logsubexp(log_a, log_b):
    """
    Compute log(a-b) = log(exp(log(a))-exp(log(b)))
    This is unsound if a < b (log_a < log_b)
    The only thing this function does is removing
    the risk of overflow in the exponentials.
    log(a-b) = log(exp(log_a)*(1 - exp(log_b-log_a))) = log(1-exp(log_b-log_a)) + log_a
    """
    where = log_a <= log_b
    if torch.all(where):
        n = where.sum()
        total = np.prod(where.shape)
        msg = 'Cannot compute the log of the a-b if it is negative, {}/{} negative values found'
        raise ValueError(msg.format(n, total))
    return torch.log(1-torch.exp(log_b-log_a)) + log_a

def torch_max(tensor, dim=None, keepdims=False):
    if isinstance(dim, int):
        return tensor.max(dim, keepdim=keepdims)[0]
    if dim is None or set(dim)==set(range(tensor.dim())):
        if keepdims: return tensor.max().view(*[1]*tensor.dim())
        else: return tensor.max()
    dim = tuple(d+tensor.dim() if d < 0 else d for d in dim)
    for d in sorted(dim, reverse=True):
        tensor = tensor.max(d, keepdim=keepdims)[0]
    return tensor

def torch_lse(tensor, dim=None, keepdims=False):
    if dim is None: dim = tuple(range(tensor.dim()))
    maxes = torch_max(tensor, dim=dim, keepdims=True)
    lse = (tensor - maxes).exp().sum(dim=dim, keepdim=True).log() + maxes
    if not keepdims:
        dim = tuple(d+tensor.dim() if d < 0 else d for d in dim)
        for d in sorted(dim, reverse=True): lse = lse.squeeze(dim=d)
    return lse

def torch_lme(tensor, dim=None, keepdims=False):
    #log mean exp
    return torch_lse(tensor, dim, keepdims) - np.sum(np.log(tensor.shape))
    
def params_to_buffer(module):
    module._buffers.update(module._parameters)
    module._parameters = OrderedDict()
    for child in module._modules.values():
        params_to_buffer(child)
        
def buffer_no_grad(module):
    for p in module.buffers():
        p.requires_grad = False
    for child in module._modules.values():
        buffer_no_grad(child)

def torch_cast(x):
    if issubclass(x.dtype.type, np.floating):
        x = torch.tensor(x, dtype=torch.float32)
    elif issubclass(x.dtype.type, np.integer):
        x = torch.tensor(x, dtype=torch.int64)
    else: raise ValueError('can only cast floating or integer, got {}'.format(x.dtype))
    if torch.cuda.is_available(): x = x.cuda()
    return x

def _fbeta_gamma_term(beta, log_q, log_s):
    b2 = beta**2
    log_b2 = np.log(b2)
    c1 = np.log(1+b2) - log_b2
    c2 = log_q + log_b2
    return c1 - torch.nn.Softplus()(log_s-c2)

def torch_negative_log_fbeta_loss(log_r_hats, log_s_hats, beta, log_q):
    log_rho = log_r_hats.mean()
    log_s_hat = torch_lme(log_s_hats)
    log_g = _fbeta_gamma_term(beta, log_q, log_s_hat)
    log_fbeta = log_rho + log_g
    return -log_fbeta

def positive_loss_adaptative_l2_reg(loss, ratio, values):
    #model bad -> high loss -> high regularization
    #model good -> low loss -> low regularization
    reg = sum((v**2).sum() for v in values)
    alpha = ratio*float(loss/reg)
    return loss + alpha*reg