import torch
import numpy as np
from functools import reduce
from collections import OrderedDict

def torch_hamming(code1, code2, dim=-1, keepdims=False):
    #obsolete, a way better implementation in radbm.metrics.hamming
    if isinstance(dim, int): dim = (dim,)
    n1 = int(np.prod([code1.shape[d] for d in dim]))
    n2 = int(np.prod([code2.shape[d] for d in dim]))
    if n1 != n2: raise ValueError()
    c = (code1==code2).sum(dim=dim, keepdims=keepdims)
    return n1 - c

def torch_soft_hamming(v, w, dim=-1, keepdims=False):
    if isinstance(dim, int): dim = (dim,)
    n1 = int(np.prod([v.shape[d] for d in dim]))
    n2 = int(np.prod([w.shape[d] for d in dim]))
    if n1 != n2: raise ValueError()
    dot = (v*w).sum(dim=dim, keepdims=keepdims)
    return .5*(n1 - dot)

def torch_maximum(*tensors):
    return reduce(torch.max, tensors)

def torch_logsumexp(*tensors):
    maxes = torch_maximum(*tensors)
    maxes = torch.where(maxes==-np.inf, torch.zeros_like(maxes), maxes)
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
    maxes = torch.where(maxes==-np.inf, torch.zeros_like(maxes), maxes)
    lse = (tensor - maxes).exp().sum(dim=dim, keepdim=True).log() + maxes
    if not keepdims:
        if isinstance(dim, (tuple,list)):
            dim = tuple(d+tensor.dim() if d < 0 else d for d in dim)
            for d in sorted(dim, reverse=True): lse = lse.squeeze(dim=d)
        else: lse = lse.squeeze(dim=dim)
    return lse

def torch_lme(tensor, dim=None, keepdims=False):
    #log mean exp
    return torch_lse(tensor, dim, keepdims) - np.sum(np.log(tensor.shape))

def torch_categorical_entropy(log_p, dim=-1, keepdims=False):
    log_p = torch.where(log_p==-np.inf, torch.zeros_like(log_p), log_p)
    return -(log_p.exp()*log_p).sum(dim=dim, keepdims=keepdims)

def torch_multibernoulli_entropy(log_p0, log_p1, dim=-1, keepdims=False):
    log_p0 = torch.where(log_p0==-np.inf, torch.zeros_like(log_p0), log_p0)
    log_p1 = torch.where(log_p1==-np.inf, torch.zeros_like(log_p1), log_p1)
    return -(log_p0.exp()*log_p0 +log_p1.exp()*log_p1).sum(dim=dim, keepdims=keepdims)
    
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

def torch_complex_polar(a, b):
    #return r, theta
    r = torch.sqrt(a**2 + b**2)
    theta = torch.atan2(b, a)
    return r, theta

def torch_complex_log(a, b):
    r, theta = torch_complex_polar(a, b)
    return r.log(), theta #<-- cartesian coordinate of log(a+ib) = log(r) + i*theta

def torch_complex_exp(a, b):
    exp_a = a.exp()
    return exp_a*torch.cos(b), exp_a*torch.sin(b) #Euler

def torch_complex_prod(a, b, dim=None):
    #use log space because addition of complex numbers is elementwise 
    #(i.e. real with real and complex with complex)
    if dim is None: dim = tuple(range(len(a.shape)))
    a, b = torch_complex_log(a, b)
    return torch_complex_exp(a.sum(dim=dim), b.sum(dim=dim))

class PoissonBinomialSum(torch.nn.Module):
    def __init__(self, n, roots_requires_grad=False):
        super().__init__()
        self.n = n
        w = 2*np.pi/(n+1)
        #use python imaginary arithmetic
        roots = np.array([np.e**(w*l*1j) for l in range(n+1)])
        roots_real = torch.tensor(roots.real, requires_grad=roots_requires_grad)
        roots_imag = torch.tensor(roots.imag, requires_grad=roots_requires_grad)
        self.register_buffer('roots_real', roots_real)
        self.register_buffer('roots_imag', roots_imag)
        
    def forward(self, q):
        q_view = q.view(1,self.n)
        z_real = self.roots_real.view(self.n+1,1)*q_view - q_view + 1
        z_imag = self.roots_imag.view(self.n+1,1)*q_view
        x_real, x_imag = torch_complex_prod(z_real, z_imag, dim=1)
        x = torch.stack([x_real, x_imag], dim=1)
        pb = torch.fft(x,1)
        return pb[:,0]/(self.n+1)
    
def torch_cast_cpu(x, max_floating_point=32):
    if not isinstance(x, np.ndarray): return torch.tensor(x)
    dtype = x.dtype.type
    if max_floating_point == 64 and issubclass(dtype, np.float128):
        return torch.tensor(x, dtype=torch.float64)
    if max_floating_point == 32 and issubclass(dtype, (np.float128, np.float64)):
        return torch.tensor(x, dtype=torch.float32)
    if max_floating_point == 16 and issubclass(dtype, (np.float128, np.float64, np.float32)):
        return torch.tensor(x, dtype=torch.float16)
    return torch.tensor(x)

def torch_cast(x, max_floating_point=32):
    x = torch_cast_cpu(x, max_floating_point=max_floating_point)
    if torch.cuda.is_available(): x = x.cuda()
    return x

class TorchNumpyRNG(object):
    def __init__(self, rng=np.random):
        self.rng=rng
        self.device='cpu'
        self.backend='numpy'
        self.numpy()
        
    def apply_switch(self, wrap):
        bypass = {'seed', 'get_state', 'set_state'}
        for attr in np.random.__all__:
            if attr in bypass: setattr(self, attr, getattr(self.rng, attr))
            elif hasattr(self.rng, attr) and callable(getattr(self.rng, attr)):
                setattr(self, attr, wrap(getattr(self.rng, attr)))
    
    def torch_cpu_wrap(self, f):
        def wrapped(*args, **kwargs):
            a = f(*args, **kwargs)
            return torch_cast_cpu(a)
        return wrapped
    
    def torch_cuda_wrap(self, f):
        def wrapped(*args, **kwargs):
            a = f(*args, **kwargs)
            return torch_cast(a)
        return wrapped
    
    def numpy_cpu_wrap(self, f):
        return f
    
    def torch(self):
        if self.device=='cpu':
            self.apply_switch(self.torch_cpu_wrap)
        elif self.device=='cuda':
            self.apply_switch(self.torch_cuda_wrap)
        self.backend = 'torch'
        return self
    
    def numpy(self):
        if self.device=='cuda':
            raise ValueError('cannot use numpy backend with cuda, use rng.cpu() first')
        self.apply_switch(self.numpy_cpu_wrap)
        self.backend = 'numpy'
        return self
    
    def cpu(self):
        if self.backend=='numpy':
            self.apply_switch(self.numpy_cpu_wrap)
        elif self.backend=='torch':
            self.apply_switch(self.torch_cpu_wrap)
        self.device = 'cpu'
        return self
    
    def cuda(self):
        if self.backend=='numpy':
            raise ValueError('cannot use cuda with numpy backend, use rng.torch() first')
        self.apply_switch(self.torch_cuda_wrap)
        self.device = 'cuda'
        return self
    
    def get_state_hash(self):
        s = self.get_state()
        #don't hash string (i.e. s[0]) since string's hash are salted at each session
        #s[0] is always 'MT19937' anyway
        return hash(tuple(s[1])+s[2:])
    
    def __repr__(self):
        s = 'TorchNumpyRNG\nBackend: [{} ({})]\nState\'s hash: {}'
        return s.format(
            self.backend,
            self.device,
            self.get_state_hash()
        )
    
class TorchNumpy(object):
    def __init__(self, data, backend, device):
        self.backend = backend
        self.device = device
        if backend not in {'numpy', 'torch'}:
            raise ValueError('backend must be numpy or torch, got {}'.format(backend))
            
        if backend=='numpy' and isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif backend=='torch' and isinstance(data, np.ndarray):
            data = torch_cast_cpu(data)
            
        if device=='cpu': self.data = data
        elif device=='cuda':
            if backend=='torch': self.data = data.cuda()
            else: raise ValueError('cannot use cuda with numpy backend')
        else: raise ValueError('device must be cpu or cuda, got {}'.format(device))
        
    def cuda(self):
        self.data = self.data.cuda()
        
    def cpu(self):
        self.data = self.data.cpu()
        
    def torch(self):
        self.data = torch_cast_cpu(self.data)
        
    def numpy(self):
        self.data = self.data.numpy()