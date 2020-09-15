import torch
from torch.nn.utils import clip_grad_value_
from torch.distributions import Categorical
from radbm.utils.torch import torch_soft_hamming
from radbm.search.elba import EfficientLearnableBinaryAccess

def categorical_entropy(cat):
    """
    -(cat*cat.log()).sum() without the annoying 0*inf
    
    Parameters
    ----------
    cat : torch.Tensor (ndim==1)
        The parameter of a Categorical distribution.
        
    Returns
    -------
    ent : torch.Tensor (a single float)
        The entropy of the Categorical distribution.
    """
    return Categorical(probs=cat).entropy()
    
def mi_categorical_bernoulli(pos_cat, neg_cat, p):
    """
    Compute the Multual Information between a categorical and a bernoulli.
    This use the fact that I(C, B) = H(C) - pH(C | B=1) - (1-p)H(C | B=0)
    with C = Cat(pi) and B = Ber(p).
    
    Parameters
    ----------
    pos_cat : torch.tensor (ndim=1, pos_cat.sum()=1)
        The parameters of C | B=1
    neg_cat : torch.tensor (ndim=1, neg_cat.sum()=1)
        The parameters of C | B=0
    p : float
        The parameters of B
        
    Returns
    -------
    I : torch.tensor (a single float)
            The Mutual Information I(C, B)
    """
    cat = p*pos_cat + (1-p)*neg_cat
    ent = categorical_entropy(cat)
    pos_ent = categorical_entropy(pos_cat)
    neg_ent = categorical_entropy(neg_cat)
    return ent - p*pos_ent - (1-p)*neg_ent

class TriangularKernel(torch.nn.Module):
    """
    Helper Module, compute the triangular kernel.
    """
    def __init__(self, centroids, widths=None):
        super().__init__()
        if widths is None:
            widths = torch.tensor(1, dtype=centroids.dtype)
        self.register_buffer('centroids', centroids)
        self.register_buffer('widths', widths)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        shape = x.shape
        x = x.view(*shape, 1)
        centroids = self.centroids.view(*len(shape)*[1], -1)
        return self.relu(1 - (centroids-x).abs()/self.widths)
    
class MIHash(EfficientLearnableBinaryAccess):
    """
    MIHash as in "MIHash: Online Hashing with Mutual Information"
    by Fatih Cakir, Kun He, Sarah Adel Bargal and Stan Sclaroff.

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
    """
    def __init__(self, fq, fd, struct, nbits, match_prob, *args, **kwargs):
        super().__init__(fq, fd, struct, *args, **kwargs)
        self.match_prob = match_prob
        self.kernel = TriangularKernel(torch.arange(0,nbits+1))
    
    def step(self, q, d, match, l2_ratio=0):
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
            
        Returns
        -------
        loss : torch.Tensor (size 1)
            The loss (negative mutual information) of the current batch.
        """
        self.zero_grad()
        qsign = torch.tanh(self.fq(q))
        dsign = torch.tanh(self.fd(d))
        sh = torch_soft_hamming(qsign[:,None], dsign[None,:]) #shape = (#queries, #documents)
        bins = self.kernel(sh)
        pos_cat = bins[match].mean(dim=0)
        neg_cat = bins[~match].mean(dim=0)
        loss = -mi_categorical_bernoulli(pos_cat, neg_cat, self.match_prob)
        loss.backward()
        clip_grad_value_(self.parameters(), 5)
        self.optim.step()
        return loss