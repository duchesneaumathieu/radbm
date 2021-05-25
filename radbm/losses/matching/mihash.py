import torch
from torch.distributions import Categorical
from radbm.utils.torch import torch_soft_hamming
from .utils import check_shape_helper

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
    
class MIHashMatchingLoss(object):
    r"""
    MIHashMatchingLoss as in `MIHash: Online Hashing with Mutual Information <https://arxiv.org/abs/1703.08919>`__.

    Parameters
    ----------
    nbits : int
        The number of bits in the codes.
    match_prob : float (in [0,1])
        The probability that there is a match given a random query
        and a random document.
    """
    def __init__(self, nbits, match_prob):
        self.match_prob = match_prob
        self.kernel = TriangularKernel(torch.arange(0,nbits+1))
    
    def __call__(self, queries_logits, documents_logits, r, step=0):
        """
        Parameters
        ----------
        q : torch.Tensor
            A batch of queries.
        d : torch.Tensor
            A batch of documents.
        r : torch.Tensor (dtype: torch.bool, ndim: 1 or 2)
            A matrix (block mode) (2D tensor) with r[i,j] indicating if q[i] match with d[j] or a vector (1D tensor) with
            r[i] indicating if q[i] match with d[i]
            
        Returns
        -------
        loss : torch.Tensor (size 1)
            The loss (negative mutual information) of the current batch.
        """
        if queries_logits.ndim != 2:
            msg = f'queries_logits.ndim must be 2, got {queries_logits.ndim}.'
            raise ValueError(msg)
        if documents_logits.ndim != 2:
            msg = f'documents_logits.ndim must be 2, got {documents_logits.ndim}.'
            raise ValueError(msg)
        if r.ndim not in {1, 2}:
            msg = f'r.ndim must be 1 (normal) or 2 (block), got {r.ndim}.'
            raise ValueError(msg)
        if r.dtype != torch.bool:
            msg = f'r must be boolean, got {r.dtype}.'
            raise TypeError(msg)
            
        block = check_shape_helper(queries_logits, documents_logits, r)
        if block:
            queries_logits = queries_logits.unsqueeze(1)
            documents_logits = documents_logits.unsqueeze(0)
            
        qsign = torch.tanh(queries_logits)
        dsign = torch.tanh(documents_logits)
        sh = torch_soft_hamming(qsign, dsign) #shape = (#queries, #documents) or (bs,)
        bins = self.kernel(sh)
        pos_cat = bins[r].mean(dim=0)
        neg_cat = bins[~r].mean(dim=0)
        loss = -mi_categorical_bernoulli(pos_cat, neg_cat, self.match_prob)
        return loss