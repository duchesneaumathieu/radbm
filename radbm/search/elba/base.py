import torch
from radbm.search.base import BaseSDS
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
    log_p : torch.tensor
        the bitwise log probability that the two Multi-Bernoulli are equal
        
    Notes
    -----
    xz and yz need not to have the same shape, but they should
    be broadcastable.
    """
    xp, yp, xn, yn = map(logsigmoid, (xz, yz, -xz, -yz))
    return torch_logsumexp(xp + yp, xn + yn)
    
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
    log_p : torch.tensor
        the bitwise log probability
        
    Notes
    -----
    xz and yz need not to have the same shape, but they should
    be broadcastable.
    """
    xp, yp, xn, yn = map(logsigmoid, (xz, yz, -xz, -yz))
    return torch_logsumexp(xp + yp, xn + yn, xn + yp)

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
    log_p : torch.tensor
        the bitwise log probability that the two Multi-Bernoulli are equal
        or the third is one.
        
    Notes
    -----
    xz and yz need not to have the same shape, but they should
    be broadcastable.
    """
    xp, yp, ap, xn, yn, an = map(logsigmoid, (xz, yz, az, -xz, -yz, -az))
    return torch_logsumexp(ap, an + xp + yp, an + xn + yn)

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
    log_p : torch.tensor
        the bitwise log probability
        
    Notes
    -----
    xz and yz need not to have the same shape, but they should
    be broadcastable.
    """
    xp, yp, ap, xn, yn, an = map(logsigmoid, (xz, yz, az, -xz, -yz, -az))
    return torch_logsumexp(ap, an + xp + yp, an + xn + yn, an + xn + yp)

class EfficientLearnableBinaryAccess(BaseSDS, torch.nn.Module):
    """
    EfficientLearnableBinaryAccess (ELBA) is a base class for concrete models.
    Given a search data structure and two parametric encoding functions, one
    for the queries (fq) and one for the documents (fd), both producing a 
    Multi-Bernoulli code (i.e. in [0,1]^n) in its logits form (pre sigmoid).
    ELBM uses the structure to store and retrieve data using the Multi-Bernoulli
    code.
    
    Parameters
    ----------
    fq : torch.nn.Module
        The parametric function of the queries outputting in logits (pre sigmoid).
    fd : torch.nn.Module
        The parametric function of the documents outputting in logits (pre sigmoid).
    struct : BaseSDS subclass
        The structure used for storing and retrieval.
    optim : torch.optim (optional)
        The optimizer (minimizer) class used for the parametric function.
        (default torch.optim.Adam)
    lr : float (optional)
        The learning rate of the optimizer. (default 0.001)
    """
    def __init__(self, fq, fd, struct, optim=torch.optim.Adam, lr=0.001):
        torch.nn.Module.__init__(self)
        self.fq = fq
        self.fd = fd
        self.struct = struct
        self.optim = optim(self.parameters(), lr)
        
    def _log_sigmoid_pairs(self, logits):
        return torch.stack([logsigmoid(-logits), logsigmoid(logits)], dim=1)
        
    def batch_insert(self, documents, indexes, *args, **kwargs):
        """
        Insert the index of each documents in the data structure
        
        Parameters
        ----------
        documents : torch.tensor
            The documents to insert the first dim being the batch.
        indexes : iterable of hashable
            most notable example is a list of int. len(indexes) most
            be equal to len(documents).
        *args
            passed to self.struct.batch_insert
        **kwargs
            passed to self.struct.batch_insert
            
        Returns
        -------
        self
        """
        dmb = self._log_sigmoid_pairs(self.fd(documents))
        self.struct.batch_insert(dmb, indexes, *args, **kwargs)
        return self
        
    def batch_search(self, queries, *args, **kwargs):
        """
        Search in the data structure for the relevant indexes for each queries.
        
        Parameters
        ----------
        queries : torch.tensor
            The search queries, the first dim being the batch.
        *args
            passed to self.struct.batch_search
        **kwargs
            passed to self.struct.batch_search
            
        Returns
        -------
        indexes_list : list of (set or list)
            Is the list of the relevant indexes for each queries. 
            len(indexes_list) = len(queries).
        """
        qmb = self._log_sigmoid_pairs(self.fq(queries))
        return self.struct.batch_search(qmb, *args, **kwargs)
        
    def batch_itersearch(self, queries, *args, **kwargs):
        """
        Iteratively search in the data structure for the relevant
        indexes for each queries.
        
        Parameters
        ----------
        queries : torch.tensor
            The search queries, the first dim being the batch.
        *args
            passed to self.struct.batch_itersearch
        **kwargs
            passed to self.struct.batch_itersearch
            
        Returns
        -------
        generator_list : list of generators (of set or list)
            Each generator yield relevant indexes for the corresponding queries. 
            len(generator_list) = len(queries).
        """
        qmb = self._log_sigmoid_pairs(self.fq(queries))
        return self.struct.batch_itersearch(qmb, *args, **kwargs)
    
    def get_state(self):
        return {
            'f': self.state_dict(),
            'optim': self.optim.state_dict(),
            'struct': self.struct.get_state(),
        }
    
    def set_state(self, state):
        self.load_state_dict(state['f'])
        self.optim.load_state_dict(state['optim'])
        self.struct.set_state(state['struct'])
        return self