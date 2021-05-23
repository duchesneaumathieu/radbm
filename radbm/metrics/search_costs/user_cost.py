import torch
import numpy as np

def user_cost_at_k_original(candidates, relevants):
    """
    Parameters
    ----------
    candidates : iterable of set
    relevants : set
    
    Returns
    -------
    user_cost_at_k : torch.tensor (dtype: float, ndim=1)
        user_cost_at_k[k] is the :math:`\mathrm{UC}_k`
    """
    relevants = relevants.copy() #because we will destroy it.
    lr = len(relevants)
    tck = torch.zeros(lr, dtype=torch.float32)
    for c in candidates:
        new = relevants.intersection(c)
        relevants -= new
        lc = len(c)
        ln = len(new)
        lr = len(relevants)
        end = -lr if lr else None
        tck[-lr-ln:end] += torch.arange(1, ln+1, dtype=torch.float32)*(lc+1)/(ln+1)
        if lr:
            tck[-lr:] += lc
        if not relevants:
            break
    if relevants:
        msg = 'Not all relevant documents were found in the candidates, {} are missing.'
        raise ValueError(msg.format(relevants))
    return tck

def user_cost_at_k_from_counts(candidates_size, relevant_counts, relevant_candidates_indices=None):
    """
    Parameters
    ----------
    candidates_size : torch.tensor (dtype: int, ndim: 1)
        Need not to contain all set of candidates. Only the firsts that contain all
        the relevant documents.
    relevant_counts : torch.tensor (dtype: int, ndim: 1)
        relevant_counts[i] is the number of relevant documents found in the the
        kth set of candidates where k = relevant_candidates_indices[i]. If 
        relevant_candidates_indices is None (default) then k = i. 
    relevant_candidates_indices: torch.tensor (dtype: int, ndim: 1) or None
        The indices of set of candidates at which we can found at least one relevant
        documents. Setting to None is equivalent to setting to
        torch.arange(len(candidates_size)). (default: None)
        
    Returns
    -------
    user_cost_at_k : torch.tensor (dtype: float32, ndim: 1)
        user_cost_at_k[k] is the :math:`\mathrm{UC}_k`
        
    Raises
    ------
    ValueError
        If len(relevants_counts) != len(candidates_size) and relevant_candidates_indices is None.
        
    Notes
    -----
    The zeros of relevant_counts can be safely removed. However, if so, it is mandatory
    to provide relevant_candidates_indices.
    """
    if relevant_candidates_indices is None:
        if len(relevant_counts) != len(candidates_size):
            msg = ('Missing relevant_counts got {} be expected len(candidates_size) = {},'
                   ' if the counts of zeros are absent, it is mandatory to provide relevant_candidates_indices.')
            raise ValueError(msg.format(len(relevant_counts), len(candidates_size)))
        relevant_candidates_indices = torch.where(relevant_counts != 0)
        relevant_counts = relevant_counts[relevant_candidates_indices]
    
    rcs = candidates_size[relevant_candidates_indices] #relevant candidates' size
    v = (rcs+1)/(relevant_counts+1).float()
    h_at_t = candidates_size.cumsum(dim=0)[relevant_candidates_indices] - rcs
    
    #repeat_v = torch.repeat_interleave(v, relevant_counts) #slow...
    #h_at_k = torch.repeat_interleave(h_at_t, relevant_counts) #very slow... zzz
    #ranges = torch.cat([torch.arange(1, i+1, device=rcs.device) for i in relevant_counts])
    
    #move to numpy because torch is not up to the task.
    v = v.cpu().numpy()
    h_at_t = h_at_t.cpu().numpy()
    relevant_counts = relevant_counts.cpu().numpy()
    repeat_v = np.repeat(v, relevant_counts)
    h_at_k = np.repeat(h_at_t, relevant_counts)
    ranges = np.concatenate([np.arange(1, i+1) for i in relevant_counts])
    uck = h_at_k + repeat_v*ranges
    return torch.tensor(uck, dtype=torch.float32, device=rcs.device) #cast back to torch

def user_cost_at_k(candidates, relevants, device='cpu'):
    """
    Parameters
    ----------
    candidates : iterable of set
    relevants : set
    device : str (optional)
        Can be 'cpu', 'cuda' or any valid torch.device. (default: 'cpu')
    
    Returns
    -------
    user_cost_at_k : torch.tensor (dtype: float, ndim=1)
        user_cost_at_k[k] is the :math:`\mathrm{UC}_k`
        
    Notes
    -----
    This function is equivalent to user_cost_at_k_original but it uses vectorization
    to speedup the computation.
    """
    relevants = relevants.copy() #because we will destroy it.
    candidates_size = list()
    relevant_counts = list()
    relevant_candidates_indices = list()
    for t, c in enumerate(candidates):
        new = relevants.intersection(c)
        relevants -= new
        candidates_size.append(len(c))
        if new:
            relevant_counts.append(len(new))
            relevant_candidates_indices.append(t)
    if relevants:
        msg = 'Not all relevant documents were found in the candidates, {} are missing.'
        raise ValueError(msg.format(relevants))
    candidates_size = torch.tensor(candidates_size, dtype=torch.int64, device=device)
    relevant_counts = torch.tensor(relevant_counts, dtype=torch.int64, device=device)
    relevant_candidates_indices = torch.tensor(relevant_candidates_indices, dtype=torch.int64, device=device)
    tck = user_cost_at_k_from_counts(candidates_size, relevant_counts, relevant_candidates_indices)
    return tck

def assert_relevants_dtype(scores, relevants):
    if relevants.dtype is torch.int64:
        if len(relevants) != len(set(relevants.tolist())):
            relevants = torch.unique(relevants)
    elif relevants.dtype is torch.bool:
        if len(relevants) != len(scores):
            msg = 'relevants.dtype is torch.bool but len(relevants) != len(scores), got {} and {} respectively.'
            raise ValueError(msg.format(len(relevants), len(scores)))
    else:
        msg = 'relevants.dtype is {} but it should be torch.bool or torch.int64'
        raise TypeError(msg.format(type(relevants.dtype)))
    return relevants
        
def user_cost_at_k_from_scores(scores, relevants):
    """
    Compute the user cost at k from a scoring of the documents. The pre-order defined by
    the scoring gives the sequences of candidates. This function avoid the overhead
    of computing the sequences of candidates and pass it to user_cost_at_k which would
    in the end only do counting. 
    
    Parameters
    ----------
    scores : torch.tensor (ndim: 1)
        The score of each documents.
    relevants : torch.tensor (ndim: 1, torch.int64)
        The dtype should be torch.int64 or torch.bool because scores[relevants]
        needs to give the scores of the relevant documents. If relevants'dtype is
        torch.int64 the duplicates are removed. 
        
    Returns
    -------
    uck : torch.tensor (dtype: float32, shape: (k,))
        uck[i] is the user cost at i.
        
    Raises
    ------
    ValueError
        If relevants.dtype is torch.bool and len(relevants) != len(scores)
    TypeError
        If relevants.dtype is not torch.bool or torch.int64
    """
    relevants = assert_relevants_dtype(scores, relevants)
    unique_scores, candidates_size = torch.unique(scores, sorted=True, return_counts=True)
    relevant_unique_scores, relevant_counts = torch.unique(scores[relevants], sorted=True, return_counts=True)
    relevant_candidates_indices = torch.searchsorted(unique_scores, relevant_unique_scores)
    #for older torch.__version__:
    #relevant_candidates_indices = torch.where((unique_scores==relevant_unique_scores[:,None]).any(dim=0))[0] 
    
    tck = user_cost_at_k_from_counts(candidates_size, relevant_counts, relevant_candidates_indices)
    return tck