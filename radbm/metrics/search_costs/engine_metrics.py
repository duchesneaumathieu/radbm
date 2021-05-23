import torch
from .user_cost import user_cost_at_k_from_counts

def costs_at_k(candidates, relevants, N=None, device='cpu'):
    """
    Computes the user cost at k and the engine cost at k.
    
    Parameters
    ----------
    candidates : iterable of tuple (<set>, <float>)
        Iterable of candidates (sets) and total engine cost (float).
    relevants : set
        The set of relevant documents.
    N : int (optional)
        The size of the database. If the candidates does not contain all the relevant documents (halting),
        N must be provided. (default: None)
    device : str (optional)
        Can be 'cpu', 'cuda' or any valid torch.device. (default: 'cpu')
    
    Returns
    -------
    user_cost_at_k : torch.tensor (dtype: float, ndim=1)
        user_cost_at_k[k] is the :math:`\mathrm{UC}_k`
    engine_cost_at_k : torch.tensor (dtype: float, ndim=1)
        engine_cost_at_k[k] is the :math:`\mathrm{EC}_k`
        
    Raises
    ------
    ValueError
        If the candidates does not contain all relevants documents and N (the database size) is not given.
    RuntimeError
        If the candidates does not contain all relevants documents and the candidates are larger or equal to N. 
    """
    relevants = relevants.copy() #because we will destroy it.
    candidates_size = list()
    relevant_counts = list()
    relevant_candidates_indices = list()
    k, eck = 0, torch.zeros(len(relevants), device=device)
    t = -1 # in case candidates is empty
    for t, (cand, cost) in enumerate(candidates):
        new = relevants.intersection(cand)
        relevants -= new
        candidates_size.append(len(cand))
        if new:
            eck[k:k+len(new)] = cost; k += len(new)
            relevant_counts.append(len(new))
            relevant_candidates_indices.append(t)
    if relevants:
        if N is None:
            msg = f'Not all relevant documents were found in the candidates and N is None, {relevants} are missing.'
            raise ValueError(msg)
        M = sum(candidates_size)
        if N < M + len(relevants):
            msg = (f'Wrong database size. {M} documents where retrieved and {len(relevants)}'
                   f' relevant documents are not yet found.'
                   f' Still, the database size is N = {N} < {M} + {len(relevants)} = {M+len(relevants)}.')
            raise RuntimeError(msg)
        eck[k:] = 0 #engine cost is zero when halting
        relevant_counts.append(len(relevants)) #add what is left
        relevant_candidates_indices.append(t+1)
        candidates_size.append(N - M)
    candidates_size = torch.tensor(candidates_size, dtype=torch.int64, device=device)
    relevant_counts = torch.tensor(relevant_counts, dtype=torch.int64, device=device)
    relevant_candidates_indices = torch.tensor(relevant_candidates_indices, dtype=torch.int64, device=device)
    uck = user_cost_at_k_from_counts(candidates_size, relevant_counts, relevant_candidates_indices)
    return uck, eck

def total_cost_at_k(candidates, relevants, N=None, device='cpu'):
    """
    Computes the total cost at k, i.e., the sum of the user cost at k and the engine cost at k.
    
    Parameters
    ----------
    candidates : iterable of tuple (<set>, <float>)
        Iterable of candidates (set) and total engine cost (float).
    relevants : set
        The set of relevant documents.
    N : int (optional)
        The size of the database. If the candidates does not contain all the relevant documents (halting),
        N must be provided. (default: None)
    device : str (optional)
        Can be 'cpu', 'cuda' or any valid torch.device. (default: 'cpu')
    
    Returns
    -------
    total_cost_at_k : torch.tensor (dtype: float, ndim=1)
        user_cost_at_k[k] is the :math:`\mathrm{UC}_k`
        
    Raises
    ------
    ValueError
        If the candidates does not contain all relevants documents and N (the database size) is not given.
    RuntimeError
        If the candidates does not contain all relevants documents and the candidates are larger or equal to N. 
    """
    uck, eck = costs_at_k(candidates, relevants, N=N, device=device)
    return uck + eck

def total_cost_ratio_from_costs(uck, eck, N):
    r"""
    Computes the total cost ratio from the user cost at k :math:`\mathrm{UC}_k`
    and the engine cost at k :math:`\mathrm{EC}_k` using this equation:
    
    .. math::
        \mathrm{TCR} = \frac{2}{N + 1} \frac{1}{|r|}\sum_{k=1}^{|r|} \mathrm{UC}_k + \mathrm{EC}_k
        
    where r is the set of relevant documents, thus :math:`|r|` is the number of relevant documents.
    
    Parameters
    ----------
    uck : torch.tensor (dtype: float, ndim=1)
        user_cost_at_k[k] is the :math:`\mathrm{UC}_k`
    eck : torch.tensor (dtype: float, ndim=1)
        engine_cost_at_k[k] is the :math:`\mathrm{EC}_k`
    N : int
        The database size.
        
    Returns
    -------
    tcr : torch.tensor (ndim: 0, dtype: float32)
        The total cost ratio.
    """
    return 2 / (N + 1) * (uck + eck).mean() 

def total_cost_ratio(candidates, relevants, N, device='cpu'):
    r"""
    Computes the total cost ratio.
    
    Parameters
    ----------
    candidates : iterable of tuple (<set>, <float>)
        Iterable of candidates (sets) and total engine cost (float).
    relevants : set
        The set of relevant documents.
    N : int
        The size of the database.
    device : str (optional)
        Can be 'cpu', 'cuda' or any valid torch.device. (default: 'cpu')
        
    Returns
    -------
    tcr : torch.tensor (ndim: 0, dtype: float32)
        The total cost ratio.
    """
    uck, eck = costs_at_k(candidates, relevants, N=N, device=device)
    return total_cost_ratio_from_costs(uck, eck, N)