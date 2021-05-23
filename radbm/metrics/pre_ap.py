import torch
from radbm.metrics import user_cost_at_k_from_scores

def pre_average_precision_from_user_cost(uck):
    r"""
    Compute the pre average precision from the user cost at k (:math:`\mathrm{UC}_k`)
    using the following equation:
    
    .. math::
        \sum_{k=1}^{|r|} \frac{k}{\mathrm{UC}_k}
    
    Parameters
    ----------
    uck : torch.tensor (shape: (k,))
        The user cost at k, uck[k] must be :math:`\mathrm{UC}_k`.
        
    Returns
    -------
    pre_ap : torch.tensor (ndim: 0, dtype: float32)
        The pre-average precision.
    """
    ks = torch.arange(1, len(uck)+1, device=uck.device)
    pre_ap = (ks/uck).mean()
    return pre_ap 

def pre_average_precision(scores, relevants):
    """
    Compute the pre average precision from a scoring of each documents
    Whenever two (or more) documents share the same score, the ordering 
    defined by this scoring is no longer antisymmetric. Thus order
    defined by this scoring becomes a total pre-order. This methods uses
    the user cost at k to compute the pre-AP, a generalization of the AP
    for pre-ranking (i.e. ranking defined with a total pre-order). If each
    score is unique then this is equivalent to the original AP.
    
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
    pre_ap : torch.tensor (ndim: 0, dtype: float32)
        The pre average precision.
        
    Raises
    ------
    ValueError
        If relevants.dtype is torch.bool and len(relevants) != len(scores)
    TypeError
        If relevants.dtype is not torch.bool or torch.int64
    """
    uck = user_cost_at_k_from_scores(scores, relevants)
    return pre_average_precision_from_user_cost(uck)

def pre_mean_average_precision(scores, relevants):
    """
    Parameters
    ----------
    scores : torch.tensor (ndim: 2)
        scores[i,j] is the scores between the ith query and the jth documents.
    relevants : torch.tensor (ndim: 2, dtype: bool) or list of torch.tensors (ndim: 1, dtype: int)
        The adjacency matrix or adjacency list of relevance. If adjacency matrix, relevants[i,j] is
        a boolean indicating if the jth document is relevant to the ith query. Otherwise, if adjacency
        list, relevants[i] are the indices of the relevant documents.
        
    Returns
    -------
    pre_map : torch.tensor (dtype: float, ndim: 0)
    """
    pre_aps = [pre_average_precision(s, r) for s, r in zip(scores, relevants)]
    pre_map = torch.tensor(pre_aps).mean()
    return pre_map

def batch_pre_average_precision(queries, documents, relevants, scoring_function, batch_size):
    """
    Computes pre_average_precision in batches to make the most out of a GPU.
    
    Parameters
    ----------
    queries : torch.tensor
    documents : torch.tensor
    relevants : iterator of torch.tensor (dtype: int64 or bool)
    scoring_function : callable
        scoring_function takes a batch of query and a batch of documents and computes the
        score if each pairs. The function should be broadcastable friendly, i.e.
        scoring_function(queries[:,None], documents[None]) should make sense.
    batch_size : int
        The size that will be used to split the queries. This is only used for memory purposes.
        In other words, this number should be as high as the memory allows it. 
    
    Returns
    -------
    pre_aps : torch.tensor (dtype: torch.float32, shape: (len(queries),))
        The pre average precision of each query.
    
    Raises
    ------
    ValueError
        If len(queries) != len(relevants).
    """
    
    if len(queries) != len(relevants):
        msg = 'len(queries) and len(relevants) must be equal, got {} and {} respectively.'
        raise ValueError(msg.format(len(queries), len(relevants)))
    pre_aps = torch.zeros(len(queries), dtype=torch.float32, device=queries.device)
    queries = queries[:,None]
    documents = documents[None]
    nfull_batch = len(queries)//batch_size
    nbatch = nfull_batch if len(queries) % batch_size == 0 else nfull_batch + 1
    for i in range(nbatch):
        beg = i*batch_size
        end = min(beg + batch_size, len(pre_aps))
        scores = scoring_function(queries[beg:end], documents)
        batch_relevants = relevants[beg:end]
        sub_pre_aps_list = [pre_average_precision(s, r) for s, r in zip(scores, batch_relevants)]
        pre_aps[beg:end] = torch.tensor(sub_pre_aps_list, device=queries.device)
    return pre_aps