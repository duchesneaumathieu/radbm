import torch
import torch
import numpy as np
from .distance_counting import conditional_distance_counts

def hamming_distance(x, y, dim=-1, keepdim=False):
    """
    Compute the Hamming distance.
    
    Parameters
    ----------
    x : torch.Tensor (dtype=torch.bool)
    y : torch.Tensor (dtype=torch.bool)
    dim : int (optional)
        The dimension along which to compute the hamming distance. (default: -1)
    keepdim : bool (optional)
        Whether to keep the reduced dimension. (default: False)
    
    Returns
    -------
    z : torch.Tensor (dtype=torch.int64)
        The Hamming distance between x and y
    """
    
    z = (x^y).sum(dim=dim, keepdim=keepdim)
    return z

def membership_hamming_cost(x, y):
    """
    Compute the membership Hamming cost, i.e., the minimum
    Hamming distance between x's code and the l y's codes.
    
    Parameters
    ----------
    x : torch.Tensor (dtype=torch.bool, shape=(..., n))
    y : torch.Tensor (dtype=torch.bool, shape=(..., l, n))
        x.unsqueeze(-2) and y should be broadcastable.
    
    Returns
    -------
    z : torch.Tensor (dtype=torch.int64)
        The membership Hamming cost between x and y.
    """
    return (x[..., None, :] ^ y).sum(dim=-1).min(dim=-1)[0]

def intersection_hamming_cost(x, y):
    """
    Compute the intersection Hamming cost, i.e., the minimum
    Hamming distance between the k x's code and the l y's codes.
    
    Parameters
    ----------
    x : torch.Tensor (dtype=torch.bool, shape=(..., k, n))
    y : torch.Tensor (dtype=torch.bool, shape=(..., l, n))
        x[..., :, None, :] and y[..., None, :, :] should be broadcastable.
    
    Returns
    -------
    z : torch.Tensor (dtype=torch.int64)
        The intersection Hamming cost between x and y.
    """
    return (x[..., :, None, :] ^ y[..., None, :, :]).sum(dim=-1).flatten(start_dim=-2).min(dim=-1)[0]

def superset_hamming_cost(x, y):
    """
    Compute the superset Hamming cost.
    
    Parameters
    ----------
    x : torch.Tensor (dtype=torch.bool, shape=(..., k, n))
    y : torch.Tensor (dtype=torch.bool, shape=(..., l, n))
        x[..., :, None, :] and y[..., None, :, :] should be broadcastable.
    
    Returns
    -------
    z : torch.Tensor (dtype=torch.int64)
        The superset Hamming cost between x and y.
    """
    return (x[..., :, None, :] ^ y[..., None, :, :]).sum(dim=-1).min(dim=-1)[0].max(dim=-1)[0]

def conditional_hamming_counts(documents, queries, relevances, batch_size=100):
    """
    Compute the count of relevant and not relevant match w.r.t all Hamming distance.
    
    Parameters
    ----------
    documents : torch.Tensor (2D, dtype=torch.bool)
        The binary reprentation of a batch of documents (database).
        documents[i] is the ith document. documents.shape[1] must be 
        equal to queries.shape[1]. (should also be on the same device as queries)
    queries : torch.Tensor (2D, dtype=torch.bool)
        The binary reprentation of a batch of queries. queries[i]
        is the ith query. queries.shape[1] must be equal to
        documents.shape[1]. (should also be on the same device as documents)
    relevances : list of set of int
        len(relevances) must be len(queries). For each corresponding query,
        it give the set of relevant documents (given by its index). Explicitly,
        j in relevances[i] iff query[i] matches with documents[j].
    batch_size : int (optional)
        The number of query for which we compute the Hamming distance at a time.
        if it is to big the results might not fit in RAM (or on the GPU). (default 100)
        
    Returns
    -------
    relevant_counts : torch.Tensor (1D, dtype=torch.float)
        len(relevant_counts) = queries.shape[1] + 1 (also equal to documents.shape[1] + 1) and
        relevant_dcounts[i] is the number of relevant documents at Hamming distance i.
    irrelevant_counts : torch.Tensor (1D, dtype=torch.float)
        len(irrelevant_countst) = queries.shape[1] + 1 (also equal to documents.shape[1] + 1) and
        irrelevant_counts[i] is the number of irrelevant documents be at Hamming distance i.
        
    Notes
    -----
    This assume that each sets in relevances is small compared to len(documents) and should
    be used on a GPU otherwise it is quite slow.
    """
    return conditional_distance_counts(documents, queries, relevances, hamming_distance, documents.shape[1], batch_size=batch_size)

def conditional_counts_to_pr_curve(relevant_counts, irrelevant_counts, return_valid_dists=False):
    total_counts = relevant_counts + irrelevant_counts
    tcs = total_counts.cumsum(dim=0)
    rcs = relevant_counts.cumsum(dim=0)
    recalls = rcs.float() / sum(relevant_counts)
    dists = torch.where(tcs != 0)[0]
    precisions = (torch.zeros_like(recalls)-1).log() #nans_like
    precisions[dists] = rcs[dists].float()/tcs[dists]
    if return_valid_dists:
        return dists, precisions[dists], recalls[dists]
    return precisions, recalls

def hamming_pr_curve(documents, queries, relevances, batch_size=100, return_valid_dists=False):
    """
    Compute the precision-recall curve w.r.t the Hamming distance. I.e. it
    computes the precision and recall for each Hamming distance decision
    thresholds.  
    
    Parameters
    ----------
    documents : torch.Tensor (2D, dtype=torch.bool)
        The binary reprentation of a batch of documents (database).
        documents[i] is the ith document. documents.shape[1] must be 
        equal to queries.shape[1]. (should be on the same device as queries)
    queries : torch.Tensor (2D, dtype=torch.bool)
        The binary reprentation of a batch of queries. queries[i]
        is the ith query. queries.shape[1] must be equal to
        documents.shape[1]. (should be on the same device as documents)
    relevances : list of set of int
        len(relevances) must be len(queries). For each corresponding query,
        it give the set of relevant documents (given by its index). Explicitly,
        j in relevances[i] iff query[i] matches with documents[j].
    batch_size : int (optional)
        The number of query for which we compute the Hamming distance at a time.
        if it is to big the results might not fit in RAM (or on the GPU). (default 100)
    return_valid_dists : bool (optional)
        Some dists might have an undefined precision. In those case the returned value 
        will be nan by default. If return_valid_dists is True those value won't be there
        and dists will be returned with precision and recall. 
        See the returns section for more info. (default False)
        
    Returns
    -------
    dists : torch.Tensor (1D, dtype=torch.int64) if return_valid_dists is True
        Only present if return_valid_dists is True. It correspond to the valid distances
        where the precision is define.
    precisions : torch.Tensor (1D, dtype=torch.float)
        len(precisions) = queries.shape[1] + 1 (also equal to documents.shape[1] + 1) and
        precision[i] is the precison w.r.t. a Hamming distance of i if return_valid_dists
        is False otherwise, len(precisions) = len(dists) and precisions[i] is the precision
        w.r.t a Hamming distance of dists[i].
    recalls : torch.Tensor (1D, dtype=torch.float)
        len(recalls) = queries.shape[1] + 1 (also equal to documents.shape[1] + 1) and
        recalls[i] is the recall w.r.t. to a Hamming distance of i if return_valid_dists
        is False otherwise, len(recalls) = len(dists) and recalls[i] is the recall
        w.r.t a Hamming distance of dists[i].
        
    Notes
    -----
    This assume that each sets in relevances is small compared to len(documents) and should
    be used on a GPU otherwise it is quite slow.
    """
    relevant_counts, irrelevant_counts = conditional_hamming_counts(
        documents, queries, relevances, batch_size=batch_size)
    return conditional_counts_to_pr_curve(
        relevant_counts, irrelevant_counts, return_valid_dists=return_valid_dists)