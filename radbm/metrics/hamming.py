import torch
import numpy as np

def hamming_distance(x, y, *args, **kwargs):
    """
    Compute the Hamming distance.
    
    Parameters
    ----------
    x : torch.Tensor (dtype=torch.bool)
    y : torch.Tensor (dtype=torch.bool)
    *args
        Passed to sum
    *kwargs
        Passed to sum
    
    Returns
    -------
    z : torch.Tensor (dtype=torch.int64)
        The Hamming distance between x and y
    """
    
    z = (x^y).sum(*args, **kwargs)
    return z

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
    m, n = queries.shape
    if n != documents.shape[1]:
        msg = 'queries and documents binary codes are not of the same lenght, got {} and {} respectively.'
        raise ValueError(msg.format(n, documents.shape[1]))
    if m != len(relevances):
        msg = 'len(queries) != len(relevances), got {} != {}.'
        raise ValueError(msg.format(m, len(relevances)))
    
    device = queries.device
    #if device != documents.device:
    #    msg = 'queries and documents must be on the same device, but they are on {} and {} respectively.'
    #    raise ValueError(msg.format(device, documents.device))
    
    nbatch = int(np.ceil(m/batch_size))
    total_count = torch.zeros(n+1, device=device, dtype=torch.int64)
    relevant_count = torch.zeros(n+1, device=device, dtype=torch.int64)
    for i in range(nbatch):
        a, b = i*batch_size, (i+1)*batch_size
        dists = hamming_distance(queries[a:b,None], documents[None], dim=2)
        udists, ucounts = torch.unique(dists, return_counts=True)
        total_count[udists] += ucounts
        for dists_row, rel in zip(dists, relevances[a:b]):
            for j in rel:
                k = dists_row[j]
                relevant_count[k] += 1
    tcs = total_count.cumsum(dim=0)
    rcs = relevant_count.cumsum(dim=0)
    relsum = sum(len(s) for s in relevances)
    recalls = rcs.float()/relsum
    dists = torch.where(tcs != 0)[0]
    precisions = (torch.zeros_like(recalls)-1).log() #nans_like
    precisions[dists] = rcs[dists].float()/tcs[dists]
    if return_valid_dists:
        return dists, precisions[dists], recalls[dists]
    return precisions, recalls