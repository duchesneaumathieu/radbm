import torch
import numpy as np

def conditional_distance_counts(documents, queries, relevances, distance_function, max_distance, batch_size=100):
    """
    Compute the count of relevant and not relevant match w.r.t all distance.
    
    Parameters
    ----------
    documents : torch.Tensor
        The code of a batch of documents (database).
        documents[i] is the ith document.
    queries : torch.Tensor (2D, dtype=torch.bool)
        The code of a batch of queries. queries[i]
        is the ith query. 
    relevances : list of set of int
        len(relevances) must be len(queries). For each corresponding query,
        it give the set of relevant documents (given by its index). Explicitly,
        j in relevances[i] iff query[i] matches with documents[j].
    distance_function : function (torch.Tensor, torch.Tensor) -> torch.Tensor (dtype: int)
        The distance should be integer only.
    max_distance: int
        The maximum distance to expect.
    batch_size : int (optional)
        The number of query for which we compute the distance at a time.
        if it is to big the results might not fit in RAM (or on the GPU). (default 100)
        
    Returns
    -------
    relevant_counts : torch.Tensor (1D, dtype=torch.float)
        len(relevant_counts) = maximum_distance + 1 and
        relevant_dcounts[i] is the number of relevant documents at distance i.
    irrelevant_counts : torch.Tensor (1D, dtype=torch.float)
        len(irrelevant_countst) = maximum_distance + 1 and
        irrelevant_counts[i] is the number of irrelevant documents at distance i.
        
    Notes
    -----
    This assume that each sets in relevances is small compared to len(documents) and should
    be used on a GPU otherwise it is quite slow.
    """
    m, n = queries.shape
    if m != len(relevances):
        msg = 'len(queries) != len(relevances), got {} != {}.'
        raise ValueError(msg.format(m, len(relevances)))
    
    device = queries.device
    nbatch = int(np.ceil(m/batch_size))
    total_counts = torch.zeros(max_distance+1, device=device, dtype=torch.int64)
    relevant_counts = torch.zeros(max_distance+1, device=device, dtype=torch.int64)
    for i in range(nbatch):
        a, b = i*batch_size, (i+1)*batch_size
        dists = distance_function(queries[a:b,None], documents[None])
        
        #update total_count
        udists, ucounts = torch.unique(dists, return_counts=True)
        total_counts[udists] += ucounts
        
        #update relevant_count
        rows, cols = zip(*((i,j) for i, rel in enumerate(relevances[a:b]) for j in rel))
        udists, ucounts = torch.unique(dists[(rows,cols)], return_counts=True)
        relevant_counts[udists] += ucounts
    
    irrelevant_counts = total_counts - relevant_counts
    return relevant_counts, irrelevant_counts