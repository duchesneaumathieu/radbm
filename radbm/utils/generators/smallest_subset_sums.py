import numpy as np
from radbm.search import KeyValueHeap

def _children(subset, subsum, values, produced):
    if not subset or subset[0] != 0: #add the smallest index, e.g., (2,5,6,8) -> (0,2,5,6,8)
        new = (0,) + subset
        if new not in produced:
            produced.add(new)
            yield values[0] + subsum, new
    for n, i in enumerate(subset, 1): #produce all shifts, e.g., (2,5,6,8) -> (3,5,6,8), (2,5,7,8), and (2,5,6,9)
        j = i + 1
        if (n < len(subset) and j < subset[n]) or (n == len(subset) and j < len(values)):
            new = subset[:n-1] + (j,) + subset[n:]
            if new not in produced:
                produced.add(new)
                yield (subsum - values[i] + values[j], new)

def smallest_subset_sums(values, yield_stats=False):
    """
    Generator that yields the subset's index for a set of values in
    increasing order of their sum. The values must be all positive.
    
    Parameters
    ----------
    values : numpy.ndarray (ndim: 1)
        The set values from which to take the subsets
    yield_stats : bool (optional)
        Whether to provide statistic about the search. If True, the function will also yield
        the number of swap, the number of comparison and the size of the heap used to generate
        the subsets. (default: False)
        
    Yields
    ------
    subset : list of int
        Subset of index of the values in increasing order of their sum.
    swap_count : int (if yield_stats is True)
        The number of swap done to maintain the heap while generating this subset.
    comp_count : int (if yield_stats is True)
        The number of comparison done to maintain the heap while generating this subset.
    heap_size : int (if yield_stats is True)
        The current size of the heap (before adding the children of the yielded subset).
    """
    if (values<0).any():
        raise ValueError('values must be positive')
        
    #handling unsorted values
    perm = np.argsort(values)
    values = values[perm] #sort them first
        
    heap = KeyValueHeap((0, ()), return_counts=True, key=lambda x: x[0])
    produced, push_sc, push_cc = set(), 0, 0
    while heap:
        (subsum, subset), pop_sc, pop_cc = heap.pop()
        unsorted_subset = [perm[i] for i in subset] #unsort them after and make it a list.
        yield (unsorted_subset, pop_sc+push_sc, pop_cc+push_cc, len(heap)) if yield_stats else unsorted_subset
        push_sc, push_cc = heap.batch_insert(_children(subset, subsum, values, produced))