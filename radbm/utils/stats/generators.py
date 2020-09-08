import heapq
import numpy as np

class _Subset(object):
    def __init__(self, items, total):
        self.items = items
        self.total = total
    
    def __hash__(self):
        return hash(tuple(self.items))
    
    def __eq__(self, subset):
        return self.items == subset.items
    
    def __lt__(self, subset):
        return self.total < subset.total
    
def _next_subsets_generator(items, total, values):
    if not items or items[0] != 0:
        yield _Subset([0]+items, total + values[0])
    for n, i in enumerate(items[:-1]):
        if i+1 < items[n+1]:
            yield _Subset(
                items[:n]+[i+1]+items[n+1:],
                total - values[i] + values[i+1]
            )
    if items:
        i = items[-1]
        if i+1 < len(values):
            yield _Subset(
                items[:-1]+[i+1],
                total - values[i] + values[i+1]
            )
        
def _least_k_subsortedset_sum_generator(values, k=None):
    if any(v < 0 for v in values): raise ValueError('values must be positive')
    if k is None: k = 2**len(values)
    k = min(k, 2**len(values))
    heap = [_Subset([], 0)]
    produced = set()
    for _ in range(k):
        subset = heapq.heappop(heap)
        yield subset.items
        for new_subset in _next_subsets_generator(subset.items, subset.total, values):
            if new_subset not in produced:
                heapq.heappush(heap, new_subset)
                produced.add(new_subset)

def least_k_subset_sum_generator(values, k=None):
    """
    Generator that yields the subset of index of values in
    increasing order of their sum. The values must be all positive.
    
    Parameters
    ----------
    values : numpy.ndarray
        The values from which to take the subsets
        
    Yields
    ------
    subset : tuple
        Subset of index of the values in increasing order of their sum
    """
    perm = list(np.argsort(values))
    values = values[perm]
    for items in _least_k_subsortedset_sum_generator(values, k):
        yield tuple(perm[i] for i in items)
        
def greatest_k_multi_bernoulli_outcomes_generator(log_probs0, log_probs1=None, k=None):
    """
    Generator that yields the outcomes of a Multi-Bernoulli in decreasing
    order of probability. This work by reducing to the problem of generating
    the subset of a set in increasing order of their sum.
    
    Notes
    -----
    Bits probability must be in ]0,1[ (i.e. they cannot be zero or one)
    
    For numerical stability, it is possible to provide 
    
    Parameters
    ----------
    log_probs0 : numpy.ndarray
        log probabilities for bits to be zero
    log_probs1 : numpy.ndarray, optional
        log probabilities for bits to be one if not given
        log_probs1 = np.log(1-np.exp(log_probs0)) which might be unstable
    k : int, optional
        The maximum number of outcomes to yield, by default all outcomes are yielded
    
    Yields
    ------
        bits : tuple
            The bits outcomes in decreasing order of prabability
    """
    if log_probs1 is None:
        log_probs1 = np.log(1-np.exp(log_probs0))
    diff = log_probs1 - log_probs0
    most_probable_outcome = (0 < diff) + 0
    flipped_diff = np.where(most_probable_outcome, diff, -diff)
    for subset in least_k_subset_sum_generator(flipped_diff, k):
        subset = list(subset)
        bits = most_probable_outcome.copy()
        bits[subset] = 1 - bits[subset]
        yield tuple(bits)